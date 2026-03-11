"""Main orchestrator for managing multi-agent swarms."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from agent_swarm.core.agent import BaseAgent, AgentStatus
from agent_swarm.core.message_bus import MessageBus, Message
from agent_swarm.workflows.dag import DAGWorkflow, WorkflowNode

logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str | None = None
    result: Any = None
    error: str | None = None
    retries: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


class SwarmConfig(BaseModel):
    max_concurrent_agents: int = 10
    task_timeout: float = 300.0
    enable_load_balancing: bool = True
    retry_delay: float = 1.0


class Orchestrator:
    """Coordinates multiple AI agents to complete complex tasks."""

    def __init__(self, config: SwarmConfig | None = None):
        self.config = config or SwarmConfig()
        self.agents: dict[str, BaseAgent] = {}
        self.agent_pools: dict[str, list[BaseAgent]] = {}
        self.message_bus = MessageBus()
        self.task_queue: asyncio.PriorityQueue[tuple[int, Task]] = asyncio.PriorityQueue()
        self.active_tasks: dict[str, Task] = {}
        self._running = False
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        role = agent.role
        if role not in self.agent_pools:
            self.agent_pools[role] = []
        self.agent_pools[role].append(agent)
        self.message_bus.subscribe(f"agent.{agent.agent_id}", agent.handle_message)
        logger.info(f"Registered agent {agent.agent_id} with role {role}")

    def _select_agent(self, role: str) -> BaseAgent | None:
        """Select the best available agent for a role using load balancing."""
        pool = self.agent_pools.get(role, [])
        available = [a for a in pool if a.status == AgentStatus.IDLE]
        if not available:
            return None
        if self.config.enable_load_balancing:
            return min(available, key=lambda a: a.tasks_completed)
        return available[0]

    async def submit_task(self, description: str, priority: TaskPriority = TaskPriority.NORMAL, **metadata: Any) -> Task:
        """Submit a task to the swarm."""
        task = Task(description=description, priority=priority, metadata=metadata)
        await self.task_queue.put((-priority.value, task))
        self.active_tasks[task.id] = task
        logger.info(f"Task {task.id} submitted: {description[:80]}")
        return task

    async def _execute_task(self, task: Task, agent: BaseAgent) -> Any:
        """Execute a single task with an agent."""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.assigned_agent = agent.agent_id
            agent.status = AgentStatus.BUSY

            try:
                await agent.initialize()
                plan = await agent.plan(task.description)
                result = await asyncio.wait_for(
                    agent.execute(plan),
                    timeout=self.config.task_timeout,
                )
                reflection = await agent.reflect(result)
                task.status = TaskStatus.COMPLETED
                task.result = {"output": result, "reflection": reflection}
                logger.info(f"Task {task.id} completed by {agent.agent_id}")
                return result

            except asyncio.TimeoutError:
                task.error = "Task timed out"
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.id} timed out")
            except Exception as e:
                task.error = str(e)
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.status = TaskStatus.RETRYING
                    logger.warning(f"Task {task.id} failed, retrying ({task.retries}/{task.max_retries})")
                    await asyncio.sleep(self.config.retry_delay)
                    return await self._execute_task(task, agent)
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.id} failed permanently: {e}")
            finally:
                agent.status = AgentStatus.IDLE
        return None

    async def execute(self, description: str, **kwargs: Any) -> Any:
        """Execute a complex task by decomposing and distributing to agents."""
        planner = self._select_agent("planner")
        if planner:
            await planner.initialize()
            subtasks = await planner.plan(description)

            if isinstance(subtasks, list):
                workflow = DAGWorkflow(name=f"workflow-{uuid4().hex[:8]}")
                nodes: list[WorkflowNode] = []
                for i, subtask in enumerate(subtasks):
                    node = WorkflowNode(
                        id=f"step-{i}",
                        task=str(subtask),
                        agent_role=subtask.get("role", "coder") if isinstance(subtask, dict) else "coder",
                    )
                    nodes.append(node)
                    workflow.add_node(node)
                    if i > 0:
                        workflow.add_edge(nodes[i - 1].id, node.id)
                return await self._execute_workflow(workflow)

        agent = self._select_agent("coder") or next(iter(self.agents.values()), None)
        if not agent:
            raise RuntimeError("No agents registered")

        task = Task(description=description)
        return await self._execute_task(task, agent)

    async def _execute_workflow(self, workflow: DAGWorkflow) -> dict[str, Any]:
        """Execute a DAG workflow."""
        results: dict[str, Any] = {}
        execution_order = workflow.topological_sort()

        for batch in execution_order:
            coros = []
            for node_id in batch:
                node = workflow.nodes[node_id]
                agent = self._select_agent(node.agent_role)
                if agent:
                    task = Task(description=node.task)
                    coros.append(self._execute_task(task, agent))
            batch_results = await asyncio.gather(*coros, return_exceptions=True)
            for node_id, result in zip(batch, batch_results):
                results[node_id] = result

        return results

    async def broadcast(self, topic: str, content: Any) -> None:
        """Broadcast a message to all agents on a topic."""
        msg = Message(topic=topic, content=content, sender="orchestrator")
        await self.message_bus.publish(msg)

    def get_status(self) -> dict[str, Any]:
        """Get current swarm status."""
        return {
            "agents": {aid: a.status.value for aid, a in self.agents.items()},
            "active_tasks": len(self.active_tasks),
            "completed": sum(1 for t in self.active_tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.active_tasks.values() if t.status == TaskStatus.FAILED),
        }
