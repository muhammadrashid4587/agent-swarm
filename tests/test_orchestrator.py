"""Tests for the orchestrator and workflow engine."""

import asyncio
import pytest
from agent_swarm.core.orchestrator import Orchestrator, SwarmConfig, Task, TaskPriority, TaskStatus
from agent_swarm.core.message_bus import MessageBus, Message
from agent_swarm.core.agent import BaseAgent, AgentStatus
from agent_swarm.workflows.dag import DAGWorkflow, WorkflowNode


class MockAgent(BaseAgent):
    async def plan(self, task: str):
        return {"task": task}

    async def execute(self, plan):
        return f"Completed: {plan['task']}"


class TestOrchestrator:
    def test_register_agent(self):
        orch = Orchestrator()
        agent = MockAgent(role="coder")
        orch.register(agent)
        assert agent.agent_id in orch.agents
        assert "coder" in orch.agent_pools

    def test_swarm_config(self):
        config = SwarmConfig(max_concurrent_agents=5, task_timeout=60.0)
        orch = Orchestrator(config=config)
        assert orch.config.max_concurrent_agents == 5

    def test_get_status(self):
        orch = Orchestrator()
        agent = MockAgent(role="coder")
        orch.register(agent)
        status = orch.get_status()
        assert "agents" in status
        assert agent.agent_id in status["agents"]


class TestDAGWorkflow:
    def test_add_nodes(self):
        wf = DAGWorkflow(name="test")
        wf.add_node(WorkflowNode(id="a", task="Task A"))
        wf.add_node(WorkflowNode(id="b", task="Task B"))
        assert len(wf.nodes) == 2

    def test_topological_sort(self):
        wf = DAGWorkflow(name="test")
        wf.add_node(WorkflowNode(id="a", task="Task A"))
        wf.add_node(WorkflowNode(id="b", task="Task B"))
        wf.add_node(WorkflowNode(id="c", task="Task C"))
        wf.add_edge("a", "b")
        wf.add_edge("a", "c")
        batches = wf.topological_sort()
        assert batches[0] == ["a"]
        assert set(batches[1]) == {"b", "c"}

    def test_cycle_detection(self):
        wf = DAGWorkflow(name="test")
        wf.add_node(WorkflowNode(id="a", task="A"))
        wf.add_node(WorkflowNode(id="b", task="B"))
        wf.add_edge("a", "b")
        with pytest.raises(ValueError, match="cycle"):
            wf.add_edge("b", "a")


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = MessageBus()
        received = []

        async def handler(msg: Message):
            received.append(msg)

        bus.subscribe("test.topic", handler)
        await bus.publish(Message(topic="test.topic", content="hello"))
        assert len(received) == 1
        assert received[0].content == "hello"

    @pytest.mark.asyncio
    async def test_dead_letter(self):
        bus = MessageBus()
        await bus.publish(Message(topic="no.subscribers", content="lost"))
        assert len(bus.get_dead_letters()) == 1
