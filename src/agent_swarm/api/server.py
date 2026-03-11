"""FastAPI server for swarm management and monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_swarm.core.orchestrator import Orchestrator, SwarmConfig, TaskPriority

app = FastAPI(title="Agent Swarm API", version="0.1.0")
orchestrator = Orchestrator(SwarmConfig())


class TaskRequest(BaseModel):
    description: str
    priority: str = "normal"

    def get_priority(self) -> TaskPriority:
        return TaskPriority[self.priority.upper()]


class WorkflowRequest(BaseModel):
    name: str
    steps: list[dict[str, Any]]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/status")
async def status() -> dict[str, Any]:
    return orchestrator.get_status()


@app.post("/tasks")
async def create_task(request: TaskRequest) -> dict[str, Any]:
    task = await orchestrator.submit_task(request.description, request.get_priority())
    return {"task_id": task.id, "status": task.status.value}


@app.post("/execute")
async def execute_task(request: TaskRequest) -> dict[str, Any]:
    try:
        result = await orchestrator.execute(request.description)
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents() -> list[dict[str, Any]]:
    return [
        {
            "id": agent.agent_id,
            "role": agent.role,
            "status": agent.status.value,
            "tasks_completed": agent.tasks_completed,
        }
        for agent in orchestrator.agents.values()
    ]


@app.get("/messages")
async def get_messages(topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    messages = orchestrator.message_bus.get_history(topic=topic, limit=limit)
    return [
        {"id": m.id, "topic": m.topic, "sender": m.sender, "timestamp": m.timestamp.isoformat()}
        for m in messages
    ]
