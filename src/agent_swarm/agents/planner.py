"""Planning agent that decomposes complex tasks into subtasks."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from agent_swarm.core.agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Agent specialized in task decomposition and workflow planning."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        super().__init__(role="planner", model=model)
        self._client: anthropic.AsyncAnthropic | None = None

    async def _setup(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def plan(self, task: str) -> list[dict[str, Any]]:
        """Decompose a complex task into ordered subtasks with role assignments."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=(
                "You are a project planner. Decompose the task into subtasks. "
                "Return a JSON array where each item has: "
                '{"step": number, "task": "description", "role": "coder|researcher|planner", '
                '"depends_on": [step_numbers], "estimated_complexity": "low|medium|high"}'
            ),
            messages=[{"role": "user", "content": f"Decompose this task: {task}"}],
        )

        text = response.content[0].text
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            subtasks = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            subtasks = [{"step": 1, "task": task, "role": "coder", "depends_on": [], "estimated_complexity": "high"}]

        self.memory.add({"type": "plan", "task": task, "subtask_count": len(subtasks)})
        return subtasks

    async def execute(self, plan: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate and optimize the execution plan."""
        optimized = sorted(plan, key=lambda x: (len(x.get("depends_on", [])), x.get("step", 0)))
        parallel_groups: list[list[dict[str, Any]]] = []
        completed_steps: set[int] = set()

        for subtask in optimized:
            deps = set(subtask.get("depends_on", []))
            placed = False
            for group in parallel_groups:
                group_steps = {t["step"] for t in group}
                if deps.issubset(completed_steps - group_steps):
                    group.append(subtask)
                    placed = True
                    break
            if not placed:
                parallel_groups.append([subtask])
            completed_steps.add(subtask["step"])

        return {
            "plan": plan,
            "parallel_groups": parallel_groups,
            "total_steps": len(plan),
            "parallel_batches": len(parallel_groups),
        }
