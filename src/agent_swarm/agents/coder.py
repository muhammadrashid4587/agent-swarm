"""Coding agent that writes, reviews, and refactors code."""

from __future__ import annotations

from typing import Any

import anthropic

from agent_swarm.core.agent import BaseAgent


class CoderAgent(BaseAgent):
    """Agent specialized in code generation, review, and refactoring."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        super().__init__(role="coder", model=model)
        self._client: anthropic.AsyncAnthropic | None = None

    async def _setup(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def plan(self, task: str) -> dict[str, Any]:
        """Break down a coding task into implementation steps."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are an expert software engineer. Break down the coding task into clear implementation steps. Return a JSON plan.",
            messages=[{"role": "user", "content": f"Plan the implementation for: {task}"}],
        )
        return {"task": task, "plan": response.content[0].text, "agent_id": self.agent_id}

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Generate code based on the plan."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="You are an expert software engineer. Write clean, well-documented, production-quality code.",
            messages=[
                {"role": "user", "content": f"Implement the following plan:\n{plan['plan']}\n\nOriginal task: {plan['task']}"},
            ],
        )
        code = response.content[0].text
        self.memory.add({"type": "generation", "task": plan["task"], "code_length": len(code)})
        return {"code": code, "task": plan["task"], "agent_id": self.agent_id}

    async def review(self, code: str) -> dict[str, Any]:
        """Review code for bugs, security issues, and improvements."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are a senior code reviewer. Find bugs, security issues, performance problems, and suggest improvements.",
            messages=[{"role": "user", "content": f"Review this code:\n```\n{code}\n```"}],
        )
        return {"review": response.content[0].text, "agent_id": self.agent_id}

    async def refactor(self, code: str, instructions: str = "") -> dict[str, Any]:
        """Refactor code for better structure and readability."""
        if not self._client:
            await self._setup()

        prompt = f"Refactor this code for better quality:\n```\n{code}\n```"
        if instructions:
            prompt += f"\n\nSpecific instructions: {instructions}"

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="You are an expert at code refactoring. Improve structure, readability, and maintainability.",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"refactored_code": response.content[0].text, "agent_id": self.agent_id}
