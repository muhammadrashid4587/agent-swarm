"""Research agent that searches and synthesizes information."""

from __future__ import annotations

from typing import Any

import anthropic

from agent_swarm.core.agent import BaseAgent


class ResearcherAgent(BaseAgent):
    """Agent specialized in research, analysis, and information synthesis."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        super().__init__(role="researcher", model=model)
        self._client: anthropic.AsyncAnthropic | None = None

    async def _setup(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def plan(self, task: str) -> dict[str, Any]:
        """Create a research plan with search strategies."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=(
                "You are a research strategist. Break down the research question into sub-questions, "
                "identify key search terms, and plan the research methodology."
            ),
            messages=[{"role": "user", "content": f"Create a research plan for: {task}"}],
        )
        return {"task": task, "plan": response.content[0].text, "agent_id": self.agent_id}

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Execute research and synthesize findings."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=(
                "You are an expert researcher. Synthesize information comprehensively, "
                "cite your reasoning, and provide actionable insights."
            ),
            messages=[
                {"role": "user", "content": f"Execute this research plan and provide findings:\n{plan['plan']}\n\nOriginal question: {plan['task']}"},
            ],
        )
        findings = response.content[0].text
        self.memory.add({"type": "research", "task": plan["task"], "findings_length": len(findings)}, persistent=True)
        return {"findings": findings, "task": plan["task"], "agent_id": self.agent_id}

    async def analyze(self, data: str, question: str) -> dict[str, Any]:
        """Analyze data to answer a specific question."""
        if not self._client:
            await self._setup()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are a data analyst. Provide clear, evidence-based analysis.",
            messages=[
                {"role": "user", "content": f"Analyze this data to answer the question.\n\nData:\n{data}\n\nQuestion: {question}"},
            ],
        )
        return {"analysis": response.content[0].text, "agent_id": self.agent_id}
