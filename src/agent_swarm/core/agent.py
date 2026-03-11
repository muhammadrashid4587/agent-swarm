"""Base Agent with lifecycle management, tool use, and memory."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    INITIALIZING = "initializing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@runtime_checkable
class Tool(Protocol):
    """Protocol for agent tools."""
    name: str
    description: str

    async def execute(self, **kwargs: Any) -> Any: ...


@dataclass
class Memory:
    """Agent memory store."""
    working: list[dict[str, Any]] = field(default_factory=list)
    long_term: list[dict[str, Any]] = field(default_factory=list)
    max_working: int = 50

    def add(self, entry: dict[str, Any], persistent: bool = False) -> None:
        self.working.append(entry)
        if len(self.working) > self.max_working:
            self.working.pop(0)
        if persistent:
            self.long_term.append(entry)

    def search(self, query: str) -> list[dict[str, Any]]:
        results = []
        for entry in self.working + self.long_term:
            content = str(entry.get("content", "")).lower()
            if query.lower() in content:
                results.append(entry)
        return results


class BaseAgent(ABC):
    """Base agent with full lifecycle: init -> plan -> execute -> reflect."""

    def __init__(self, role: str = "general", model: str = "claude-sonnet-4-6"):
        self.agent_id: str = f"{role}-{uuid4().hex[:8]}"
        self.role: str = role
        self.model: str = model
        self.status: AgentStatus = AgentStatus.IDLE
        self.memory: Memory = Memory()
        self.tools: dict[str, Tool] = {}
        self.tasks_completed: int = 0
        self._message_handlers: dict[str, Any] = {}

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    async def initialize(self) -> None:
        """Initialize agent resources."""
        self.status = AgentStatus.INITIALIZING
        await self._setup()
        self.status = AgentStatus.IDLE

    async def _setup(self) -> None:
        """Override for custom initialization."""
        pass

    @abstractmethod
    async def plan(self, task: str) -> Any:
        """Create an execution plan for the task."""
        ...

    @abstractmethod
    async def execute(self, plan: Any) -> Any:
        """Execute the plan and return results."""
        ...

    async def reflect(self, result: Any) -> dict[str, Any]:
        """Reflect on execution results for self-improvement."""
        reflection = {
            "agent_id": self.agent_id,
            "success": result is not None,
            "summary": str(result)[:200] if result else "No result",
        }
        self.memory.add(reflection, persistent=True)
        self.tasks_completed += 1
        return reflection

    async def handle_message(self, message: Any) -> None:
        """Handle incoming messages from other agents."""
        topic = getattr(message, "topic", "")
        handler = self._message_handlers.get(topic)
        if handler:
            await handler(message)
        else:
            self.memory.add({"type": "message", "content": message})

    async def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a registered tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
        return await tool.execute(**kwargs)

    async def send_message(self, bus: Any, target: str, content: Any) -> None:
        """Send a message to another agent via the message bus."""
        from agent_swarm.core.message_bus import Message
        msg = Message(topic=f"agent.{target}", content=content, sender=self.agent_id)
        await bus.publish(msg)
