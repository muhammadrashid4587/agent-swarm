"""Async message bus for inter-agent communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

logger = logging.getLogger(__name__)

Subscriber = Callable[["Message"], Awaitable[None]]


@dataclass
class Message:
    topic: str
    content: Any
    sender: str = "system"
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Async pub/sub message bus for agent-to-agent communication."""

    def __init__(self, max_queue_size: int = 1000):
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._wildcard_subscribers: list[Subscriber] = []
        self._dead_letter: list[Message] = []
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._history: list[Message] = []
        self._max_history = 500

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        """Subscribe to messages on a topic."""
        if topic == "*":
            self._wildcard_subscribers.append(handler)
        else:
            self._subscribers[topic].append(handler)
        logger.debug(f"Subscriber added for topic: {topic}")

    def unsubscribe(self, topic: str, handler: Subscriber) -> None:
        """Unsubscribe from a topic."""
        if topic == "*":
            self._wildcard_subscribers.remove(handler)
        elif handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)

    async def publish(self, message: Message) -> None:
        """Publish a message to all subscribers of its topic."""
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        exact_handlers = self._subscribers.get(message.topic, [])
        handlers = list(exact_handlers) + list(self._wildcard_subscribers)

        # Also match prefix patterns (e.g., "agent.*" matches "agent.abc123")
        for pattern, subs in self._subscribers.items():
            if pattern == message.topic:
                continue  # Already included from exact match above
            if pattern.endswith(".*") and message.topic.startswith(pattern[:-2]):
                handlers.extend(subs)

        if not handlers:
            self._dead_letter.append(message)
            logger.warning(f"No subscribers for topic '{message.topic}', sent to dead letter")
            return

        tasks = [self._safe_deliver(handler, message) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_deliver(self, handler: Subscriber, message: Message) -> None:
        """Safely deliver a message to a handler."""
        try:
            await handler(message)
        except Exception as e:
            logger.error(f"Handler failed for message {message.id}: {e}")
            self._dead_letter.append(message)

    async def request_reply(self, topic: str, content: Any, sender: str, timeout: float = 30.0) -> Any:
        """Send a request and wait for a reply (RPC pattern)."""
        reply_topic = f"reply.{uuid4().hex[:8]}"
        future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

        async def reply_handler(msg: Message) -> None:
            if not future.done():
                future.set_result(msg)

        self.subscribe(reply_topic, reply_handler)
        try:
            request = Message(
                topic=topic, content=content, sender=sender,
                metadata={"reply_to": reply_topic},
            )
            await self.publish(request)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.unsubscribe(reply_topic, reply_handler)

    def get_dead_letters(self) -> list[Message]:
        return list(self._dead_letter)

    def get_history(self, topic: str | None = None, limit: int = 50) -> list[Message]:
        msgs = self._history if not topic else [m for m in self._history if m.topic == topic]
        return msgs[-limit:]
