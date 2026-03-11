"""Agent Swarm — Multi-Agent Orchestration Framework."""

__version__ = "0.1.0"

from agent_swarm.core.orchestrator import Orchestrator
from agent_swarm.core.agent import BaseAgent
from agent_swarm.core.message_bus import MessageBus
from agent_swarm.agents.coder import CoderAgent
from agent_swarm.agents.researcher import ResearcherAgent
from agent_swarm.agents.planner import PlannerAgent

__all__ = [
    "Orchestrator",
    "BaseAgent",
    "MessageBus",
    "CoderAgent",
    "ResearcherAgent",
    "PlannerAgent",
]
