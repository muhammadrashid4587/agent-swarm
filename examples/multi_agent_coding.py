"""Example: Multi-agent collaboration on a coding task."""

import asyncio

from agent_swarm import Orchestrator, CoderAgent, ResearcherAgent, PlannerAgent


async def main():
    # Create orchestrator
    orchestrator = Orchestrator()

    # Register specialized agents
    orchestrator.register(PlannerAgent())
    orchestrator.register(CoderAgent())
    orchestrator.register(ResearcherAgent())

    print("=== Agent Swarm: Multi-Agent Coding ===\n")
    print(f"Registered agents: {list(orchestrator.agents.keys())}\n")

    # Execute a complex task — the planner will decompose it,
    # and the coder/researcher will handle subtasks in parallel
    task = "Build a REST API for a task management app with user authentication, CRUD operations, and rate limiting"

    print(f"Task: {task}\n")
    print("Executing with agent swarm...\n")

    result = await orchestrator.execute(task)

    print("=== Results ===")
    if isinstance(result, dict):
        for step, output in result.items():
            print(f"\n--- {step} ---")
            print(str(output)[:500])
    else:
        print(result)

    print("\n=== Swarm Status ===")
    print(orchestrator.get_status())


if __name__ == "__main__":
    asyncio.run(main())
