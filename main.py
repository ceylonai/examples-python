import asyncio
import rakun_python as rk

from agents.cordinating_agent import AgentSensor
from agents.world_simulator_agent import WorldSensing


async def main():
    agent_manager = rk.AgentManager()
    agent_manager.register(WorldSensing, "world")
    agent_manager.register(AgentSensor, "sensor")
    await agent_manager.start()


if __name__ == "__main__":
    asyncio.run(main())
