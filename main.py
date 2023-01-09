import asyncio
import rakun_python as rk

import agents
from agents.angular_velocity_agent import AngularVelocityAgent
from agents.cordinating_agent import CoordinatorAgent
from agents.lander_position_agent import PositionMangeAgent
from agents.linear_velocity_agent import LinearVelocityAgent
from agents.state_agent import LegStateAgent
from agents.world_simulator_agent import WorldSensing


async def main():
    agent_manager = rk.AgentManager()
    agent_manager.register(WorldSensing, agents.Agent.World)
    agent_manager.register(CoordinatorAgent, agents.Agent.CoordinatorAgent)
    agent_manager.register(LinearVelocityAgent, agents.Agent.LinearVelocityAgent)
    agent_manager.register(AngularVelocityAgent, agents.Agent.AngularVelocityAgent)
    agent_manager.register(PositionMangeAgent, agents.Agent.PositionMangeAgent)
    agent_manager.register(LegStateAgent, agents.Agent.LegStateAgent)
    await agent_manager.start()


if __name__ == "__main__":
    asyncio.run(main())
