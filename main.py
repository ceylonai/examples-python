import asyncio
import enum
import random

import gym
import rakun_python as rk
import matplotlib.pyplot as plt


class WorldState(enum.Enum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@rk.Agent
class WorldSensing:

    def __init__(self):
        self.env = gym.make("LunarLander-v2")
        self.state = WorldState.IDLE
        self.next_action = None

    async def run(self):
        if self.state == WorldState.IDLE:
            self.state = WorldState.RUNNING
            observation, info = self.env.reset()
            msg = {
                "receiver": "sensor",
                "data": {
                    "observation": observation.tolist(),
                    "reward": 0,
                    "terminated": False,
                    "info": info
                }
            }
            await self.core.message(msg)

        if self.state == WorldState.RUNNING:
            while True:
                if self.next_action is not None:
                    observation, reward, terminated, truncated, info = self.env.step(self.next_action)
                    msg = {
                        "receiver": "sensor",
                        "data": {
                            "observation": observation.tolist(),
                            "reward": reward,
                            "terminated": terminated or truncated,
                            "info": info
                        }
                    }
                    await self.core.message(msg)
                    self.next_action = None

    async def receiver(self, sender, message):
        action = message.data["action"]
        self.next_action = action


@rk.Agent
class AgentSensor:
    state = []
    rewards = []

    async def run(self):
        pass

    async def receiver(self, sender, message):
        self.state = message.data

        terminated = message.data["terminated"]
        reward = message.data["reward"]
        if terminated:
            self.state = []
            avg_reward = sum(self.rewards) / len(self.rewards)
            print("Terminated with avg reward: {}".format(avg_reward))
        else:
            self.rewards.append(reward)
            print("reward: {}".format(reward))
            self.core.message({"receiver": "world", "data": {"action": random.randint(0, 3)}})


async def main():
    agent_manager = rk.AgentManager()
    agent_manager.register(WorldSensing, "world")
    agent_manager.register(AgentSensor, "sensor")
    await agent_manager.start()


if __name__ == "__main__":
    asyncio.run(main())
