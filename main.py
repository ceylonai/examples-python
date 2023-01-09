import asyncio
import enum
import random

import gym
import rakun_python as rk
from gym import wrappers


class WorldState(enum.Enum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@rk.Agent
class WorldSensing:

    def __init__(self):
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        self.env = wrappers.record_video.RecordVideo(env, "video")
        self.state = WorldState.IDLE
        self.next_action = None
        self.episode_images = []
        self.stop_training = False

    async def run(self):
        while not self.stop_training:
            if self.state == WorldState.IDLE:
                self.state = WorldState.RUNNING
                observation, info = self.env.reset()
                msg = {
                    "receiver": "sensor",
                    "data": {
                        "begin_episode": True,
                        "observation": observation.tolist(),
                        "reward": 0,
                        "terminated": False,
                        "info": info
                    }
                }
                await self.core.send(msg)

            while self.state == WorldState.RUNNING:
                if self.next_action is not None:
                    self.env.unwrapped.render()
                    observation, reward, terminated, truncated, info = self.env.step(self.next_action)
                    msg = {
                        "receiver": "sensor",
                        "data": {
                            "begin_episode": False,
                            "observation": observation.tolist(),
                            "reward": reward,
                            "terminated": terminated or truncated,
                            "info": info
                        }
                    }
                    await self.core.send(msg)
                    if terminated or truncated:
                        self.state = WorldState.IDLE
                    self.next_action = None

    async def receiver(self, sender, message):
        if "action" in message.data:
            action = message.data["action"]
            self.next_action = action
        elif "terminate" in message.data:
            self.stop_training = True
            self.env.close()


@rk.Agent
class AgentSensor:
    state = []
    rewards = []
    episode_index = 0
    steps = 0

    async def run(self):
        pass

    async def receiver(self, sender, message):
        self.state = message.data
        begin_episode = message.data["begin_episode"]
        terminated = message.data["terminated"]
        reward = message.data["reward"]

        if begin_episode:
            self.rewards = []
            self.steps = 0
            self.episode_index += 1

        if terminated:
            self.state = []
            avg_reward = sum(self.rewards) / len(self.rewards)
            print(f"EP.{self.episode_index} Terminated with avg reward: {avg_reward}")
            if avg_reward > 1:
                self.core.send({"receiver": "world", "data": {"terminate": True}})
                self.core.exit()
        else:
            self.rewards.append(reward)
            self.steps += 1
            self.core.send({"receiver": "world", "data": {"action": random.randint(0, 3)}})
            if self.steps % 20 == 0:
                print(f"EP.{self.episode_index}-{self.steps} reward: {reward}")


async def main():
    agent_manager = rk.AgentManager()
    agent_manager.register(WorldSensing, "world")
    agent_manager.register(AgentSensor, "sensor")
    await agent_manager.start()


if __name__ == "__main__":
    asyncio.run(main())
