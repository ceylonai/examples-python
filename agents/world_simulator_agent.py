import enum

import gym
import rakun_python as rk
from gym import wrappers

from agents import Agent


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
                    "receiver": Agent.CoordinatorAgent,
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
                        "receiver": Agent.CoordinatorAgent,
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
