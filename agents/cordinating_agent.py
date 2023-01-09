import random

import rakun_python as rk

from agents import Agent


@rk.Agent
class CoordinatorAgent:
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
        observation = message.data["observation"]
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
                await self.core.send({"receiver": Agent.World, "data": {"terminate": True}})
                self.core.exit()
        else:
            self.rewards.append(reward)
            self.steps += 1
            action = await self.process(observation)
            await self.core.send({"receiver": Agent.World, "data": {"action": action}})
            if self.steps % 20 == 0:
                print(f"EP.{self.episode_index}-{self.steps} reward: {reward}")

    async def process(self, state):
        lander_position = state[0:2]
        linear_velocity = state[2:4]
        angular_velocity = state[4:6]
        leg_status = state[6:8]

        await self.core.send({"receiver": Agent.LegStateAgent, "data": {"state": leg_status}})
        await self.core.send({"receiver": Agent.PositionMangeAgent, "data": {"state": lander_position}})
        await self.core.send({"receiver": Agent.LinearVelocityAgent, "data": {"state": linear_velocity}})
        await self.core.send({"receiver": Agent.AngularVelocityAgent, "data": {"state": angular_velocity}})

        return random.randint(0, 3)
