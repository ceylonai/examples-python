import random
from collections import namedtuple

import rakun_python as rk

from agents import Agent


@rk.Agent
class CoordinatorAgent:
    world_state = []
    rewards = []
    episode_index = 0
    steps = 0

    agents_response = (None, None, None, None)

    async def run(self):
        pass

    async def action(self):
        while True:
            ag_res = self.agents_response
            leg_state_idea = ag_res[0]
            position_idea = ag_res[1]
            linear_velocity_idea = ag_res[2]
            angular_velocity_idea = ag_res[3]
            if leg_state_idea is not None \
                    and position_idea is not None \
                    and linear_velocity_idea is not None \
                    and angular_velocity_idea is not None:
                return random.randint(0, 3)

    async def receiver(self, sender, message):
        print(f"Coordinator received message from {sender}")
        if sender == Agent.World:
            self.agents_response = (None, None, None, None)
            await self.process_world_messages(message)
        elif sender == Agent.LegStateAgent:
            self.agents_response = (message.data, *self.agents_response[1:])
        elif sender == Agent.PositionMangeAgent:
            self.agents_response = (*self.agents_response[:1], message.data, *self.agents_response[2:])
        elif sender == Agent.LinearVelocityAgent:
            self.agents_response = (*self.agents_response[:2], message.data, *self.agents_response[3:])
        elif sender == Agent.AngularVelocityAgent:
            self.agents_response = (*self.agents_response[:3], message.data)

    async def process_world_messages(self, message):
        self.world_state = message.data
        begin_episode = message.data["begin_episode"]
        terminated = message.data["terminated"]
        observation = message.data["observation"]
        reward = message.data["reward"]

        if begin_episode:
            self.rewards = []
            self.steps = 0
            self.episode_index += 1

        if terminated:
            self.world_state = []
            avg_reward = sum(self.rewards) / len(self.rewards)
            print(f"EP.{self.episode_index} Terminated with avg reward: {avg_reward}")
            if avg_reward > 1:
                await self.core.send({"receiver": Agent.World, "data": {"terminate": True}})
                self.core.exit()
        else:
            self.rewards.append(reward)
            self.steps += 1
            await self.distribute_state(observation)

            action = await self.action()
            await self.core.send({"receiver": Agent.World, "data": {"action": action}})
            print(f"EP.{self.episode_index}-{self.steps} reward: {reward}")

    async def distribute_state(self, state):
        lander_position = state[0:2]
        linear_velocity = state[2:4]
        angular_velocity = state[4:6]
        leg_status = state[6:8]

        await self.core.send({"receiver": Agent.LegStateAgent, "data": {"state": leg_status}})
        await self.core.send({"receiver": Agent.PositionMangeAgent, "data": {"state": lander_position}})
        await self.core.send({"receiver": Agent.LinearVelocityAgent, "data": {"state": linear_velocity}})
        await self.core.send({"receiver": Agent.AngularVelocityAgent, "data": {"state": angular_velocity}})
