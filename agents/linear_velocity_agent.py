import rakun_python as rk

from agents import Agent


@rk.Agent
class LinearVelocityAgent:
    async def run(self):
        pass

    async def receiver(self, sender, message):
        print(self.code, sender, message.data)
        self.core.send({"receiver": Agent.CoordinatorAgent, "data": 0.1})
