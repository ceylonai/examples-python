import rakun_python as rk


@rk.Agent
class LinearVelocityAgent:
    async def run(self):
        pass

    async def receiver(self, sender, message):
        print(self.code, sender, message.data)
