import rakun_python as rk


@rk.Agent
class PositionMangeAgent:

    async def run(self):
        pass

    async def receiver(self, sender, message):
        print(self.code, sender, message.data)
