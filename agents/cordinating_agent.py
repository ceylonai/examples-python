import rakun_python as rk


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
                self.core.send({"receiver": "world", "data": {"terminate": True}})
                self.core.exit()
        else:
            self.rewards.append(reward)
            self.steps += 1
            action = await self.process(observation)
            self.core.send({"receiver": "world", "data": {"action": action}})
            if self.steps % 20 == 0:
                print(f"EP.{self.episode_index}-{self.steps} reward: {reward}")

    async def process(self, state):
        print(f"Processing state: {state}")
        return random.randint(0, 3)
