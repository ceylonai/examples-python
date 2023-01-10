import rakun_python as rk

from agents import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionModel(nn.Module):

    def __init__(self):
        super(PositionModel, self).__init__()
        self.l1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        return x


@rk.Agent
class PositionMangeAgent:
    state_list = []

    def __init__(self):
        self.model = PositionModel()

    async def run(self):
        while True:
            if len(self.state_list) % 10:
                print(self.state_list[0])  # Recall and train here

    async def receiver(self, sender, message):
        # print(self.code, sender, message.data)
        state = message.data["state"]
        reward = message.data["reward"]
        if len(self.state_list) > 0:
            self.state_list[-1][2] = reward
        res = self.model(torch.tensor(state))
        x = res.item()
        self.state_list.append([state, x, None])
        self.core.send({"receiver": Agent.CoordinatorAgent, "data": x})
