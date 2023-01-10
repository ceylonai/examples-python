import numpy as np
import rakun_python as rk

from agents import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    loss_list = []

    def __init__(self):
        self.model = PositionModel()

    async def run(self):
        while True:
            if len(self.state_list) == 0:
                continue
            if len(self.state_list) % 10 == 0:
                state_list = np.array(self.state_list[:-1])
                x = torch.tensor(state_list[:, 0:2], dtype=torch.float)
                x_pred = torch.tensor(state_list[:, 2:3], dtype=torch.float)
                reward = torch.tensor(state_list[:, 3:4], dtype=torch.float)
                y = self.model(x)
                loss = F.mse_loss(y, reward)
                self.loss_list.append(loss.item())
                loss.backward()

                with open('lander_position_agent_loss.txt', 'w') as f:
                    f.write(str(self.loss_list))

    async def receiver(self, sender, message):
        # print(self.code, sender, message.data)
        state = message.data["state"]
        reward = message.data["reward"]
        if len(self.state_list) > 0:
            self.state_list[-1][-1] = reward
        res = self.model(torch.tensor(state))
        x = res.item()
        self.state_list.append([*state, x, None])
        self.core.send({"receiver": Agent.CoordinatorAgent, "data": x})
