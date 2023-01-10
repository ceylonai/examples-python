import matplotlib.pyplot as plt

data = []

f = open('../lander_position_agent_loss.txt', 'r')
for row in f:
    data = row
x = data
y = range(len(x))
plt.plot(x, y)
plt.show()
