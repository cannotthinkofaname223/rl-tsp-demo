import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

#输入城市数量
user_input = input("Enter number of cities (default 8, Max 15): ")

if user_input == "" :
    num_cities = 8
else:
    num_cities = int(user_input)
    num = int(user_input)
    if num > 15:
        num_cities = 8
    else:
        num_cities = num
print("Number of cities:", num_cities)

#生成城市坐标和模拟图
cities = np.random.rand(num_cities, 2)

print ("Cities coordinates:")
print (cities)

plt.scatter(cities[:, 0], cities[:, 1])

for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12)

plt.title("TSP Cities")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#定义函数
def calculate_distance(path, cities):
    distance = 0

    for i in range(len(path)):
        current_city = cities[path[i]]
        next_city = cities[path[(i + 1) % len(path)]]

        distance += np. linalg.norm(current_city - next_city)

    return distance

"""def plot_route(cities, path, title="TSP Rout"):

    plt.figure(figsize=(6,6))

    plt.scatter(cities[:,0], cities[:,1])

    for i, (x, y) in enumerate(cities):
        plt.text(x,y,str(i),fontsize=12)

    for i in range(len(path)):
        start = cities[path[i]]
        end = cities[path[(i + 1) % len(path)]]

        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            'r-',
            linewidth=2
        )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()

path = np.random.permutation(num_cities)

print("Random path:", path)

dist = calculate_distance(path, cities)

print("Path length:", dist)

plot_route(cities, path,"Random Path")

"""


path = np.random.permutation(num_cities)

print("Random path:")
print(path)

dist = calculate_distance(path, cities)

print("Path  length:", dist)

#画出路径
plt.scatter(cities[:, 0], cities[:, 1])

for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12)

    for i in range(len(path)):
        start = cities[path[i]]
        end = cities[path[(i + 1)%num_cities]]

        plt.plot([start[0], end[0]],[start[1], end[1]], 'r-')

    plt.title("Random TSP Route")
    plt.show()

#真正进入RL
class PolicyNetwork(nn.Module):

    def __init__(self, num_cities):
        super(PolicyNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_cities * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_cities)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

policy = PolicyNetwork(num_cities)

optimizer = optim.Adam(policy.parameters(), lr=0.01)

def generate_path(policy, cities):

    state = torch.tensor(cities.flatten(), dtype=torch.float32)

    visited = set()
    path = []
    log_probs = []

    for _ in range(len(cities)):

        probs = policy(state)

        # 屏蔽已经访问的城市
        mask = torch.ones(len(cities))
        for v in visited:
            mask[v] = 0

        probs = probs * mask
        probs = probs / probs.sum()

        dist = torch.distributions.Categorical(probs)
        city = dist.sample()

        log_probs.append(dist.log_prob(city))

        city = city.item()

        path.append(city)
        visited.add(city)

    return path, log_probs


num_episodes = 4000
reward_history = []

best_distance = float('inf')
best_path = None

for episode in range(num_episodes):

    path, log_probs = generate_path(policy, cities)

    distance = calculate_distance(path, cities)

    reward = -distance

    if distance < best_distance:
        best_distance = distance
        best_path = path.copy()

    loss = 0
    for log_prob in log_probs:
        loss += -log_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    reward_history.append(distance)

    if episode % 50 == 0:
        print("Episode:", episode, "Distance:", distance, "Best", best_distance)

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Path Length")
plt.title("Training Curve")
plt.show()

plt.figure()

smooth = moving_average(reward_history, window=100)

plt.plot(smooth)

plt.title("RL Training Curve (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Path Length")

plt.show()

#Random vs RL 对比
def random_path(num_cities):
    return list(np.random.permutation(num_cities))

rand_path = random_path(num_cities)
rand_distance = calculate_distance(rand_path, cities)

print("Random path:", rand_path)
print("Random distance:", rand_distance)

best_path, _ = generate_path(policy, cities)

rl_distance = calculate_distance(best_path, cities)

print("RL path:", best_path)
print("RL distance:", rl_distance)

print("Best RL path:", best_path)
print("Best RL distance:", best_distance)

print("Training finished")

