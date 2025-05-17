import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from V import value_iteration, plot_policy_map, plot_value_map

class Game:
    def __init__(self):
        self.bomb = "bomb"
        self.wall = "wall"
        self.final = "final"

        self.map = np.array([
            [0, 0, 0, self.final],
            [0, self.wall, 0, self.bomb],
            [0, 0, 0, 0]
        ])

class Agent:
    def __init__(self, env: Game):
        self.env = env
        self.actions = ["up", "down", "left", "right"]
        
        self.gamma = 0.9
        self.start = (2, 0)
        self.end = (0, 3)
        self.current_state = self.start
        
    def get_next_state(self, state, action):
        i, j = state
        if action == "up":
            i = max(i - 1, 0)
        elif action == "down":
            i = min(i + 1, self.env.map.shape[0] - 1)
        elif action == "left":
            j = max(j - 1, 0)
        elif action == "right":
            j = min(j + 1, self.env.map.shape[1] - 1)

        if self.env.map[i, j] == self.env.wall:
            return state
        return (i, j)

    def get_reward(self, state):
        cell = self.env.map[state]
        if cell == self.env.final:
            return 10
        elif cell == self.env.bomb:
            return -10
        elif cell == self.env.wall:
            return 0
        else:
            return -1 

    def is_terminal(self, state):
        return self.env.map[state] in [self.env.final, self.env.bomb]



def plot_map(game: Game, agent_pos=None):
    grid = game.map
    value_map = np.zeros(grid.shape)
    text_map = np.full(grid.shape, "", dtype=object)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell = grid[i, j]
            if cell == game.final:
                value_map[i, j] = 3
                text_map[i, j] = "G"
            elif cell == game.wall:
                value_map[i, j] = 1
                text_map[i, j] = "W"
            elif cell == game.bomb:
                value_map[i, j] = 2
                text_map[i, j] = "B"
            else:
                value_map[i, j] = 0
                text_map[i, j] = "0"

    cmap = ListedColormap(["white", "gray", "red", "green"])

    plt.figure(figsize=(6, 4))
    plt.imshow(value_map, cmap=cmap, extent=(0, grid.shape[1], grid.shape[0], 0))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.text(j + 0.5, i + 0.5, text_map[i, j], ha='center', va='center', fontsize=12, weight='bold')

    if agent_pos is not None:
        ai, aj = agent_pos
        plt.text(aj + 0.5, ai + 0.5, "A", ha='center', va='center', fontsize=14, color='blue', weight='bold')

    plt.grid(color='black')
    plt.xticks(np.arange(0, grid.shape[1]+1, 1))
    plt.yticks(np.arange(0, grid.shape[0]+1, 1))
    plt.gca().set_xticks(np.arange(0.5, grid.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, grid.shape[0], 1), minor=True)
    plt.gca().grid(which='minor', color='black', linewidth=1)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("GridWorld Map", fontsize=14)
    plt.show()
    
game = Game()
agent = Agent(game)
V, policy = value_iteration(agent, max_iter=10)
plot_value_map(game, V, agent_pos=agent.current_state)
plot_policy_map(game, policy, agent_pos=agent.current_state)



print("State Value V*(s):")
print(np.round(V, 2))

print("\nOptimal Policy π*(s):")
print(policy)
