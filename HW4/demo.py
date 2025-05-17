import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from V import v_value

class Game:
    def __init__(self):
        self.bomb = "bomb"
        self.wall = "wall"
        self.final = "final"

        self.map = np.array([
            [0, 0, 0, 0],
            [0, self.wall, 0, self.bomb],
            [0, 0, 0, self.final]
        ])

class Agent:
    def __init__(self, env: Game):
        self.env = env
        self.actions = ["up", "down", "left", "right"]
        
        self.gamma = 0.9
        self.start = (0, 0)
        self.end = (2, 3)
        self.bomb = (1, 3)
        self.current_state = self.start
        
    def get_next_state(self, state, action):
        i, j = state
        if action == "up":
            next_state = [(i+1, j), (i, j+1), (i, j-1)]
        elif action == "down":
            next_state = [(i-1, j), (i, j+1), (i, j-1)]
        elif action == "left":
            next_state = [(i, j-1), (i+1, j), (i-1, j)]
        elif action == "right":
            next_state = [(i, j+1), (i+1, j), (i-1, j)]
        return next_state

    def get_reward(self, state, next_state, V):
        reward = []
        nrows, ncols = self.env.map.shape
        
        for pos in next_state:
            i, j = pos
            if not (0 <= i < nrows and 0 <= j < ncols):
                reward.append(round(float(V[state]), 2))
                continue
            if self.env.map[i, j] == self.env.wall:
                reward.append(round(float(V[state]), 2))
            else:
                reward.append(round(float(V[pos]), 2))
        return reward
        
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
V, policy = v_value(agent, 1000)