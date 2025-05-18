import numpy as np

from V import v_value
from Q import q_value
from policy import policy_iteration

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
        """_summary_

        Args:
            state (list): 現在的格子座標
            action (string): 要做的動作

        Returns:
            list: 給可能會到的三個格子座標
        """
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
        """_summary_

        Args:
            state (_type_): 現在的格子座標
            next_state (_type_): 可能會到的三個格子座標
            V (_type_): 12個格子的值

        Returns:
            list: 可能會到的三個格子座標的reward
        """
        reward = []
        nrows, ncols = self.env.map.shape
        
        for pos in next_state:
            i, j = pos
            if not (0 <= i < nrows and 0 <= j < ncols):
                reward.append(round(float(V[state]), 2))
                continue
            elif self.env.map[i, j] == self.env.wall:
                reward.append(round(float(V[state]), 2))
            else:
                reward.append(round(float(V[pos]), 2))
        return reward
        
    def is_terminal(self, state):
        return self.env.map[state] in [self.env.final, self.env.bomb]


game = Game()
agent = Agent(game)
V, policy = v_value(agent, 1000)
Q, policy = q_value(agent, 1000)
policy_iteration(agent)