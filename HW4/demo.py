import numpy as np

class Game:
    def __init__(self):
        self.bomb = "bomb"
        self.wall = "wall"
        self.final = "final"

        self.gamma = 0.9
        self.start = (2, 0)
        self.end = (0, 3)
        self.current_state = self.start

        self.map = np.array([
            [0, 0, 0, self.final],
            [0, self.wall, 0, self.bomb],
            [0, 0, 0, 0]
        ])
    
    def get_next_state(self, state, action):
        i, j = state
        if action == "up":
            i = max(i - 1, 0)
        elif action == "down":
            i = min(i + 1, self.map.shape[0] - 1)
        elif action == "left":
            j = max(j - 1, 0)
        elif action == "right":
            j = min(j + 1, self.map.shape[1] - 1)
    
    
        if self.map[i, j] == self.wall:
            return state
        return (i, j)
    
    def get_reward(self, state):
        val = self.map[state]
        if val == self.final:
            return 10
        elif val == self.bomb:
            return -10
        elif val == self.wall:
            return 0
        else:
            return -1

    def is_terminal(self, state):
        return self.map[state] in [self.final, self.bomb]

    def reset(self):
        self.current_state = self.start
        return self.current_state

class Agent:
    def __init__(self, env: Game):
        self.env = env
        self.state = env.reset()
        self.actions = ["up", "down", "left", "right"]
        self.gamma = env.gamma
    
    def act(self, action):
        next_state = self.env.get_next_state(self.state, action)
        reward = self.env.get_reward(next_state)
        done = self.env.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done
        