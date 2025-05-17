import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
"""
    in my opinion: 建立一個結構Q_table,
    來存取每個格子四個動作的期望rward.
"""

class Q_TABLE:
    def __init__(self):
        self.table = []
    
    def get_items(self):
        items = {
            "pos":(0,0),
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0
        }
        return items

def q_value(agent, max_iter):
    shape = agent.env.map.shape
    V = np.zeros(shape)
    V[agent.end] = 1
    V[agent.bomb] = -1
    policy = np.full(shape, "", dtype=object)

    for iteration in range(max_iter):
        new_V = np.copy(V)
        q_table = Q_TABLE().table
        
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                state = (i, j)
                
                q = Q_TABLE().get_items()
                q["pos"] = state
                max_value = -np.inf
                best_action = None

                if agent.env.map[state] == agent.env.wall:
                    continue 

                if agent.env.map[state] == agent.env.final:
                    new_V[state] = 1
                    continue
                
                if agent.env.map[state] == agent.env.bomb:
                    new_V[state] = -1
                    continue

                """
                Get Q_table, but use V*to procsssing
                """
                for action in agent.actions:
                    value = 0
                    next_state = agent.get_next_state(state, action)
                    reward = agent.get_reward(state, next_state, V)
                    
                    value = 0 + agent.gamma * (reward[0]*0.8 + reward[1]*0.1 + reward[2]*0.1) 
                    value = round(float(value), 2)
                    q[action] = value
                    
                    if value > max_value:
                        max_value = value
                        best_action = action

                new_V[state] = max_value
                policy[state] = best_action
                row.append(q)
            q_table.append(row)
        V = new_V

    return q_table, policy