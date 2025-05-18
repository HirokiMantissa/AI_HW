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
            q_table.insert(0, row)
            
        if np.array_equal(V, new_V):
            break
        V = new_V
    
    q_table = [cell for row in q_table for cell in row]
    plot_map(q_table)
    return q_table, policy

def plot_map(q_table, title="Q-values and Policy"):
    
    arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

    policy_dict = {}
    for cell in q_table:
        pos = cell['pos']
        best_action = max(['up', 'down', 'left', 'right'], key=lambda a: cell[a])
        policy_dict[pos] = best_action

    max_i = max(cell['pos'][0] for cell in q_table)
    max_j = max(cell['pos'][1] for cell in q_table)
    rows, cols = max_i + 1, max_j + 1

    q_dict = {cell['pos']: cell for cell in q_table}
    
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(rows):
        for j in range(cols):
            x = j
            y = i
            pos = (i, j)
            
            if pos == (1,3):
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='red', edgecolor='white'))
                ax.text(x + 0.5, y + 0.5, f"{-1}", ha='center', va='center', fontsize=16, color='white')
            elif pos == (2,3):
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='green', edgecolor='white'))
                ax.text(x + 0.5, y + 0.5, f"{1}", ha='center', va='center', fontsize=16, color='white') 
            elif pos == (1,1):
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='gray', edgecolor='white'))
            else:
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='darkgreen', edgecolor='white'))

                if pos in q_dict:
                    q = q_dict[pos]

                    ax.text(x + 0.5, y + 0.9, f"{q['up']:.2f}", ha='center', va='center', fontsize=8, color='white')
                    ax.text(x + 0.5, y + 0.1, f"{q['down']:.2f}", ha='center', va='center', fontsize=8, color='white')
                    ax.text(x + 0.1, y + 0.5, f"{q['left']:.2f}", ha='center', va='center', fontsize=8, color='white')
                    ax.text(x + 0.9, y + 0.5, f"{q['right']:.2f}", ha='center', va='center', fontsize=8, color='white')

                    arrow = arrow_map[policy_dict[pos]]
                    ax.text(x + 0.5, y + 0.5, arrow, ha='center', va='center', fontsize=16, color='white')

    plt.title(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
