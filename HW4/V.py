import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def v_value(agent, max_iter):
    shape = agent.env.map.shape
    V = np.zeros(shape)
    V[agent.end] = 1
    V[agent.bomb] = -1
    policy = np.full(shape, "", dtype=object)

    for iteration in range(max_iter):
        print("========={}===========".format(iteration))
        new_V = np.copy(V)

        for i in range(shape[0]):
            for j in range(shape[1]):
                state = (i, j)
                print(state)
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
                calculate v* processing
                """
                for action in agent.actions:
                    value = 0
                    next_state = agent.get_next_state(state, action)
                    print(next_state)
                    reward = agent.get_reward(state, next_state, V)
                    
                    value = 0 + agent.gamma * (reward[0]*0.8 + reward[1]*0.1 + reward[2]*0.1)
                    value = round(float(value), 2)
                    print("{} = 0 + {} * ({}*0.8 + {}*0.1) + {}*0.1".format(value, agent.gamma, reward[0], reward[1], reward[2]))
                    
                    if value > max_value:
                        max_value = value
                        best_action = action

                new_V[state] = max_value
                policy[state] = best_action
                print("\n")
        if np.array_equal(V, new_V):
            break
        V = new_V
        
    plot_map(V,policy)
    return V, policy

def plot_map(V, policy):

    arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

    V = np.flipud(V)
    policy = np.flipud(policy)
    rows, cols = V.shape
    fig, ax = plt.subplots(figsize=(cols * 1.2, rows * 1.2))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(rows):
        for j in range(cols):
            x = j
            y = rows - 1 - i  

        
            val = V[i, j]
            if val == 1.0:
                color = 'green'
            elif val == -1.0:
                color = 'red'
            elif val == 0.0 and policy[i, j] == '':
                color = 'gray'
            else:
                color = 'darkgreen'

        
            ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white'))

            ax.text(x + 0.5, y + 0.7, f"{val:.2f}", ha='center', va='center', color='white', fontsize=10)

            action = policy[i, j]
            if action in arrow_map:
                ax.text(x + 0.5, y + 0.3, arrow_map[action], ha='center', va='center', color='white', fontsize=16)

    plt.title("V-Value and Policy Result", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()