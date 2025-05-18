import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def policy_iteration(agent):
    shape = agent.env.map.shape
    V = np.zeros(shape)
    V[agent.end] = 1
    V[agent.bomb] = -1
    policy = np.full(shape, "", dtype=object)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if agent.env.map[i, j] not in [agent.env.wall, agent.env.final, agent.env.bomb]:
                policy[i, j] = np.random.choice(agent.actions)
    time = 0
    
    # ====== Policy Evaluation ======
    while True:
        
        delta = 0
        new_V = np.copy(V)
        for i in range(shape[0]):
            for j in range(shape[1]):
                state = (i, j)

                if agent.env.map[state] in [agent.env.wall, agent.env.final, agent.env.bomb]:
                    continue

                action = policy[state]
                next_states = agent.get_next_state(state, action)
                rewards = agent.get_reward(state, next_states, V)
                
                value = sum([
                    0.8 * (0 + agent.gamma * rewards[0]),
                    0.1 * (0 + agent.gamma * rewards[1]),
                    0.1 * (0 + agent.gamma * rewards[2])
                ])
                value = round(float(value), 4)
                delta = max(delta, abs(value - V[state]))
                new_V[state] = value

        V = new_V
        time += 1
        if delta < 1e-4:
            break

        # ====== Policy Improvement ======
        policy_stable = True

        for i in range(shape[0]):
            for j in range(shape[1]):
                state = (i, j)

                if agent.env.map[state] in [agent.env.wall, agent.env.final, agent.env.bomb]:
                    continue

                old_action = policy[state]
                best_value = -np.inf
                best_action = None

                for action in agent.actions:
                    next_states = agent.get_next_state(state, action)
                    rewards = agent.get_reward(state, next_states, V)
                    value = sum([
                        0.8 * (0 + agent.gamma * rewards[0]),
                        0.1 * (0 + agent.gamma * rewards[1]),
                        0.1 * (0 + agent.gamma * rewards[2])
                    ])
                    value = round(float(value), 4)
                    if value > best_value:
                        best_value = value
                        best_action = action

                policy[state] = best_action
                if best_action != old_action:
                    policy_stable = False

        if policy_stable:
            break
    plot_map(V, policy)
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

    plt.title("Policy Iteration Result", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()