import numpy as np
import matplotlib.pyplot as plt

def value_iteration(agent, threshold=1e-4, max_iter=1000):
    shape = agent.env.map.shape
    V = np.zeros(shape)  # 初始化 V(s) = 0
    policy = np.full(shape, "", dtype=object)
    
    goal_state = agent.end
    goal_reward = 10  # 你定義的 goal 獎勵（也可從 agent.get_reward(goal_state) 取）

    for iteration in range(max_iter):
        delta = 0
        new_V = np.copy(V)

        for i in range(shape[0]):
            for j in range(shape[1]):
                state = (i, j)

                if agent.env.map[state] == agent.env.wall:
                    continue  # 牆壁不更新

                if agent.env.map[state] == agent.env.final:
                    # ✅ 強制將終點 V 設為 reward（避免未更新造成誤差）
                    new_V[state] = goal_reward
                    policy[state] = "⏹"  # 終點無動作
                    continue

                max_value = -np.inf
                best_action = None

                for action in agent.actions:
                    next_state = agent.get_next_state(state, action)
                    reward = agent.get_reward(next_state)
                    value = reward + agent.gamma * V[next_state]

                    if value > max_value:
                        max_value = value
                        best_action = action

                new_V[state] = max_value
                policy[state] = best_action
                delta = max(delta, abs(new_V[state] - V[state]))

        V = new_V
        if delta < threshold:
            print(f"Converged at iteration {iteration+1}")
            break

    return V, policy


def plot_value_map(game, V, agent_pos=None):
    grid = game.map
    shape = grid.shape

    V_min = np.min(V)
    V_max = np.max(V)
    if V_max - V_min > 0:
        V_norm = (V - V_min) / (V_max - V_min)
    else:
        V_norm = np.zeros_like(V)

    label_map = np.full(shape, "", dtype=object)

    for i in range(shape[0]):
        for j in range(shape[1]):
            cell = grid[i, j]
            if cell == game.final:
                V_norm[i, j] = 1.0
                label_map[i, j] = "G"
            elif cell == game.wall:
                V_norm[i, j] = 0.0
                label_map[i, j] = "W"
            elif cell == game.bomb:
                V_norm[i, j] = 0.0
                label_map[i, j] = "B"
            else:
                label_map[i, j] = f"{V_norm[i, j]:.2f}"

    plt.figure(figsize=(6, 4))
    cmap = plt.cm.get_cmap('coolwarm')
    plt.imshow(V_norm, cmap=cmap, extent=(0, shape[1], shape[0], 0), vmin=0, vmax=1)

    # 標示數值 or G/W/B
    for i in range(shape[0]):
        for j in range(shape[1]):
            text = label_map[i, j]
            color = 'black'
            plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=11, color=color)

    if agent_pos is not None:
        ai, aj = agent_pos
        plt.text(aj + 0.5, ai + 0.5, "A", ha='center', va='center', fontsize=14, color='blue', weight='bold')

    plt.title("State Value V(s)")
    plt.grid(color='black')
    plt.xticks(np.arange(0, shape[1]+1, 1))
    plt.yticks(np.arange(0, shape[0]+1, 1))
    plt.gca().set_xticks(np.arange(0.5, shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, shape[0], 1), minor=True)
    plt.gca().grid(which='minor', color='black', linewidth=1)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.colorbar(shrink=0.8)
    plt.show()


def plot_policy_map(game, policy, agent_pos=None):
    grid = game.map
    shape = grid.shape

    arrow_map = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    label_map = np.full(shape, "", dtype=object)

    for i in range(shape[0]):
        for j in range(shape[1]):
            cell = grid[i, j]
            if cell == game.final:
                label_map[i, j] = "G"
            elif cell == game.wall:
                label_map[i, j] = "W"
            elif cell == game.bomb:
                label_map[i, j] = "B"
            else:
                action = policy[i, j]
                label_map[i, j] = arrow_map.get(action, "?")

    value_map = np.zeros(shape)
    plt.figure(figsize=(6, 4))
    plt.imshow(value_map, cmap="Greys", extent=(0, shape[1], shape[0], 0))

    # 畫文字（箭頭或標籤）
    for i in range(shape[0]):
        for j in range(shape[1]):
            text = label_map[i, j]
            color = "black" if text not in ["W", "B", "G"] else "red"
            plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=14, weight='bold', color=color)

    if agent_pos is not None:
        ai, aj = agent_pos
        plt.text(aj + 0.5, ai + 0.5, "A", ha='center', va='center', fontsize=14, color='blue', weight='bold')

    plt.title("Optimal Policy π(s)")
    plt.grid(color='black')
    plt.xticks(np.arange(0, shape[1]+1, 1))
    plt.yticks(np.arange(0, shape[0]+1, 1))
    plt.gca().set_xticks(np.arange(0.5, shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, shape[0], 1), minor=True)
    plt.gca().grid(which='minor', color='black', linewidth=1)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()