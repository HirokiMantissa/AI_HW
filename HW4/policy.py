import numpy as np

def policy_iteration(agent, max_iter=100):
    shape = agent.env.map.shape
    V = np.zeros(shape)
    policy = np.full(shape, "", dtype=object)

    # 隨機初始化 policy（除了終點和炸彈）
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = (i, j)
            if agent.env.map[state] not in [agent.env.wall, agent.env.final, agent.env.bomb]:
                policy[state] = np.random.choice(agent.actions)

    for iteration in range(max_iter):
        # ====== Policy Evaluation ======
        while True:
            delta = 0
            new_V = np.copy(V)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    state = (i, j)

                    if agent.env.map[state] in [agent.env.wall, agent.env.final, agent.env.bomb]:
                        continue

                    for action in agent.actions:
                        value = 0
                        next_state = agent.get_next_state(state, action)
                        reward = agent.get_reward(state, next_state, V)
                        
                        value = 0 + agent.gamma * (reward[0]*0.8 + reward[1]*0.1 + reward[2]*0.1) 
                        value = round(float(value), 2)

                    delta = max(delta, abs(value - V[state]))
                    new_V[state] = value

            V = new_V
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
                    probs = [0.8, 0.1, 0.1]

                    q_value = 0
                    for k in range(3):
                        print(next_states[k])
                        q_value  += probs[k] * (rewards[k] + agent.gamma * V[next_states[k]])
                    
                    if q_value > best_value:
                            best_value = q_value
                            best_action = action

                policy[state] = best_action
                if best_action != old_action:
                    policy_stable = False

        if policy_stable:
            print(f"✅ Policy Iteration Converged at iteration {iteration}")
            break

    return V, policy
