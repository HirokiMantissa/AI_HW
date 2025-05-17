import numpy as np

def v_value(agent, max_iter):
    shape = agent.env.map.shape
    V = np.zeros(shape)  # 初始化 V(s) = [0]
    V[agent.end] = 1
    V[agent.bomb] = -1
    policy = np.full(shape, "", dtype=object)

    for iteration in range(max_iter):
        new_V = np.copy(V)

        for i in range(shape[0]):
            for j in range(shape[1]):
                state = (i, j)

                if agent.env.map[state] == agent.env.wall:
                    continue 

                if agent.env.map[state] == agent.env.final:
                    new_V[state] = 1
                    continue
                
                if agent.env.map[state] == agent.env.bomb:
                    new_V[state] = -1
                    continue

                max_value = -np.inf
                best_action = None

                """
                calculate v* processing
                """
                for action in agent.actions:
                    next_state = agent.get_next_state(state, action)
                    reward = agent.get_reward(state, next_state, V)
                    value = 0 + agent.gamma * (reward[0]*0.8 + reward[1]*0.1 + reward[2]*0.1) 
                    value = round(float(value), 2)
                    
                    if value > max_value:
                        max_value = value
                        best_action = action

                new_V[state] = max_value
                policy[state] = best_action
        V = new_V
        
    print(V)
    print(policy)
    return V, policy
