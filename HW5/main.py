import numpy as np
import matplotlib.pyplot as plt

# set parameter
R = 0.2       
L = 0.000002 
Ke = 0.0017   
Kt = 0.0017  
J = 2.0      
B = 0.2     
alpha = 0.1   
gamma = 0.99   
epsilon = 0.1 
episodes = 3

omega_targ = 2.0 # init condition
Ts = 0.001   

error_bins = np.linspace(-3, 3, 31)  # e(k)
delta_error_bins = np.linspace(-2, 2, 21)  # Δe(k)
actions = np.linspace(0, 5, 11)  # 電壓 u(k)：0V ~ 5V 之間共 11 個選項

# 初始化 Q-table
Q_table = np.zeros((len(error_bins), len(delta_error_bins), len(actions)))

def discretize(value, bins):
    return np.digitize([value], bins)[0] - 1

def motor_model_discrete(x, u):
    """
    離散時間馬達模型
    x: 狀態向量 [i, omega]
    u: 控制輸入（電壓）
    回傳: 下一步狀態 [i_next, omega_next]
    """
    i, omega = x

    # 限制輸入電壓範圍
    u = np.clip(u, 0, 5)

    # 計算 di/dt 與 dω/dt（不 clip i）
    di_dt = (1/L) * (-R * i - Ke * omega + u)
    domega_dt = (1/J) * (-B * omega + Kt * i)

    # 限制變化率（避免一次跳太大）
    di_dt = np.clip(di_dt, -1e5, 1e5)
    domega_dt = np.clip(domega_dt, -1e4, 1e4)

    # Euler integration
    i_next = i + Ts * di_dt
    omega_next = omega + Ts * domega_dt

    # 限制下一步的值（這裡才 clip）
    i_next = np.clip(i_next, -100.0, 100.0)
    omega_next = np.clip(omega_next, -200.0, 200.0)

    return np.array([i_next, omega_next])

# training
for episode in range(episodes):
    x = np.array([0.0, 0.0])  # 初始狀態（i, ω）
    e_prev = omega_targ

    for t in range(3): # 一個 episode 最多 2000 步
        omega = x[1]
        e = omega_targ - omega
        de = e - e_prev
        e_prev = e

        e = np.clip(e, -3, 3)
        de = np.clip(de, -2, 2)

        s1 = discretize(e, error_bins)
        s2 = discretize(de, delta_error_bins)
        
        # ε-greedy 選擇動作
        a_idx = np.argmax(Q_table[s1, s2])
        
        u = actions[a_idx]
        print(u)
        x_next = motor_model_discrete(x, u)
        print(x_next)

        omega_next = x_next[1]
        e_next = omega_targ - omega_next
        e_next = np.clip(e_next, -3, 3)

        if abs(e_next) < 0.05:
            reward = 10.0  # 明確鼓勵「誤差小」
        else:
            reward = -abs(e_next)

        s1_next = discretize(e_next, error_bins)
        s2_next = discretize(e_next - e, delta_error_bins)

        # Q-learning 更新
        Q_table[s1, s2, a_idx] += alpha * (
            reward + gamma * np.max(Q_table[s1_next, s2_next]) - Q_table[s1, s2, a_idx]
        )
        
        x = x_next

        if abs(e_next) < 0.01:
            break  # 收斂提早結束

print("訓練完成 ✅")