import numpy as np
import matplotlib.pyplot as plt

# set parameter
R = 0.2
L = 0.0002
Ke = 0.0017
Kt = 0.0017
J = 2.0
B = 0.2

alpha = 0.1  
gamma = 0.99
epsilon = 0.1
episodes = 500

omega_targ = 2.0
Ts = 0.001

error_bins = np.linspace(-3, 3, 31)  # e(k)：共 31 個選項
delta_error_bins = np.linspace(-2, 2, 21)  # Δe(k)：共 21 個選項
actions = np.linspace(0, 5, 11)  # 電壓 u(k)：共 11 個選項

# 初始化 Q-table 31x21x11
Q_table = np.zeros((len(error_bins), len(delta_error_bins), len(actions)))

def discretize(value, bins):
    return np.digitize([value], bins)[0] - 1

def motor_model_discrete(x, u):
    i, omega = x

    # 限制輸入電壓
    u = np.clip(u, 0, 5)

    # 計算 di/dt 與 dω/dt
    di_dt = (1/L) * (-R * i - Ke * omega + u)
    di_dt = np.clip(di_dt, -1e4, 1e4)  # 防止數值爆炸

    domega_dt = (1/J) * (-B * omega + Kt * i)
    domega_dt = np.clip(domega_dt, -1e3, 1e3)

    i_next = i + Ts * di_dt
    omega_next = omega + Ts * domega_dt

    # 限制狀態範圍
    i_next = np.clip(i_next, -100, 100)
    omega_next = np.clip(omega_next, -200, 200)

    return np.array([i_next, omega_next])

# training
for episode in range(episodes):
    x = np.array([0.0, 0.0])  # 初始狀態（i, ω）
    e_prev = 0

    for t in range(1000):  # 訓練步數增加
        omega = x[1]

        # 計算狀態
        e = omega_targ - omega
        de = e - e_prev
        e_prev = e

        e = np.clip(e, -3, 3)
        de = np.clip(de, -2, 2)

        s1 = discretize(e, error_bins)
        s2 = discretize(de, delta_error_bins)

        # ε-greedy 策略
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(len(actions))
        else:
            a_idx = np.argmax(Q_table[s1, s2])

        u = actions[a_idx]
        x_next = motor_model_discrete(x, u)

        omega_next = x_next[1]
        e_next = omega_targ - omega_next
        e_next = np.clip(e_next, -3, 3)

        reward = -e_next**2

        s1_next = discretize(e_next, error_bins)
        s2_next = discretize(e_next - e, delta_error_bins)

        # Q-learning 更新
        Q_table[s1, s2, a_idx] += alpha * (
            reward + gamma * np.max(Q_table[s1_next, s2_next]) - Q_table[s1, s2, a_idx]
        )

        x = x_next

        # debug 印出
        print(f"u = {u:.1f}V, i = {x[0]:.2f}A, ω = {x[1]:.5f}rad/s")

        if abs(e_next) < 0.01:
            print(f"✅ 收斂：step {t}, ω = {x[1]:.5f}")
            break
