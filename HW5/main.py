import numpy as np
import matplotlib.pyplot as plt

# 參數設定
R = 0.2       # 歐姆
L = 0.000002  # 亨利
Ke = 0.0017   # V / (rad/s)
Kt = 0.0017   # Nm / A
J = 2.0       # kg*m^2
B = 0.2       # N*m*s

alpha = 0.1     # 學習率
gamma = 0.99    # 折扣率
epsilon = 0.1   # 探索率
episodes = 500

omega_targ = 2.0
Ts = 0.001    # 取樣時間（秒）

error_bins = np.linspace(-3, 3, 31)  # e(k)
delta_error_bins = np.linspace(-2, 2, 21)  # Δe(k)
actions = np.linspace(0, 5, 11)  # 電壓 u(k)

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

    # 限制狀態範圍，避免發散（穩定控制範圍內）
    u = np.clip(u, 0, 5)          # 限制輸入電壓在 [0V, 5V]
    i = np.clip(i, -10, 10)
    omega = np.clip(omega, -50, 50)

    di_dt = (1/L) * (-R * i - Ke * omega + u)
    domega_dt = (1/J) * (-B * omega + Kt * i)

    i_next = i + Ts * di_dt
    omega_next = omega + Ts * domega_dt

    return np.array([i_next, omega_next])


# training
for episode in range(episodes):
    x = np.array([0.0, 0.0])  # 初始狀態
    e_prev = omega_targ

    for t in range(2000):
        omega = x[1]
        e = omega_targ - omega
        de = e - e_prev
        e_prev = e

        # 🛠 修正這裡，加在 e 計算後
        e = np.clip(e, -3, 3)
        de = np.clip(de, -2, 2)

        s1 = discretize(e, error_bins)
        s2 = discretize(de, delta_error_bins)
        
        # ε-greedy 選擇動作
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(len(actions))
        else:
            a_idx = np.argmax(Q_table[s1, s2])

        u = actions[a_idx]
        x_next = motor_model_discrete(x, u)

        omega_next = x_next[1]
        e_next = omega_targ - omega_next
        e_next = np.clip(e_next, -3, 3)

        reward = -min(e_next**2, 10)  # 安全的 reward 函數

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

# test
x = np.array([0.0, 0.0])
e_prev = omega_targ
omega_list = []

for t in range(1000):
    omega = x[1]
    e = omega_targ - omega
    de = e - e_prev
    e_prev = e

    s1 = discretize(e, error_bins)
    s2 = discretize(de, delta_error_bins)
    a_idx = np.argmax(Q_table[s1, s2])
    u = actions[a_idx]

    x = motor_model_discrete(x, u)
    omega_list.append(x[1])

plt.plot(omega_list)
plt.xlabel("Time step (k)")
plt.ylabel("Motor Speed ω (rad/s)")
plt.title("Q-Learning 控制下的轉速響應")
plt.grid(True)
plt.show()

# plot
from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(error_bins, delta_error_bins, indexing='ij')
Z = np.max(Q_table, axis=2)  # 對每個狀態取最大 Q 值

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("Error e(k)")
ax.set_ylabel("Delta e(k)")
ax.set_zlabel("Max Q value")
ax.set_title("Q-table 最大值 3D Mesh")
plt.show()
