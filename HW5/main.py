import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ✅ 收斂：step 171, ω = 1.99144

# set parameter
R = 0.2
L = 0.0002
Ke = 0.0017
Kt = 0.017
J = 0.01
B = 0.2

alpha = 0.1  
gamma = 0.99
epsilon = 0.1
episodes = 500

omega_targ = 2.0
Ts = 0.001

error_bins = np.linspace(-300, 300, 301)  # 對應 e * 100
delta_error_bins = np.linspace(-1000, 1000, 301)  # 每格 0.5
actions = np.linspace(0, 20, 51) 
omega_train_record = []
target_episodes = [10, 20, 30, 40, 50]

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
    
    omega_history = []

    for t in range(1000):  
        omega = x[1]

        # 計算狀態
        e = omega_targ - omega
        de = e - e_prev
        e_prev = e

        SCALE_E = 100
        SCALE_DE = 100
        
        e_scaled = np.clip(e * SCALE_E, -3 * SCALE_E, 3 * SCALE_E)
        de_scaled = np.clip(de * SCALE_DE, -10 * SCALE_DE, 10 * SCALE_DE)
        
        s1 = discretize(e_scaled, error_bins)
        s2 = discretize(de_scaled, delta_error_bins)

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
        omega_history.append(x[1])
        
        # debug 印出
        print(f"u = {u:.1f}V, i = {x[0]:.2f}A, ω = {x[1]:.5f}rad/s, e = {e}, de = {de}")

        if abs(e_next) < 0.01:
            print(f"✅ 收斂：step {t}, ω = {x[1]:.5f}")
            break
    if (episode + 1) in target_episodes:
        omega_train_record.append(omega_history)
        
def plot_q_table_3d(Q_table, action_index=25):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 建立網格座標
    E, DE = np.meshgrid(error_bins, delta_error_bins, indexing='ij')
    Z = Q_table[:, :, action_index]

    ax.plot_surface(E, DE, Z, cmap='viridis')

    ax.set_xlabel('Error')
    ax.set_ylabel('Delta Error')
    ax.set_zlabel(f'Q-value (Action index: {action_index})')
    ax.set_title(f'3D Mesh Plot of Q-table (Action = {actions[action_index]:.2f})')
    plt.tight_layout()
    plt.show()
        
plot_q_table_3d(Q_table, action_index=50)


def plot_omega(omega_train_record):
    plt.figure(figsize=(10, 5))
    for i, omega_seq in enumerate(omega_train_record):
        plt.plot(np.arange(len(omega_seq)) * Ts, omega_seq, label=f'Episode {(i+1)*10}')

    plt.axhline(y=omega_targ, color='r', linestyle='--', label='Target ω')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (ω)')
    plt.title('ω Trajectory During Training Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_omega(omega_train_record)
