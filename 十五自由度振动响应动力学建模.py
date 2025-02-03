import numpy as np
import matplotlib.pyplot as plt

# 定义参数
m_i, m_o, m_r, m_b = 1, 1, 1, 1  # 内圈、外圈、滚子和基座质量
k_i, k_o, k_r, k_b = 1, 1, 1, 1  # 内圈、外圈、滚子和基座刚度
c_i, c_o, c_r, c_b = 0.1, 0.1, 0.1, 0.1  # 内圈、外圈、滚子和基座阻尼
I_i, I_o, I_r, I_b = 1, 1, 1, 1  # 内圈、外圈、滚子和基座转动惯量

# 定义仿真参数
dt = 0.01
timesteps = 1000
time = np.linspace(0, dt * timesteps, timesteps)

# 初始化状态变量
x_i = np.zeros(timesteps)
x_o = np.zeros(timesteps)
x_r = np.zeros(timesteps)
x_b = np.zeros(timesteps)
y_i = np.zeros(timesteps)
y_o = np.zeros(timesteps)
y_r = np.zeros(timesteps)
y_b = np.zeros(timesteps)
theta_i = np.zeros(timesteps)
theta_o = np.zeros(timesteps)
theta_r = np.zeros(timesteps)
theta_b = np.zeros(timesteps)

# 定义外部负载
F_load = np.sin(time)  # 这里只是一个示例，实际应用中需要根据具体情况定义

# 模拟系统动力学行为
for t in range(1, timesteps):
    # 内圈运动方程
    F_contact_i = 0  # 内圈接触力
    x_dot_i = y_i[t - 1]
    y_dot_i = (F_load[t] - k_i * x_i[t - 1] - c_i * y_i[t - 1] - F_contact_i) / m_i
    theta_dot_i = y_i[t - 1]

    x_i[t] = x_i[t - 1] + dt * x_dot_i
    y_i[t] = y_i[t - 1] + dt * y_dot_i
    theta_i[t] = theta_i[t - 1] + dt * theta_dot_i

    # 外圈运动方程
    F_contact_o = 0  # 外圈接触力
    x_dot_o = y_o[t - 1]
    y_dot_o = (-k_o * x_o[t - 1] - c_o * y_o[t - 1] - F_contact_o) / m_o
    theta_dot_o = y_o[t - 1]

    x_o[t] = x_o[t - 1] + dt * x_dot_o
    y_o[t] = y_o[t - 1] + dt * y_dot_o
    theta_o[t] = theta_o[t - 1] + dt * theta_dot_o

    # 滚子运动方程
    F_contact_r = 0  # 滚子接触力
    x_dot_r = y_r[t - 1]
    y_dot_r = (-k_r * x_r[t - 1] - c_r * y_r[t - 1] - F_contact_r) / m_r
    theta_dot_r = y_r[t - 1]

    x_r[t] = x_r[t - 1] + dt * x_dot_r
    y_r[t] = y_r[t - 1] + dt * y_dot_r
    theta_r[t] = theta_r[t - 1] + dt * theta_dot_r

    # 基座运动方程
    F_contact_b = 0  # 基座接触力
    x_dot_b = y_b[t - 1]
    y_dot_b = (-k_b * x_b[t - 1] - c_b * y_b[t - 1] - F_contact_b) / m_b
    theta_dot_b = y_b[t - 1]

    x_b[t] = x_b[t - 1] + dt * x_dot_b
    y_b[t] = y_b[t - 1] + dt * y_dot_b
    theta_b[t] = theta_b[t - 1] + dt * theta_dot_b

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(time, x_i, label='Inner Race')
plt.plot(time, x_o, label='Outer Race')
plt.plot(time, x_r, label='Roller')
plt.plot(time, x_b, label='Base')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Vibration Response of Rotating Machinery with Faults')
plt.legend()
plt.grid(True)
plt.show()
