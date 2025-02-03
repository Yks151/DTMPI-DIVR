import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DigitalBearingDataBase import SKF6312, CWRU_SKF6203
import os
from scipy.fftpack import fft
from sklearn.metrics import mean_absolute_percentage_error
from scipy.signal import butter, filtfilt
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 构建虚拟模型并生成虚拟信号
def generate_virtual_signal(params, num_samples=6000):
    try:
        Fs, f_n, B, m, g, e, rho = params
        bearing = SKF6312(Fs, f_n, B, m, g, e, rho)
        data_sim = bearing.getSingleSample(rev=1750, ns=num_samples)
        return data_sim['OuterRace']['x']
    except Exception as e:
        print(f"Error generating virtual signal: {e}")
        return np.zeros(num_samples)  # 返回一个零数组作为占位符


# 计算频率分析和相对误差
def frequency_analysis_and_error(virtual_signal, real_signal):
    try:
        fft_virtual = np.abs(fft(virtual_signal))
        fft_real = np.abs(fft(real_signal))
        relative_error = np.mean(np.abs(fft_virtual - fft_real) / (fft_real + 1e-10))  # 避免除以0
        return relative_error
    except Exception as e:
        print(f"Error in frequency analysis: {e}")
        return float('inf')


# 加载真实数据
cwru = CWRU_SKF6203()
file_path = r'E:/可解释性故障诊断学习/Fault-Diagnosis-Predictive-Maintainence-Digital-Twin-master-main/0HP/007/OR007@6_0.mat'

if os.path.exists(file_path):
    data_real, _ = cwru.readOR(type='007', load='0', location='@6')
    data_real = data_real[:6000]  # 取前6000个数据点作为测试

    initial_params = [10000, 3000, 0.5, 50, 9.8, 50e-6, 5e-3]
    virtual_signal = generate_virtual_signal(initial_params)

    # 计算相对误差
    error = frequency_analysis_and_error(virtual_signal, data_real)
    print(f"Relative Error: {error * 100:.2f}%")

    # 如果误差大于5%，调整建模方法（这里简化为打印提示）
    if error > 0.05:
        print("Relative error is larger than 5%, adjust the modeling method.")
    else:
        print("Modeling method is acceptable.")
else:
    print(f"文件不存在: {file_path}")


# 滤波算法
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# 预处理信号
def preprocess_signals(signal, real_signal, fs=12000, cutoff=500):
    filtered_signal = butter_lowpass_filter(signal, cutoff, fs)
    filtered_real_signal = butter_lowpass_filter(real_signal, cutoff, fs)

    scaler = MinMaxScaler()
    normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
    normalized_real_signal = scaler.fit_transform(filtered_real_signal.reshape(-1, 1)).flatten()

    return normalized_signal, normalized_real_signal


preprocessed_signal, preprocessed_real_signal = preprocess_signals(virtual_signal, data_real)

# 计算物理信号和生成信号之间的相关性
correlation = np.corrcoef(preprocessed_signal, preprocessed_real_signal)[0, 1]
print(f"Correlation between preprocessed signals: {correlation:.2f}")

# 如果相关性低于某个阈值，则调整参数（这里简化为打印提示）
if correlation < 0.8:
    print("Correlation is lower than 0.8, consider adjusting the parameters.")
else:
    print("Correlation is acceptable.")

# 定义敏感性分析问题
problem = {
    'num_vars': 7,
    'names': ['Fs', 'f_n', 'B', 'm', 'g', 'e', 'rho'],
    'bounds': [[1000, 12000], [1000, 5000], [0.1, 1], [30, 100], [9.7, 9.9], [10e-6, 100e-6], [0.1e-3, 10e-3]]
}

# 生成参数样本
param_values = saltelli.sample(problem, 128)

# 计算仿真信号
def simulate(param_set):
    return generate_virtual_signal(param_set)

# 计算所有样本的误差
Y = []
try:
    for param_set in param_values:
        signal = simulate(param_set)
        error = frequency_analysis_and_error(signal, data_real)
        Y.append(error)
except ValueError as e:
    print(f"Error occurred during simulation: {e}")

Y = np.array(Y)

# 敏感性分析
Si = sobol.analyze(problem, Y)

# 打印敏感性分析结果
print("Sensitivity Analysis Results:")
for i in range(problem['num_vars']):
    print(f"{problem['names'][i]}: {Si['S1'][i]:.4f}")

# 可视化敏感性分析结果
plt.figure(figsize=(10, 6))
plt.bar(problem['names'], Si['S1'])
plt.xlabel('参数')
plt.ylabel('一阶敏感度指数')
plt.title('参数敏感性分析结果')
plt.show()

# 建立代理模型（PRSM）
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 拟合代理模型
gpr.fit(param_values, Y)

# 预测并分析拟合精度
y_pred, sigma = gpr.predict(param_values, return_std=True)
mse = mean_squared_error(Y, y_pred)
print(f"PRSM Model Fitting MSE: {mse:.4f}")


# 目标函数：使用代理模型预测误差
def objective_function(params):
    params = np.array(params).reshape(1, -1)
    pred, sigma = gpr.predict(params, return_std=True)
    return pred[0]


# 萤火虫算法参数
num_fireflies = 20  # 萤火虫数量
max_iter = 100  # 最大迭代次数
alpha = 0.2  # 随机步长因子
beta0 = 1  # 吸引度基准值
gamma = 1  # 吸引度衰减因子


# 萤火虫算法
def firefly_algorithm(obj_func, lower_bounds, upper_bounds):
    num_params = len(lower_bounds)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    fireflies = np.random.rand(num_fireflies, num_params) * (upper_bounds - lower_bounds) + lower_bounds
    fitness = np.zeros(num_fireflies)
    param_history = []

    for i in range(num_fireflies):
        fitness[i] = obj_func(fireflies[i])

    best_idx = np.argmin(fitness)
    best_firefly = fireflies[best_idx, :].copy()
    best_fitness = fitness[best_idx]

    history = [best_fitness]

    for iteration in range(max_iter):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(fireflies[i, :] - fireflies[j, :])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    fireflies[i, :] += beta * (fireflies[j, :] - fireflies[i, :]) + alpha * (np.random.rand(num_params) - 0.5)
                    fireflies[i, :] = np.clip(fireflies[i, :], lower_bounds, upper_bounds)
                    fitness[i] = obj_func(fireflies[i, :])
        param_history.append(fireflies.copy())
        best_idx = np.argmin(fitness)
        best_firefly = fireflies[best_idx, :].copy()
        best_fitness = fitness[best_idx]
        history.append(best_fitness)

    return best_firefly, history, param_history


# 优化模型参数
lower_bounds = [1000, 1000, 0.1, 30, 9.7, 10e-6, 0.1e-3]
upper_bounds = [12000, 5000, 1, 100, 9.9, 100e-6, 10e-3]

best_params, history, param_history = firefly_algorithm(objective_function, lower_bounds, upper_bounds)

# 打印最优参数
print("最优参数:")
for name, value in zip(problem['names'], best_params):
    print(f"{name}: {value:.4f}")

# 绘制优化过程中的误差变化
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('迭代次数')
plt.ylabel('最优目标值')
plt.title('萤火虫算法优化过程')
plt.show()

# 获取优化过程中的每个参数的值
param_history = np.array(param_history)


# 可视化每个参数在优化过程中的变化
def visualize_optimization_process(history, param_names):
    num_iterations = len(history)
    num_params = len(param_names)

    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params))

    for i in range(num_params):
        for j in range(num_iterations):
            axes[i].plot(range(j * num_fireflies, (j + 1) * num_fireflies), history[j][:, i], 'o', markersize=2)
        axes[i].set_xlabel('迭代次数')
        axes[i].set_ylabel(f'{param_names[i]} 值')
        axes[i].set_title(f'{param_names[i]} 在优化过程中的变化')

    plt.tight_layout()
    plt.show()


# 可视化优化过程
visualize_optimization_process(param_history, problem['names'])

# 更新动力学模型
updated_signal = generate_virtual_signal(best_params)

# 预处理更新后的信号
preprocessed_updated_signal, _ = preprocess_signals(updated_signal, data_real)

# 可视化更新后的仿真信号
plt.figure()
plt.plot(preprocessed_real_signal, label="真实数据")
plt.plot(preprocessed_signal, label="初始仿真数据")
plt.plot(preprocessed_updated_signal, label="更新仿真数据")
plt.xlabel("时间")
plt.ylabel("振动信号")
plt.title("更新后的仿真信号与真实数据")
plt.legend()
plt.show()


# 可视化频率分析的误差随迭代次数的变化
def visualize_error_evolution(history):
    plt.figure()
    plt.plot(history)
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.title('频率分析误差随迭代次数的变化')
    plt.show()


# 可视化误差随迭代次数的变化
visualize_error_evolution(history)


# 可视化虚拟信号与真实信号的频域比较
def visualize_frequency_domain_comparison(signal, real_signal):
    fft_signal = np.abs(fft(signal))
    fft_real_signal = np.abs(fft(real_signal))
    freqs = np.fft.fftfreq(len(signal))

    plt.figure()
    plt.plot(freqs[:len(freqs) // 2], fft_signal[:len(freqs) // 2], label='虚拟信号')
    plt.plot(freqs[:len(freqs) // 2], fft_real_signal[:len(freqs) // 2], label='真实信号')
    plt.xlabel('频率')
    plt.ylabel('幅度')
    plt.title('虚拟信号与真实信号的频域比较')
    plt.legend()
    plt.show()


# 可视化频域比较
visualize_frequency_domain_comparison(virtual_signal, data_real)
visualize_frequency_domain_comparison(updated_signal, data_real)
