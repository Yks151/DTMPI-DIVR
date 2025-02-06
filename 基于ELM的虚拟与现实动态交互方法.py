import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DigitalBearingDataBase import SKF6312, CWRU_SKF6203
import os
from scipy.fftpack import fft
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy.signal import butter, filtfilt
from SALib.sample import saltelli
from SALib.analyze import sobol
from elm import ELM

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 构建虚拟模型并生成虚拟信号
def generate_virtual_signal(params, num_samples=10000):
    Fs, f_n, B, m, g, e, rho = params
    bearing = SKF6312(Fs, f_n, B, m, g, e, rho)
    data_sim = bearing.getSingleSample(rev=1750, ns=num_samples)
    return data_sim['OuterRace']['x']

# 定义初始参数
initial_params = [10000, 3000, 0.5, 50, 9.8, 50e-6, 5e-3]
virtual_signal = generate_virtual_signal(initial_params)

# 可视化虚拟信号
plt.figure()
plt.plot(virtual_signal)
plt.xlabel("时间")
plt.ylabel("振动信号")
plt.title("虚拟信号")
plt.show()

# 计算频率分析和相对误差
def frequency_analysis_and_error(signal, real_signal):
    fft_signal = np.abs(fft(signal))
    fft_real_signal = np.abs(fft(real_signal))

    # 计算相对误差
    relative_error = mean_absolute_percentage_error(fft_real_signal, fft_signal)
    return relative_error

# 加载真实数据
cwru = CWRU_SKF6203()
file_path = r'E:/可解释性故障诊断学习/Fault-Diagnosis-Predictive-Maintainence-Digital-Twin-master-main/0HP/007/OR007@6_0.mat'

if os.path.exists(file_path):
    data_real, _ = cwru.readOR(type='007', load='0', location='@6')
    data_real = data_real[:10000]  # 取前10000个数据点作为测试

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

Y = np.array([frequency_analysis_and_error(simulate(param_set), data_real) for param_set in param_values])

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

# 使用极限学习机（ELM）建立代理模型
elm = ELM(param_values.shape[1], 1)
elm.add_neurons(20, "sigm")
elm.train(param_values, Y)

# 预测并分析拟合精度
y_pred = elm.predict(param_values).flatten()
mse = mean_squared_error(Y, y_pred)
print(f"ELM Model Fitting MSE: {mse:.4f}")

# 目标函数：使用代理模型预测误差
def objective_function(params):
    params = np.array(params).reshape(1, -1)
    pred = elm.predict(params).flatten()
    return pred[0]

# 萤火虫算法参数
num_fireflies = 50  # 萤火虫数量
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
                    beta = beta0 * np.exp(-gamma * r **2)
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

print("最优参数:")
print("Fs:", best_params[0])
print("f_n:", best_params[1])
print("B:", best_params[2])
print("m:", best_params[3])
print("g:", best_params[4])
print("e:", best_params[5])
print("rho:", best_params[6])

# 可视化优化过程
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.title('萤火虫算法优化过程')
plt.show()

# 使用最优参数生成虚拟信号并与真实信号对比
optimized_signal = generate_virtual_signal(best_params)

# 可视化优化后的虚拟信号与真实信号对比
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(optimized_signal)
plt.xlabel("时间")
plt.ylabel("优化后的虚拟信号")
plt.title("优化后的虚拟信号")

plt.subplot(2, 1, 2)
plt.plot(data_real)
plt.xlabel("时间")
plt.ylabel("真实信号")
plt.title("真实信号")
plt.tight_layout()
plt.show()
