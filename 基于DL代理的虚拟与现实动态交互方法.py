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
import torch
import torch.nn as nn
import torch.optim as optim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 构建虚拟模型并生成虚拟信号
def generate_virtual_signal(params, num_samples=6000):
    Fs, f_n, B, m, g, e, rho = params
    bearing = SKF6312(Fs, f_n, B, m, g, e, rho)
    try:
        data_sim = bearing.getSingleSample(rev=1750, ns=num_samples)
        return data_sim['OuterRace']['x']
    except Exception as e:
        print(f"Error generating virtual signal: {e}")
        return None


# 定义初始参数
initial_params = [10000, 3000, 0.5, 50, 9.8, 50e-6, 5e-3]
virtual_signal = generate_virtual_signal(initial_params)

if virtual_signal is None:
    print("Failed to generate initial virtual signal.")
    exit()

# 可视化虚拟信号
plt.figure()
plt.plot(virtual_signal)
plt.xlabel("时间")
plt.ylabel("振动信号")
plt.title("虚拟信号")
plt.show()


# 计算频率分析和相对误差
def frequency_analysis_and_error(signal, real_signal):
    min_length = min(len(signal), len(real_signal))
    signal = signal[:min_length]
    real_signal = real_signal[:min_length]

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
    data_real = data_real[:6000]  # 取前6000个数据点作为测试

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

# 检查预处理后信号的形状
print(f"Preprocessed signal shape: {preprocessed_signal.shape}")
print(f"Preprocessed real signal shape: {preprocessed_real_signal.shape}")

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
    signal = generate_virtual_signal(param_set)
    if signal is not None:
        return signal
    else:
        return np.zeros(6000)


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


# 定义CNN-BiLSTM模型
class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 1), stride=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 1), stride=2)
        self.flatten = nn.Flatten()
        self.bilstm = nn.LSTM(16, 25, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(25 * 2, 1)  # 因为双向LSTM输出维度是hidden_size的两倍

    def forward(self, x):
        x = x.unsqueeze(1)  # 在维度1增加一个维度，以匹配Conv2d的输入形状
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x, _ = self.bilstm(x.unsqueeze(1))
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x


# 定义CustomDataset类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# 检查数据形状并重塑
preprocessed_signal = preprocessed_signal.reshape(-1, 1)
preprocessed_real_signal = preprocessed_real_signal.reshape(-1, 1)

# 创建数据集和数据加载器
train_dataset = CustomDataset(preprocessed_signal, preprocessed_real_signal)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练CNN-BiLSTM模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTM(input_dim=preprocessed_signal.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1).unsqueeze(3)  # 调整输入数据形状
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 绘制训练损失变化
plt.figure()
plt.plot(range(num_epochs), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# 验证模型
model.eval()
with torch.no_grad():
    inputs = torch.tensor(preprocessed_signal, dtype=torch.float32).unsqueeze(1).unsqueeze(3).to(device)
    targets = torch.tensor(preprocessed_real_signal, dtype=torch.float32).to(device)
    outputs = model(inputs)
    predictions = outputs.cpu().numpy()

    # 计算评估指标
    from sklearn.metrics import mean_squared_error, r2_score, pearsonr

    rmse = np.sqrt(mean_squared_error(preprocessed_real_signal, predictions))
    r2 = r2_score(preprocessed_real_signal, predictions)
    pearson_corr, _ = pearsonr(preprocessed_real_signal.flatten(), predictions.flatten())

    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
