import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, \
    QMainWindow, QWidget, QMessageBox, QFileDialog
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtCore import QTimer
from scipy.signal import hilbert, butter, filtfilt
import pywt
from DigitalTwinsFunctions import SimRollingBearing, CalFailureFrequency
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mat4py
import pandas as pd

class SKF6312:
    def __init__(self, Fs, f_n, B, m=50.0, g=9.8, e=50 * 1e-06, rho=1e-3):
        BearingParameter = {'d': 22.2, 'Dm': 96.987, 'alpha': 0, 'z': 8}
        self.d = BearingParameter['d']
        self.Dm = BearingParameter['Dm']
        self.alpha = BearingParameter['alpha']
        self.z = BearingParameter['z']
        self.BearPara = BearingParameter

        self.Fs = Fs
        self.Ns = int(Fs * 5.0)
        self.f_n = f_n
        self.B = B
        self.Psi0 = 0
        self.m = m
        self.g = g
        self.Psi1 = 0
        self.e = e
        self.rho = rho

    def getSingleSample(self, rev, ns, fault_position='外圈故障', fault_type='早期点蚀', fault_severity='轻微',
                        vibration_freq=5000, noise_level=0.01, filter_type='低通滤波器', cutoff_freq=2000):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns, f_n=self.f_n,
                                        B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        if fault_position == '外圈故障':
            x_fault, t_fault = simRbearing.OuterRace()
        elif fault_position == '内圈故障':
            x_fault, t_fault = simRbearing.InnerRace()
        elif fault_position == '滚子故障':
            x_fault, t_fault = simRbearing.RollingElement()

        if fault_type == '早期点蚀':
            x_fault = SimRollingBearing.CalFailurePosition(faultType='早期点蚀', x=x_fault, D=self.Dm, d=self.d, r=50e-06)
        elif fault_type == '完全剥落':
            x_fault = SimRollingBearing.CalFailurePosition(faultType='完全剥落', x=x_fault, D=self.Dm, d=self.d, r=50e-06)

        x_fault = SimRollingBearing.FaultSeverity(e=self.e, B=self.B, g=self.g, x=x_fault, Dm=self.Dm, m=self.m,
                                                  Psi0=self.Psi0, rev=rev, rho=self.rho, t=t_fault, Fs=self.Fs,
                                                  f_n=self.f_n, Psi1=self.Psi1, z=self.z, fault_severity=fault_severity)

        x_fault = self.addNoise(x_fault, noise_level=noise_level)
        x_fault = self.filterSignal(x_fault, filter_type=filter_type, cutoff_freq=cutoff_freq)

        s_x_fault, s_t_fault = self.randomSample1Sample(x=x_fault, ns=ns, Fs=self.Fs)
        s_t_fault = s_t_fault[:ns]

        # Simulate sound signal from vibration signal
        s_x_sound = self.simulateSoundSignal(s_x_fault)

        dataset = {'Fault': {'x': s_x_fault, 't': s_t_fault}, 'Sound': {'x': s_x_sound, 't': s_t_fault}}

        return dataset

    def simulateSoundSignal(self, vibration_signal):
        sound_amplitude = 0.1  # Sound signal is weaker than the vibration signal
        return sound_amplitude * vibration_signal

    def randomSample1Sample(self, x, ns, Fs):
        index = np.arange(0, len(x) - ns)
        init_ind = np.random.choice(index, size=1)[0]
        sample_x = x[init_ind:init_ind + ns]
        t = np.arange(0, ns) / Fs
        return sample_x, t

    def addNoise(self, x, noise_level=0.01):
        noise = noise_level * np.random.normal(size=len(x))
        return x + noise

    def filterSignal(self, x, filter_type='低通滤波器', cutoff_freq=2000):
        if filter_type == '低通滤波器':
            b, a = butter(4, cutoff_freq / (self.Fs / 2), btype='low', analog=False)
            return filtfilt(b, a, x)
        elif filter_type == '高通滤波器':
            b, a = butter(4, cutoff_freq / (self.Fs / 2), btype='high', analog=False)
            return filtfilt(b, a, x)
        elif filter_type == '带通滤波器':
            b, a = butter(4, [cutoff_freq[0] / (self.Fs / 2), cutoff_freq[1] / (self.Fs / 2)], btype='band', analog=False)
            return filtfilt(b, a, x)
        elif filter_type == '带阻滤波器':
            b, a = butter(4, [cutoff_freq[0] / (self.Fs / 2), cutoff_freq[1] / (self.Fs / 2)], btype='bandstop', analog=False)
            return filtfilt(b, a, x)
        else:
            return x

    def importRealData(self, file_path, key='data', columns=None, signal_type='Fault'):
        try:
            if file_path.endswith('.mat'):
                data = mat4py.loadmat(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format.")

            if columns:
                data = data[columns]

            x = data[key]
            t = np.arange(len(x)) / self.Fs

            if signal_type == 'Fault':
                x = self.filterSignal(x, filter_type='低通滤波器', cutoff_freq=2000)
                s_x_fault, s_t_fault = self.randomSample1Sample(x=x, ns=len(x), Fs=self.Fs)
                s_t_fault = s_t_fault[:len(x)]

                # Simulate sound signal from vibration signal
                s_x_sound = self.simulateSoundSignal(s_x_fault)

                dataset = {'Fault': {'x': s_x_fault, 't': s_t_fault}, 'Sound': {'x': s_x_sound, 't': s_t_fault}}

                return dataset
            else:
                return {'x': x, 't': t}  # Return raw data
        except Exception as e:
            print(f"Error importing real data: {e}")
            return None

class SKF6203:
    def __init__(self, Fs, Ns, f_n, B, m=50.0, g=9.8, e=50 * 1e-06, rho=1e-3):
        BearingParameter = {'d': 6.7462, 'Dm': 28.4988, 'alpha': 0, 'z': 8}
        self.d = BearingParameter['d']
        self.Dm = BearingParameter['Dm']
        self.alpha = BearingParameter['alpha']
        self.z = BearingParameter['z']
        self.BearPara = BearingParameter

        self.Fs = Fs
        self.Ns = Ns
        self.f_n = f_n
        self.B = B
        self.Psi0 = 0
        self.m = m
        self.g = g
        self.Psi1 = 0
        self.e = e
        self.rho = rho

    def getSingleSample(self, rev, ns, fault_position='外圈故障', fault_type='早期点蚀', fault_severity='轻微',
                        vibration_freq=5000, noise_level=0.01, filter_type='低通滤波器', cutoff_freq=2000):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns, f_n=self.f_n,
                                        B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        if fault_position == '外圈故障':
            x_fault, t_fault = simRbearing.OuterRace()
        elif fault_position == '内圈故障':
            x_fault, t_fault = simRbearing.InnerRace()
        elif fault_position == '滚子故障':
            x_fault, t_fault = simRbearing.RollingElement()

        if fault_type == '早期点蚀':
            x_fault = SimRollingBearing.CalFailurePosition(faultType='早期点蚀', x=x_fault, D=self.Dm, d=self.d, r=50e-06)
        elif fault_type == '完全剥落':
            x_fault = SimRollingBearing.CalFailurePosition(faultType='完全剥落', x=x_fault, D=self.Dm, d=self.d, r=50e-06)

        x_fault = SimRollingBearing.FaultSeverity(e=self.e, B=self.B, g=self.g, x=x_fault, Dm=self.Dm, m=self.m,
                                                  Psi0=self.Psi0, rev=rev, rho=self.rho, t=t_fault, Fs=self.Fs,
                                                  f_n=self.f_n, Psi1=self.Psi1, z=self.z, fault_severity=fault_severity)

        x_fault = self.addNoise(x_fault, noise_level=noise_level)
        x_fault = self.filterSignal(x_fault, filter_type=filter_type, cutoff_freq=cutoff_freq)

        s_x_fault, s_t_fault = self.randomSample1Sample(x=x_fault, ns=ns, Fs=self.Fs)
        s_t_fault = s_t_fault[:ns]

        # Simulate sound signal from vibration signal
        s_x_sound = self.simulateSoundSignal(s_x_fault)

        dataset = {'Fault': {'x': s_x_fault, 't': s_t_fault}, 'Sound': {'x': s_x_sound, 't': s_t_fault}}

        return dataset

    def simulateSoundSignal(self, vibration_signal):
        sound_amplitude = 0.1  # Sound signal is weaker than the vibration signal
        return sound_amplitude * vibration_signal

    def randomSample1Sample(self, x, ns, Fs):
        index = np.arange(0, len(x) - ns)
        init_ind = np.random.choice(index, size=1)[0]
        sample_x = x[init_ind:init_ind + ns]
        t = np.arange(0, ns) / Fs
        return sample_x, t

    def addNoise(self, x, noise_level=0.01):
        noise = noise_level * np.random.normal(size=len(x))
        return x + noise

    def filterSignal(self, x, filter_type='低通滤波器', cutoff_freq=2000):
        if filter_type == '低通滤波器':
            b, a = butter(4, cutoff_freq / (self.Fs / 2), btype='low', analog=False)
            return filtfilt(b, a, x)
        elif filter_type == '高通滤波器':
            b, a = butter(4, cutoff_freq / (self.Fs / 2), btype='high', analog=False)
            return filtfilt(b, a, x)
        elif filter_type == '带通滤波器':
            b, a = butter(4, [cutoff_freq[0] / (self.Fs / 2), cutoff_freq[1] / (self.Fs / 2)], btype='band', analog=False)
            return filtfilt(b, a, x)
        elif filter_type == '带阻滤波器':
            b, a = butter(4, [cutoff_freq[0] / (self.Fs / 2), cutoff_freq[1] / (self.Fs / 2)], btype='bandstop', analog=False)
            return filtfilt(b, a, x)
        else:
            return x

    def importRealData(self, file_path, key='data', columns=None, signal_type='Fault'):
        try:
            if file_path.endswith('.mat'):
                data = mat4py.loadmat(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format.")

            if columns:
                data = data[columns]

            x = data[key]
            t = np.arange(len(x)) / self.Fs

            if signal_type == 'Fault':
                x = self.filterSignal(x, filter_type='低通滤波器', cutoff_freq=2000)
                s_x_fault, s_t_fault = self.randomSample1Sample(x=x, ns=len(x), Fs=self.Fs)
                s_t_fault = s_t_fault[:len(x)]

                # Simulate sound signal from vibration signal
                s_x_sound = self.simulateSoundSignal(s_x_fault)

                dataset = {'Fault': {'x': s_x_fault, 't': s_t_fault}, 'Sound': {'x': s_x_sound, 't': s_t_fault}}

                return dataset
            else:
                return {'x': x, 't': t}  # Return raw data
        except Exception as e:
            print(f"Error importing real data: {e}")
            return None

class DataTwinsPlatform(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('数据孪生平台')
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QVBoxLayout()

        # File import layout
        file_layout = QHBoxLayout()
        self.file_label = QLabel('选择文件:')
        self.file_input = QLineEdit(self)
        self.file_button = QPushButton('浏览', self)
        self.file_button.clicked.connect(self.loadFile)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_button)

        # Signal selection layout
        signal_layout = QHBoxLayout()
        self.signal_label = QLabel('信号类型:')
        self.signal_combo = QComboBox(self)
        self.signal_combo.addItems(['Fault', 'Sound'])
        signal_layout.addWidget(self.signal_label)
        signal_layout.addWidget(self.signal_combo)

        # Import and plot button
        self.import_button = QPushButton('导入并绘制', self)
        self.import_button.clicked.connect(self.importAndPlotData)

        # Plot area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Add all widgets to main layout
        main_layout.addLayout(file_layout)
        main_layout.addLayout(signal_layout)
        main_layout.addWidget(self.import_button)
        main_layout.addWidget(self.canvas)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def loadFile(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件', '', 'All Files (*);;MAT Files (*.mat);;Excel Files (*.xlsx);;CSV Files (*.csv)', options=options)
        if file_path:
            self.file_input.setText(file_path)

    def importAndPlotData(self):
        file_path = self.file_input.text()
        signal_type = self.signal_combo.currentText()

        bearing = SKF6312(Fs=12000, f_n=30, B=10)

        data = bearing.importRealData(file_path, signal_type=signal_type)

        if data:
            x = data[signal_type]['x']
            t = data[signal_type]['t']

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(t, x)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{signal_type} Signal')
            self.canvas.draw()
        else:
            QMessageBox.warning(self, 'Error', '数据导入失败，请检查文件格式或内容。')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    platform = DataTwinsPlatform()
    platform.show()
    sys.exit(app.exec_())
