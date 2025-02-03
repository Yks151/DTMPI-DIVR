import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.stats import pearsonr, spearmanr, kendalltau


class SignalProcsee:
    def __init__(self):
        pass

    def combineSignal(self, x1, x2):
        x = x1 + x2
        return x

    def numpy_minmax(self, x):
        xmin = x.min()
        return (x - xmin) / (x.max() - xmin)

    def numpy_Standard(self, x):
        return (x - np.mean(x)) / np.std(x)

    # def normalize(self, x, type):
    #     pass

    def correlation(self, x1, x2, Fs, type=0, Name='', normalize=True):
        pp = postProcessing()
        temp_x1 = []
        temp_x2 = []
        if type == 0:
            temp_x1 = x1
            temp_x2 = x2
            if normalize:
                temp_x1 = self.numpy_Standard(x=temp_x1)
                temp_x2 = self.numpy_Standard(x=temp_x2)
        elif type == 1:
            _, temp_x1 = pp.myspecfft(x=x1, Fs=Fs)
            _, temp_x2 = pp.myspecfft(x=x2, Fs=Fs)
            if normalize:
                temp_x1 = self.numpy_minmax(x=temp_x1)
                temp_x2 = self.numpy_minmax(x=temp_x2)
        elif type == 2:
            temp_x1 = pp.myenvelope(x=x1)
            temp_x2 = pp.myenvelope(x=x2)
            if normalize:
                temp_x1 = self.numpy_Standard(x=temp_x1)
                temp_x2 = self.numpy_Standard(x=temp_x2)
        elif type == 3:
            _, temp_x1 = pp.env_specfft(x=x1, Fs=Fs)
            _, temp_x2 = pp.env_specfft(x=x2, Fs=Fs)
            if normalize:
                temp_x1 = self.numpy_minmax(x=temp_x1)
                temp_x2 = self.numpy_minmax(x=temp_x2)
        else:
            a = 1
        if normalize:
            print('------------------' + Name + '--Normalize--Type:{}--------'.format(type))
        else:
            print('------------------' + Name + '---Type:{}-------------------'.format(type))

        correlation, p_value = pearsonr(temp_x1, temp_x2)
        print('#Pearsonr correlation: {:.4f}, p_value: {:.4f}. '.format(correlation, p_value))
        correlation, p_value = spearmanr(temp_x1, temp_x2)
        print('#Spearmanr correlation: {:.4f}, p_value: {:.4f}. '.format(correlation, p_value))
        correlation, p_value = kendalltau(temp_x1, temp_x2)
        print('#Kendalltau correlation: {:.4f}, p_value: {:.4f}. '.format(correlation, p_value))
        print('------------------------------------------------')

    def power(self, x):
        tt = np.power(x, 2)
        Ps = np.sum(np.power(x, 2))

        return Ps


class CalFailureFrequency:
    def __init__(self, rev, d, Dm, z, alpha):
        self.rev = rev
        self.d = d  # ball diameter
        self.Dm = Dm  # pitch diameter
        self.alpha = alpha  # contact angle
        self.z = z  # number of rolling elements

        sign = self.forward()

    def forward(self):
        self.Freq_in = self.rev / (120.0) * (1 + self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0)) * self.z
        self.Freq_out = self.rev / (120.0) * (1 - self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0)) * self.z
        self.Freq_roll = self.rev / (60.0) * (self.Dm / self.d) * (
                    1 - np.power(self.d, 2) / np.power(self.Dm, 2) * np.power(np.cos(self.alpha * np.pi / 180.0), 2))
        self.Freq_cagein = self.rev / (120.0) * (1 + self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0))
        self.Freq_cageout = self.rev / (120.0) * (1 - self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0))

        return True

    def OuterRace(self, rev):
        Freq_out = rev / (120.0) * (1 - self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0)) * self.z
        return Freq_out

    def InnerRace(self, rev):
        Freq_in = rev / (120.0) * (1 + self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0)) * self.z
        return Freq_in

    def Cagein(self, rev):
        Freq_cagein = rev / (120.0) * (1 + self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0))
        return Freq_cagein

    def Cageout(self, rev):
        Freq_cageout = rev / (120.0) * (1 - self.d / self.Dm * np.cos(self.alpha * np.pi / 180.0))
        return Freq_cageout

    def Ball(self, rev):
        Freq_roll = rev / (60.0) * (self.Dm / self.d) * \
                    (1 - np.power(self.d, 2) / np.power(self.Dm, 2) * np.power(np.cos(self.alpha * np.pi / 180.0), 2))
        return Freq_roll


class SimRollingBearing:
    def __init__(self, BearingParameter, rev, Fs, Ns, f_n, B, m=50.0, g=9.8, e=50 * 1e-06, rho=1e-3):
        self.Fs = Fs
        self.Ns = Ns
        self.f_n = f_n
        self.B = B
        self.Psi0 = 0
        self.m = m
        self.g = g
        self.Psi1 = 0
        self.f0 = rev / 60.0
        self.e = e
        self.rho = rho

        self.rev = rev
        self.d = BearingParameter['d']  # ball diameter
        self.Dm = BearingParameter['Dm']  # pitch diameter
        self.alpha = BearingParameter['alpha']  # contact angle
        self.z = BearingParameter['z']  # number of rolling elements
        calFFreq = CalFailureFrequency(rev=self.rev, d=self.d, Dm=self.Dm, z=self.z, alpha=self.alpha)

        self.Freq_out = calFFreq.Freq_out
        self.Freq_in = calFFreq.Freq_in
        self.Freq_ball = calFFreq.Freq_roll
        self.Freq_cage = calFFreq.Freq_cageout
        self.Freq_cagein = calFFreq.Freq_cagein
        self.Freq_cageout = calFFreq.Freq_cageout

        a = 1

    def OuterRace(self):

        F_freq = self.Freq_out
        # % impulseseries
        t = np.linspace(1, self.Ns, self.Ns) / self.Fs
        delta_t = np.round(self.Fs / F_freq)
        n = len(t)
        U_impulse = np.zeros((n))
        for i in range(0, n):
            if np.mod(i, delta_t) == 0:
                U_impulse[i] = 1

        # % decaying oscillation
        s_t = np.exp(-self.B * t) * np.cos(2 * np.pi * self.f_n * t)

        # % determinate load
        M = self.m * self.g * np.cos(self.Psi0 / 180.0 * np.pi + self.Psi0)

        # % determinate Amplitude
        Am = self.rho * M

        # % alternate load

        omega = 60 * self.f0 * 2 * np.pi / 60.0
        T = self.m * self.e * np.power(omega, 2) * np.cos(2 * np.pi * self.f0 * t + self.Psi1 / 180.0 * np.pi)
        At = self.rho * T
        A = Am + At

        A_implulse = A * U_impulse
        Sign_out = np.convolve(A_implulse, s_t)
        Sign_out = Sign_out[0:self.Ns]

        return Sign_out, t

    def InnerRace(self):

        F_freq = self.Freq_in
        # % impulseseries
        t = np.linspace(1, self.Ns, self.Ns) / self.Fs
        delta_t = np.round(self.Fs / F_freq)
        n = len(t)
        U_impulse = np.zeros((n))
        for i in range(0, n):
            if np.mod(i, delta_t) == 0:
                U_impulse[i] = 1

        # % decaying oscillation
        s_t = np.exp(-self.B * t) * np.cos(2 * np.pi * self.f_n * t)

        # % determinate load

        M = self.m * self.g * np.cos(2 * np.pi * self.f0 * t + self.Psi0)

        # % determinate Amplitude
        Am = self.rho * M

        # % alternate load

        omega = 60 * self.f0 * 2 * np.pi / 60.0
        T = self.m * self.e * np.power(omega, 2) * np.cos(self.Psi1)
        At = self.rho * T
        A = Am + At

        A_implulse = A * U_impulse
        Sign_in = np.convolve(A_implulse, s_t)
        Sign_in = Sign_in[0:self.Ns]

        return Sign_in, t

    def RollingElement(self):

        F_freq = self.Freq_ball
        # % impulseseries
        t = np.linspace(1, self.Ns, self.Ns) / self.Fs
        delta_t = np.round(self.Fs / F_freq)
        n = len(t)
        U_impulse = np.zeros((n))
        for i in range(0, n):
            if np.mod(i, delta_t) == 0:
                U_impulse[i] = 1

        # % decaying oscillation
        s_t = np.exp(-self.B * t) * np.cos(2 * np.pi * self.f_n * t)

        # % determinate load
        M = self.m * self.g * np.cos(2 * np.pi * self.Freq_cage * t + self.Psi0)

        # % determinate Amplitude
        Am = self.rho * M

        # % alternate load
        Psi = 2 * np.pi * (self.f0 - self.Freq_cage) * t + self.Psi1

        omega = 60 * self.f0 * 2 * np.pi / 60.0
        T = self.m * self.e * np.power(omega, 2) * np.cos(Psi)

        At = self.rho * T
        A = Am + At

        A_implulse = A * U_impulse
        Sign_Roll = np.convolve(A_implulse, s_t)
        Sign_Roll = Sign_Roll[0:self.Ns]

        return Sign_Roll, t


class PlotResults:

    def __init__(self, num_figure):
        fig_list, ax_list = [], []
        for i in range(num_figure):
            t_fig, t_ax = plt.subplots(1)
            fig_list.append(t_fig)
            ax_list.append(t_ax)
        self.fig_list, self.ax_list = fig_list, ax_list

        plt.ion()

    def Close(self):
        # 关闭交互模式
        plt.ioff()
        # plt.close()
        plt.close('all')

    def plotSimSignal(self, x, t, Fs, ind_figure=0):
        fig = self.fig_list[ind_figure]
        ax = self.ax_list[ind_figure]

        pp = postProcessing()

        x = x
        f, Af = pp.myspecfft(x=x, Fs=Fs, ifPlot=False)
        hx = pp.myenvelope(x=x)
        f_hx, Af_hx = pp.myspecfft(x=hx, Fs=Fs, ifPlot=False)
        fig.clf()
        fig.suptitle(' ')
        ax.cla()
        # ax.clf()
        ax = fig.add_subplot(221)
        ax.plot(t, x)
        # ax.set_xlim(0,0.1)
        ax = fig.add_subplot(222)
        ax.plot(f, Af)
        ax = fig.add_subplot(223)
        ax.plot(t, hx)
        # ax.set_xlim(0,0.1)
        ax = fig.add_subplot(224)
        ax.plot(f_hx, Af_hx)
        ax.set_xlim(0, 1000)
        # fig, ax = plt.subplots()
        # ax_i.plot(t, predRUL_mid, '.', label='y_pred')
        # ax.plot(t, predRUL_mid, marker='o', label='y_pred')
        # ax.fill_between(t, predRUL_down, predRUL_upper, alpha=0.2)
        # ax.plot(t, trueRUL, '-.', color='black', label='y_true')
        # ax_i.set_title(name)
        # ax_i.set_xlabel(fontsize=16)
        # ax_i.set_ylabel(fontsize=16)
        # plt.setp(ax.get_xticklabels(), fontsize=14)
        # plt.setp(ax.get_yticklabels(), fontsize=14)

        # ax.legend(loc='lower left', fontsize=14)
        # plt.show()
        plt.pause(0.1)
        a = 1

        # plt.subplot(2, 2, 1)
        # plt.plot(t, x)
        # plt.xlim(0, 0.1)
        # plt.subplot(2, 2, 2)
        # plt.plot(f, Af)
        # plt.subplot(2, 2, 3)
        # plt.plot(t, hx)
        # plt.xlim(0, 0.1)
        # plt.subplot(2, 2, 4)
        # plt.plot(f_hx, Af_hx)
        # plt.xlim(0,1000)
        # # plt.ylabel('Frequency (Hz)')
        # # plt.xlabel('|X(f)|')
        # # plt.title("Single-Sided Amplitude Spectrum of x(t)")
        # plt.show()

    def plotRealSignal(self, x, t, Fs, freq, rev, type, ind_figure=0, supple_info=None):
        fig = self.fig_list[ind_figure]
        ax = self.ax_list[ind_figure]

        pp = postProcessing()

        # x = x
        f, Af = pp.myspecfft(x=x, Fs=Fs, ifPlot=False)
        hx = pp.myenvelope(x=x)
        f_hx, Af_hx = pp.myspecfft(x=hx, Fs=Fs, ifPlot=False)
        fig.clf()
        title = type + ' freq:' + str(np.around(freq, 4)) + ' rev:' + str(rev)
        if supple_info is not None:
            title = title + ' ' + supple_info
        # fig.suptitle(title)
        fig.suptitle('')
        ax.cla()
        # ax.clf()
        ax = fig.add_subplot(221)
        ax.plot(t, x)
        ax.set_title('(a)')
        # ax.set_xlabel('Time/s')
        ax.set_ylabel('Amp')

        # ax.set_xlim(0,0.1)
        ax = fig.add_subplot(222)
        ax.plot(f, Af)
        ax.set_title('(b)')
        # ax.set_xlabel('Frequency/Hz')
        # ax.set_ylabel('Amp')
        ax = fig.add_subplot(223)
        ax.plot(t, hx)
        ax.set_title('(c)')
        ax.set_xlabel('Time/s')
        ax.set_ylabel('Amp')
        # ax.set_xlim(0,0.1)
        ax = fig.add_subplot(224)
        ax.plot(f_hx, Af_hx)
        ax.set_xlim(0, 1000)
        ax.set_title('(d)')
        ax.set_xlabel('Frequency/Hz')
        fig.tight_layout()
        # ax.set_ylabel('Amp')
        # fig, ax = plt.subplots()
        # ax_i.plot(t, predRUL_mid, '.', label='y_pred')
        # ax.plot(t, predRUL_mid, marker='o', label='y_pred')
        # ax.fill_between(t, predRUL_down, predRUL_upper, alpha=0.2)
        # ax.plot(t, trueRUL, '-.', color='black', label='y_true')
        # ax_i.set_title(name)
        # ax_i.set_xlabel(fontsize=16)
        # ax_i.set_ylabel(fontsize=16)
        # plt.setp(ax.get_xticklabels(), fontsize=14)
        # plt.setp(ax.get_yticklabels(), fontsize=14)
        fig.savefig(os.path.join('./results', type + '.png'), dpi=300)
        # ax.legend(loc='lower left', fontsize=14)
        plt.show()

        # fig.savefig(os.path.join('./results', type + '.png'), dpi=300)
        # savefig(os.path.join(path, m_n + '_' + str(i) + '.png'), dpi=300)
        # plt.pause(0.1)
        # a = 1

        # plt.subplot(2, 2, 1)
        # plt.plot(t, x)
        # plt.xlim(0, 0.1)
        # plt.subplot(2, 2, 2)
        # plt.plot(f, Af)
        # plt.subplot(2, 2, 3)
        # plt.plot(t, hx)
        # plt.xlim(0, 0.1)
        # plt.subplot(2, 2, 4)
        # plt.plot(f_hx, Af_hx)
        # plt.xlim(0,1000)
        # # plt.ylabel('Frequency (Hz)')
        # # plt.xlabel('|X(f)|')
        # # plt.title("Single-Sided Amplitude Spectrum of x(t)")
        # plt.show()


class postProcessing:
    def __init__(self):
        pass

    def plotSimSignal(self, x, t, Fs):

        x = x
        f, Af = self.myspecfft(x=x, Fs=Fs, ifPlot=False)
        hx = self.myenvelope(x=x)
        f_hx, Af_hx = self.myspecfft(x=hx, Fs=Fs, ifPlot=False)

        plt.subplot(2, 2, 1)
        plt.plot(t, x)
        plt.xlim(0, 0.1)
        plt.subplot(2, 2, 2)
        plt.plot(f, Af)
        plt.subplot(2, 2, 3)
        plt.plot(t, hx)
        plt.xlim(0, 0.1)
        plt.subplot(2, 2, 4)
        plt.plot(f_hx, Af_hx)
        plt.xlim(0, 1000)
        # plt.ylabel('Frequency (Hz)')
        # plt.xlabel('|X(f)|')
        # plt.title("Single-Sided Amplitude Spectrum of x(t)")
        plt.show()

    def env_specfft(self, x, Fs):
        # f, Af = self.myspecfft(x=x, Fs=Fs, ifPlot=False)
        hx = self.myenvelope(x=x)
        f_hx, Af_hx = self.myspecfft(x=hx, Fs=Fs, ifPlot=False)

        return f_hx, Af_hx

    def myspecfft(self, x, Fs, ifPlot=False):

        L = x.shape[0]
        x = x.reshape(L)
        x = x - np.mean(x)
        X = np.fft.fft(x) / L
        f = Fs / 2 * np.linspace(0, 1, int(L / 2))
        Af = 2 * np.abs(X[0:int(L / 2)])

        if ifPlot:
            plt.subplot(2, 1, 1)
            plt.plot(x)
            plt.subplot(2, 1, 2)
            plt.plot(f, Af)
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('|X(f)|')
            plt.title("Single-Sided Amplitude Spectrum of x(t)")
            plt.show()
        # Spectrum = {'Af': Af, "f": f}
        return f, Af

    def myenvelope(self, x, ifPlot=False):

        # analytic_signal = hilbert(x)
        # amplitude_envelope = np.abs(analytic_signal)
        hx = fftpack.hilbert(x)
        Hx = np.sqrt(x ** 2 + hx ** 2)

        if ifPlot:
            plt.subplot(2, 1, 1)
            plt.plot(x)
            plt.subplot(2, 1, 2)
            plt.plot(Hx)
            # plt.ylabel('Frequency (Hz)')
            plt.xlabel('data')
            plt.title("Hilbert Wave of x(t)")
            plt.show()

        Envelope = {"Hx": Hx}

        return Hx

    def list2numpy(self, X):

        X_np = np.vstack(tuple(X))

        return X_np
