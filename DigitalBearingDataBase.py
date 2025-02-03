import numpy as np
from DigitalTwinsFunctions import *
import mat4py
import os


class SKF6312:
    def __init__(self, Fs, f_n, B, m=50.0, g=9.8, e=50 * 1e-06, rho=1e-3):
        # element parameters
        BearingParameter = {'d': 22.2, 'Dm': 96.987, 'alpha': 0, 'z': 8}
        self.d = BearingParameter['d']  # ball diameter
        self.Dm = BearingParameter['Dm']  # pitch diameter
        self.alpha = BearingParameter['alpha']  # contact angle
        self.z = BearingParameter['z']  # number of rolling elements
        self.BearPara = BearingParameter

        # self.rev = 3000.0

        self.Fs = Fs  # sampling frequency
        self.Ns = int(Fs * 5.0)  # sampling number set 10x Fs
        self.f_n = f_n  # nature frequency
        self.B = B  # factor relating to damping
        self.Psi0 = 0
        self.m = m  # Mass of the rotating shaft
        self.g = g  # gravity acceleration
        self.Psi1 = 0
        # self.f0 = 50.0
        self.e = e  # offset
        self.rho = rho  # factor to resize

    def getSingleSample(self, rev, ns):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns,
                                        f_n=self.f_n, B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        x_out, t_out = simRbearing.OuterRace()
        x_in, t_in = simRbearing.InnerRace()
        x_roll, t_roll = simRbearing.RollingElement()

        s_x_out, s_t_out = self.randomSample1Sample(x=x_out, ns=ns, Fs=self.Fs)
        s_x_in, s_t_in = self.randomSample1Sample(x=x_in, ns=ns, Fs=self.Fs)
        s_x_roll, s_t_roll = self.randomSample1Sample(x=x_roll, ns=ns, Fs=self.Fs)

        dataset = {'OuterRace': {'x': s_x_out, 't': s_t_out},
                   'InnerRace': {'x': s_x_in, 't': s_t_in},
                   'Ball': {'x': s_x_roll, 't': s_t_roll}}

        return dataset

    def randomSample1Sample(self, x, ns, Fs):
        index = np.arange(0, len(x) - ns)
        init_ind = np.random.choice(index, size=1)[0]

        sample_x = x[init_ind:init_ind + ns]

        t = np.arange(0, ns) / Fs

        return sample_x, t


class SKF6203:

    def __init__(self, Fs, Ns, f_n, B, m=50.0, g=9.8, e=50 * 1e-06, rho=1e-3):
        # element parameters
        BearingParameter = {'d': 6.7462, 'Dm': 28.4988, 'alpha': 0, 'z': 8}
        self.d = BearingParameter['d']  # ball diameter
        self.Dm = BearingParameter['Dm']  # pitch diameter
        self.alpha = BearingParameter['alpha']  # contact angle
        self.z = BearingParameter['z']  # number of rolling elements
        self.BearPara = BearingParameter

        # self.rev = 3000.0

        self.Fs = Fs  # sampling frequency
        self.Ns = Ns  # sampling number set 10x Fs
        self.f_n = f_n  # nature frequency
        self.B = B  # factor relating to damping
        self.Psi0 = 0
        self.m = m  # Mass of the rotating shaft
        self.g = g  # gravity acceleration
        self.Psi1 = 0
        # self.f0 = 50.0
        self.e = e  # offset
        self.rho = rho  # factor to resize

    def getSingleSample(self, rev, ns):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns,
                                        f_n=self.f_n, B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        x_out, t_out = simRbearing.OuterRace()
        x_in, t_in = simRbearing.InnerRace()
        x_roll, t_roll = simRbearing.RollingElement()

        s_x_out, s_t_out = self.randomSample1Sample(x=x_out, ns=ns, Fs=self.Fs)
        s_x_in, s_t_in = self.randomSample1Sample(x=x_in, ns=ns, Fs=self.Fs)
        s_x_roll, s_t_roll = self.randomSample1Sample(x=x_roll, ns=ns, Fs=self.Fs)

        dataset = {'OuterRace': {'x': s_x_out, 't': s_t_out},
                   'InnerRace': {'x': s_x_in, 't': s_t_in},
                   'Ball': {'x': s_x_roll, 't': s_t_roll}}

        return dataset

    def getSingleSample_OuterRace(self, rev, ns):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns,
                                        f_n=self.f_n, B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        x_out, t_out = simRbearing.OuterRace()
        freq_out = simRbearing.Freq_out
        rev_out = rev
        # x_in, t_in = simRbearing.InnerRace()
        # x_roll, t_roll = simRbearing.RollingElement()

        s_x_out, s_t_out = self.randomSample1Sample(x=x_out, ns=ns, Fs=self.Fs)
        # s_x_in, s_t_in = self.randomSample1Sample(x=x_in, ns=ns, Fs=self.Fs)
        # s_x_roll, s_t_roll = self.randomSample1Sample(x=x_roll, ns=ns, Fs=self.Fs)

        dataset = {'x': s_x_out, 't': s_t_out, 'freq': freq_out, 'rev': rev_out}

        return dataset

    def getSingleSample_InnerRace(self, rev, ns):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns,
                                        f_n=self.f_n, B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        # x_out, t_out = simRbearing.OuterRace()
        # freq_out = simRbearing.Freq_out
        # rev_out = rev
        x_in, t_in = simRbearing.InnerRace()
        freq_in = simRbearing.Freq_in
        rev_in = rev
        # x_roll, t_roll = simRbearing.RollingElement()

        # s_x_out, s_t_out = self.randomSample1Sample(x=x_out, ns=ns, Fs=self.Fs)
        s_x_in, s_t_in = self.randomSample1Sample(x=x_in, ns=ns, Fs=self.Fs)
        # s_x_roll, s_t_roll = self.randomSample1Sample(x=x_roll, ns=ns, Fs=self.Fs)

        dataset = {'x': s_x_in, 't': s_t_in, 'freq': freq_in, 'rev': rev_in}

        return dataset

    def getSingleSample_Ball(self, rev, ns):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns,
                                        f_n=self.f_n, B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        # x_out, t_out = simRbearing.OuterRace()
        # freq_out = simRbearing.Freq_out
        # rev_out = rev
        # x_in, t_in = simRbearing.InnerRace()
        # freq_in = simRbearing.InnerRace()
        # rev_in = rev
        x_ball, t_ball = simRbearing.RollingElement()
        freq_ball = simRbearing.InnerRace()
        rev_ball = rev

        # s_x_out, s_t_out = self.randomSample1Sample(x=x_out, ns=ns, Fs=self.Fs)
        # s_x_in, s_t_in = self.randomSample1Sample(x=x_in, ns=ns, Fs=self.Fs)
        s_x_ball, s_t_ball = self.randomSample1Sample(x=x_ball, ns=ns, Fs=self.Fs)

        dataset = {'x': s_x_ball, 't': s_t_ball, 'freq': freq_ball, 'rev': rev_ball}

        return dataset

    def randomSample1Sample(self, x, ns, Fs):
        index = np.arange(0, len(x) - ns)
        init_ind = np.random.choice(index, size=1)[0]

        sample_x = x[init_ind:init_ind + ns]

        t = np.arange(0, ns) / Fs

        return sample_x, t


class SKF6205:

    def __init__(self, Fs, f_n, B, m=50.0, g=9.8, e=50 * 1e-06, rho=1e-3):
        # element parameters
        BearingParameter = {'d': 7.94, 'Dm': 39.04, 'alpha': 0, 'z': 9}
        self.d = BearingParameter['d']  # ball diameter
        self.Dm = BearingParameter['Dm']  # pitch diameter
        self.alpha = BearingParameter['alpha']  # contact angle
        self.z = BearingParameter['z']  # number of rolling elements
        self.BearPara = BearingParameter

        # self.rev = 3000.0

        self.Fs = Fs  # sampling frequency
        self.Ns = int(Fs * 5.0)  # sampling number set 10x Fs
        self.f_n = f_n  # nature frequency
        self.B = B  # factor relating to damping
        self.Psi0 = 0
        self.m = m  # Mass of the rotating shaft
        self.g = g  # gravity acceleration
        self.Psi1 = 0
        # self.f0 = 50.0
        self.e = e  # offset
        self.rho = rho  # factor to resize

    def getSingleSample(self, rev, ns):
        simRbearing = SimRollingBearing(BearingParameter=self.BearPara, rev=rev, Fs=self.Fs, Ns=self.Ns,
                                        f_n=self.f_n, B=self.B, m=self.m, g=self.g, e=self.e, rho=self.rho)

        x_out, t_out = simRbearing.OuterRace()
        x_in, t_in = simRbearing.InnerRace()
        x_roll, t_roll = simRbearing.RollingElement()

        s_x_out, s_t_out = self.randomSample1Sample(x=x_out, ns=ns, Fs=self.Fs)
        s_x_in, s_t_in = self.randomSample1Sample(x=x_in, ns=ns, Fs=self.Fs)
        s_x_roll, s_t_roll = self.randomSample1Sample(x=x_roll, ns=ns, Fs=self.Fs)

        dataset = {'OuterRace': {'x': s_x_out, 't': s_t_out},
                   'InnerRace': {'x': s_x_in, 't': s_t_in},
                   'Ball': {'x': s_x_roll, 't': s_t_roll}}

        return dataset

    def randomSample1Sample(self, x, ns, Fs):
        index = np.arange(0, len(x) - ns)
        init_ind = np.random.choice(index, size=1)[0]

        sample_x = x[init_ind:init_ind + ns]

        t = np.arange(0, ns) / Fs

        return sample_x, t


class SAMPLINGSample:
    def __init__(self):
        pass

    def random_1(self, x, ns, Fs):
        index = np.arange(0, len(x) - ns)
        init_ind = np.random.choice(index, size=1)[0]

        sample_x = x[init_ind:init_ind + ns]

        t = np.arange(0, ns) / Fs

        return sample_x, t


class CWRU_SKF6203:
    def __init__(self):
        self.normal_path = './0HP/normal'
        self.fault_path = './0HP'
        self.Fs = 12000
        # element parameters
        self.BearingParameter = {'d': 6.7462, 'Dm': 28.4988, 'alpha': 0, 'z': 8}
        # self.rev = rev

    def readNormal(self, load='0'):
        name = 'normal_' + load + '.mat'
        # idx_str = '_DE_'
        idx_str = '_FE_'
        dataset = mat4py.loadmat(os.path.join(self.normal_path, name))
        name = list(dataset.keys())
        s_str = [s for s in name if idx_str in s][0]
        data = np.array(dataset[s_str])
        # RPM_str = 'RPM'
        # rpm_str = [s for s in name if RPM_str in s][0]
        # rev = dataset[rpm_str]
        rev = 1750
        n_d = 240000
        data = data[0:n_d]
        data = data.reshape(-1)
        # n_x = 1200
        # m = 500
        # X_normal, Y_normal = self.data_segmentation(data=data,m=m,n=n_x,label=0)
        # print(X_normal.shape)
        # print(Y_normal.shape)

        return data, rev

    def readOR(self, type='007', load='0', location='@6'):
        # name = 'OR'+ type + '@6_' + load + '.mat'
        name = 'OR' + type + location + '_' + load + '.mat'
        # idx_str = '_DE_'
        idx_str = '_FE_'
        dataset = mat4py.loadmat(os.path.join(self.fault_path, type, name))
        name = list(dataset.keys())
        s_str = [s for s in name if idx_str in s][0]
        data = np.array(dataset[s_str])
        RPM_str = 'RPM'
        rpm_str = [s for s in name if RPM_str in s][0]
        rev = dataset[rpm_str]
        n_d = 120000
        data = data[0:n_d]
        data = data.reshape(-1)
        # n_x = 1200
        # m = 500
        # X_normal, Y_normal = self.data_segmentation(data=data,m=m,n=n_x,label=0)
        # print(X_normal.shape)
        # print(Y_normal.shape)

        return data, rev

    def readBall(self, type='007', load='0'):
        name = 'B' + type + '_' + load + '.mat'
        # idx_str = '_DE_'
        idx_str = '_FE_'

        dataset = mat4py.loadmat(os.path.join(self.fault_path, type, name))
        name = list(dataset.keys())
        s_str = [s for s in name if idx_str in s][0]
        data = np.array(dataset[s_str])
        RPM_str = 'RPM'
        rpm_str = [s for s in name if RPM_str in s][0]
        rev = dataset[rpm_str]
        n_d = 120000
        data = data[0:n_d]
        data = data.reshape(-1)
        # n_x = 1200
        # m = 500
        # X_normal, Y_normal = self.data_segmentation(data=data,m=m,n=n_x,label=0)
        # print(X_normal.shape)
        # print(Y_normal.shape)

        return data, rev

    def readIR(self, type='007', load='0'):
        name = 'IR' + type + '_' + load + '.mat'
        # idx_str = '_DE_'
        idx_str = '_FE_'

        dataset = mat4py.loadmat(os.path.join(self.fault_path, type, name))
        name = list(dataset.keys())
        s_str = [s for s in name if idx_str in s][0]
        data = np.array(dataset[s_str])
        RPM_str = 'RPM'
        rpm_str = [s for s in name if RPM_str in s][0]
        rev = dataset[rpm_str]
        n_d = 120000
        data = data[0:n_d]
        data = data.reshape(-1)
        # n_x = 1200
        # m = 500
        # X_normal, Y_normal = self.data_segmentation(data=data,m=m,n=n_x,label=0)
        # print(X_normal.shape)
        # print(Y_normal.shape)

        return data, rev

    def getSingleSample(self, ns, type, load, Outerlocation='@6'):

        self.d = self.BearingParameter['d']  # ball diameter
        self.Dm = self.BearingParameter['Dm']  # pitch diameter
        self.alpha = self.BearingParameter['alpha']  # contact angle
        self.z = self.BearingParameter['z']  # number of rolling elements

        #
        # self.Freq_out = calFFreq.Freq_out
        # self.Freq_in = calFFreq.Freq_in
        # self.Freq_ball = calFFreq.Freq_roll
        # self.Freq_cage = calFFreq.Freq_cageout
        x_normal, rev_normal = self.readNormal(load=load)
        x_out, rev_out = self.readOR(type=type, load=load, location=Outerlocation)
        x_in, rev_in = self.readIR(type=type, load=load)
        x_roll, rev_roll = self.readBall(type=type, load=load)
        ss = SAMPLINGSample()
        s_x_normal, s_t_normal = ss.random_1(x=x_normal, ns=ns, Fs=self.Fs)
        s_x_out, s_t_out = ss.random_1(x=x_out, ns=ns, Fs=self.Fs)
        s_x_in, s_t_in = ss.random_1(x=x_in, ns=ns, Fs=self.Fs)
        s_x_roll, s_t_roll = ss.random_1(x=x_roll, ns=ns, Fs=self.Fs)

        calFFreq = CalFailureFrequency(rev=1750, d=self.d, Dm=self.Dm, z=self.z, alpha=self.alpha)
        freq_out = calFFreq.OuterRace(rev=rev_out)
        freq_in = calFFreq.InnerRace(rev=rev_in)
        freq_ball = calFFreq.Ball(rev=rev_roll)
        freq_cagein = calFFreq.Cagein(rev=rev_roll)
        freq_cageout = calFFreq.Cageout(rev=rev_roll)

        dataset = {'OuterRace': {'x': s_x_out, 't': s_t_out, 'freq': freq_out, 'rev': rev_out},
                   'InnerRace': {'x': s_x_in, 't': s_t_in, 'freq': freq_in, 'rev': rev_in},
                   'Ball': {'x': s_x_roll, 't': s_t_roll, 'freq': freq_ball, 'rev': rev_roll},
                   'Normal': {'x': s_x_normal, 't': s_t_normal, 'freq': 0, 'rev': rev_normal},
                   'Cage': {'in': freq_cagein, 'out': freq_cageout}}

        return dataset

    def data_segmentation(self, data, m, n, label):
        L = len(data)
        ind = np.arange(0, L - n, int((L - n) / m))
        ind = ind[0:m]
        X = []
        sign = False
        for i in ind:
            temp_X = data[i:i + n]
            # print(temp_X.shape)
            temp_X = temp_X.reshape((1, n))
            # print(temp_X.shape)
            if sign:
                X = np.vstack((X, temp_X))
                # print(X.shape)
            else:
                X = temp_X
                sign = True

        Y = label * np.ones(m, dtype=np.int)
        # print(Y)
        return X, Y
