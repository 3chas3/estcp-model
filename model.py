#!/usr/bin/python3

'''
A model of two-stage immersion cooling. A heat source warms a bath of
fluorinert whichs boils. A water-cooled condenser above the bath removes
the heat from the vapor. The heat in the water is removed via drycooler.
Ideally, the water loop is kept as warm as possible to maximum exchange
with the ambient air and keep the drycooler power consumption as low
as possible.

This model was written to estimate the potential behavior of the real
system outlined in the research project below:

https://serdp-estcp.mil/projects/details/3a1c268b-0139-4424-a7a0-7f850df52aad
'''

import math
import scipy.optimize

# Correction factor for 25% glycol at 120F
correctionFactor = 0.95

const_KtoF = 9.0/5.0


def JoulesToBTUH(joules):
    return joules / 1055.0


def JoulesToKwH(joules):
    return joules / 3600.0 / 1000.0


def KwHtoJoules(kwh):
    return kwh * 3600.0 * 1000.0


def CtoF(c):
    return (c * const_KtoF) + 32


def KtoF(k):
    return k * const_KtoF


def FtoC(f):
    return (f - 32) / const_KtoF


def density_air(T):
    # A linear approximation between these two points
    # 0C    1.293
    # 40C    1.127
    return 1.293 + (T/40.0)*(1.127 - 1.293)


def calc_C_air(cfm, T):
    # The specific heat of air at 35C: 1.005 J/(g*K)
    # The density of air at 40C 1127.0 g/m^3
    # 1 ft^3 = 0.0283168 m^3
    #
    # Cp = cfm * 60 * 1.005 * 0.0283168 * 1127.0    # J/(H*K)
    rho_air = density_air(T) * 1000.0
    Cp = cfm * 60 * 1.005 * 0.0283168 * rho_air    # J/(H*K)
    return Cp


def calc_C_water(gpm, T):
    # The specific heat of water at 45C: 4.181 J/(g*K)
    # 1 gallon = 3785 g
    Cp = gpm * 60 * 4.181 * correctionFactor * 3785        # J/(H*K)
    return Cp


class Heatexchanger_eNTU:
    '''
    A heatexchanger model using the number of transfer units (NTU) method

    q = E*C_min*(T_h,i - T_c,i)
    E = effectiveness (measured)
    '''

    def __init__(self, E):
        self.E = E

    def iterate(self, T_h_i, T_c_i, GPM, CFM):
        C_water = calc_C_water(GPM, T_c_i)
        C_air = calc_C_air(CFM, T_c_i)
        C_min = min(C_air, C_water)
        q_max = C_min * (T_h_i - T_c_i)

        # heat transferred by the fluids
        q = self.E * q_max

        # heat transferred from water
        T_h_o = T_h_i - (q/C_water)

        # heat transferred to air
        T_c_o = T_c_i + (q/C_air)

        return (q, T_h_o, T_c_o)    # Joules, C, C


def T_h_from_LMTD(T_h, T_c_in, T_c_out, lmtd):
    def LMTD(dt_o, dt_i):
        return (dt_o - dt_i)/math.log(dt_o/dt_i)

    return lmtd - LMTD(T_h - T_c_out, T_h - T_c_in)


class Heatexchanger_LMTD:
    '''
    A heatexchanger using the log mean temperature difference (LMTD) method

    q = U * A * F * lmtd
    lmtd = q / (U * A * F)
    '''
    def __init__(self, U, A, F):
        self.U = U
        self.A = A
        self.F = F

    def LMTD(self, q):
        return q / (self.U * self.A * self.F)

    def calc_T_h(self, T_c_in, T_c_out, lmtd):
        #
        #         T_h
        #
        #       |||||||
        #       |||||||== T_c_out
        #       |||||||
        #       |||||||== T_c_in
        #       |||||||
        #
        #         T_h
        #
        #  delta_T1 = T_h - T_c_in
        #  delta_T2 = T_h - T_c_out
        #
        a = T_c_out + 0.001
        b = 50.0

        # bisect(f, a, b, args)
        # f: function returning a number
        # a: one end of the bracketing interval [a, b]
        # b: other end of the bracketing interval [a, b]
        # args: tuple, extra arguments for function f: apply(f, (x) + args)
        T_c_out = scipy.optimize.bisect(T_h_from_LMTD, a, b,
                                        (T_c_in, T_c_out, lmtd))
        return T_c_out


class EstcpModel:
    def __init__(self):
        self.num_timesteps = 288    # Timesteps in a day

        # Pump
        self.GPM = 31.7

        # Drycooler
        self.VFD = 1.0
        self.CFM = 11053
        self.Valve_drycooler = 1.0
        self.GPM_drycooler = self.GPM * self.Valve_drycooler
        self.drycooler = Heatexchanger_eNTU(E=0.515)

        # Bath
        self.Q_bath = 28.8      # kW
        VaporHeight = 0.9       # Assume no heat exchange in the superheat region
        self.bath = Heatexchanger_LMTD(U=1.0, A=25000000.0 * VaporHeight, F=1.0)
        self.Valve_bath = 1.0
        self.GPM_bath = self.GPM * self.Valve_bath
        self.targetSupply = 37.777  # 100F, well below the boiling point
                                    # of fluorinert (133F)

        self.T_initial = FtoC(72.0)  # F
        self.T_range = 14.0  # A 50-degree swing

        self.T_ambient = self.T_initial
        self.T_in_bath = self.T_ambient

        self.clock = 0
        self.last_T_out_bath = 0.0
        self.last_T_out_drycooler = 0.0

    def kW_drycooler(self):
        '''
        min is 2.9A, max is 4.0A
        '''
        current = ((self.VFD - 0.1) * (4.0 - 2.9) / 0.9) + 2.9
        return (current * math.sqrt(3) * 277.0) / 1000.0

    def timestep(self):
        iterations = 0
        converged = 0
        while True:
            iterations = iterations + 1

            # energy balance    q = c_min(T_b - T_a)
            C_water = calc_C_water(self.GPM, -1)
            dT = KwHtoJoules(self.Q_bath)/C_water
            self.T_out_bath = self.T_in_bath + dT

            # bath
            lmtd = self.bath.LMTD(KwHtoJoules(self.Q_bath))
            if converged > 0:
                self.T_vapor = self.bath.calc_T_h(self.T_in_bath, self.T_out_bath, lmtd)
            else:
                self.T_vapor = -1.0

            # dry cooler
            self.T_in_drycooler = self.T_out_bath
            self.Q_drycooler, self.T_out_drycooler, self.T_air_out = self.drycooler.iterate(
                                            self.T_in_drycooler,
                                            self.T_ambient,
                                            self.GPM_drycooler,
                                            self.CFM * self.VFD)
            self.Q_drycooler = JoulesToKwH(self.Q_drycooler)

            self.GPM_drycooler = self.GPM * self.Valve_drycooler
            self.T_in_bath = (self.T_out_drycooler * self.Valve_drycooler
                              + self.T_out_bath * (1.0 - self.Valve_drycooler))

            dT = self.T_in_bath - self.targetSupply
            if abs(dT) > 1.0:
                # PID
                dVFD = dT * (0.005 * math.exp(-iterations/50.0))
                self.VFD = round(self.VFD + dVFD, 2)
                self.VFD = max(self.VFD, 0.10)    # min VFD speed is 10%
                self.VFD = min(self.VFD, 1.0)

            if self.VFD < 0.15:
                if abs(dT) > 1.0:
                    dValve = dT * (0.001 * math.exp(-iterations/100.0))
                    self.Valve_drycooler = round(self.Valve_drycooler + dValve, 2)
                    self.Valve_drycooler = min(self.Valve_drycooler, 1.0)
                    self.Valve_drycooler = max(self.Valve_drycooler, 0.1)
            else:
                self.Valve_drycooler = 1.0

            if ((abs(self.Q_bath - self.Q_drycooler) < 0.001 and
                 abs(self.T_out_bath - self.last_T_out_bath) < 0.001 and
                 abs(self.T_out_drycooler - self.last_T_out_drycooler) < 0.001)):
                converged = converged + 1
            else:
                converged = 0

            if converged > 10:
                break

            self.last_T_out_bath = self.T_out_bath
            self.last_T_out_drycooler = self.T_out_drycooler

        seconds = (self.clock % int(self.num_timesteps)) * 86400.0 / self.num_timesteps
        hours = (seconds // 3600) % 24
        mins = (seconds // 60) % 60
        secs = seconds % 60

        print("T_ambient %.2fC time %s (iter %d)" % (self.T_ambient, "%02d:%02d:%02d" % (hours, mins, secs), iterations))
        if self.T_vapor > 49.0:
            print("  --- BATH TOO HOT ---")
        print("  Bath      Ti %.3f  To %.3f  T_vapor %.3f  LMTD %.3f" %
              (self.T_in_bath, self.T_out_bath, self.T_vapor, lmtd))
        print("  DryCooler Ti %.3f  To %.3f  Q %.2f  Bypass %.2f  GPM %.2f  VFD %.2f" %
              (self.T_in_drycooler,
               self.T_out_drycooler,
               self.Q_drycooler,
               self.Valve_drycooler,
               self.GPM * self.Valve_drycooler,
               self.VFD))

        self.clock = self.clock + 1
        self.T_ambient = self.T_initial - self.T_range * math.sin(self.clock / self.num_timesteps * 2.0 * math.pi)

    def run(self):
        for _ in range(0, self.num_timesteps):
            model.timestep()


if __name__ == "__main__":
    model = EstcpModel()
    model.run()
