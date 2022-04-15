import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

G  = 6.67430e-8 # cm**3 / g / s**2
kB = 1.380649e-16 # erg/K
c  = 2.99792458e10 # cm/s

MeV_erg = 1.6021766339999998e-06
fm3_cm3 = 1e39
cm_km = 1e-5
g_Msun = 1 / (1.99e33)

fname = os.path.join("EoS_data", "68_APR", "eos.table")

x = np.loadtxt(fname).T
T, nb, Yq, P, ε = x[0:5]
idx = np.argsort(P)

T  = T[idx] / kB * MeV_erg
nb = nb[idx] * fm3_cm3
Yq = Yq[idx]
P  = P[idx] * MeV_erg * fm3_cm3
ε  = ε[idx] * MeV_erg * fm3_cm3

log_P_max = np.log10(P[-1])
ΔP = 0.6861537984007633*log_P_max - 22.41009625094139
P_c = np.logspace(log_P_max-ΔP, log_P_max, num=100)

class System:
    """ Class defining a system of first-order ODEs. """
    
    def __init__(self, *rhsides, init_conds=None):
        """
        Takes a sequence of functions representing a system of differential equations, and optionally some initial
        conditions (which should include the independent variable as the first element).
        """
        
        self.rhsides = np.array(rhsides)
        self.ivar = None
        self.dvars = None
        
        if init_conds:
            self.set_conds(*init_conds)
        else:
            self.set_conds()
            
    def set_conds(self, *conds):
        """
        Sets the current state of the system. The first argument should be the value of the independent variable,
        and the remaining parameters should be the corresponding values of the dependent variables in the same
        order that their derivatives were supplied to the constructor.
        """
        
        if conds:
            self.ivar = conds[0]
            self.dvars = np.array(conds[1:], dtype=np.float64)
        else:
            self.ivar = 0.0
            self.dvars = np.zeros_like(rhsides, dtype=np.float64)
            

@np.vectorize
def call(func, *args):
    """ Calls all functions in an array with the given arguments, returning the results. """
    
    return func(*args)
    
def rk4(system, step, completed):
    """ Abstract implementation of classical 4th-order Runge-Kutta. """
    
    comp_level = 0
    
    while comp_level <= 0:
        
        with warnings.catch_warnings():
            
            # Just because the RuntimeWarnings from the nans bug me
            warnings.simplefilter(action='ignore', category=RuntimeWarning)

            # Compute k-values
            k1 = step * call(system.rhsides, system.ivar, *system.dvars)
            k2 = step * call(system.rhsides, system.ivar + step / 2, *(system.dvars + k1 / 2))
            k3 = step * call(system.rhsides, system.ivar + step / 2, *(system.dvars + k2 / 2))
            k4 = step * call(system.rhsides, system.ivar + step, *(system.dvars + k3))
        
        if np.isnan([k1, k2, k3, k4]).any():
            # y went negative, meaning that the step was too large
            comp_level = -1
        else:
            # Try an update
            newvars = system.dvars + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            comp_level = completed(system.ivar + step, newvars)
        
        # Halve the step if necessary, otherwise update
        
        if comp_level < 0:
            step /= 2
        else:
            system.ivar += step
            system.dvars = newvars
            yield np.append([system.ivar], system.dvars)
            
def Pprime(r_i, P_i, M_i):
    """ Derivative of P with respect to r. """
    
    ε_i = np.interp(P_i, P, ε)
    return -G * (M_i + 4*np.pi*r_i**3 * P_i / c**2) * (ε_i + P_i) / (r_i * (c**2*r_i - 2*G*M_i))
    
def Mprime(r_i, P_i, M_i):
    """ Derivative of M with respect to r. """
    
    ε_i = np.interp(P_i, P, ε)
    return 4*np.pi * r_i**2 * ε_i / c**2
    
def completed(ivar, dvars, epsilon=P[0]):
    """ Returns 1 if the desired endpoint has been reached, -1 if the step should be halved, and 0 otherwise. """
    
    if dvars[0] < 0.0:
        return -1
    if dvars[0] < epsilon:
        return 1
    return 0

def integrate_TOV(step, P0):
    """Use 4th-order Runge-Kutta to solve the TOV equation for a particular EoS and central pressure."""
    
    r0 = step
    M0 = 4 * np.pi * step**3 * np.interp(P0, P, ε) / c**2
    
    system = System(Pprime, Mprime, init_conds=(r0, P0, M0))
    res = np.array(list(rk4(system, step, completed)))
    return res

def _progress_bar(frac, size=50):
    
    n = round(size*frac)
    bar = '[' + '⊙'*n + ' '*(size-n) + ']'
    if frac < 1.0:
        end = '\r'
    else:
        end = '\n'
    print(bar, f'{round(100*frac)}%', end=end)
    
dr = 1e2
results = [None]*len(P_c)
_progress_bar(0)
for i, P_ci in enumerate(P_c):
    results[i] = integrate_TOV(dr, P_ci)
    _progress_bar((i+1)/len(P_c))
    
R = np.array([res[-1][0] for res in results])
M = np.array([res[-1][2] for res in results])
plt.plot(R*cm_km, M*g_Msun)
plt.xlabel(r"$R~[\mathrm{km}]$")
plt.ylabel(r"$M~[\mathrm{M_\odot}]$")
plt.show()
