import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from utils import *


def f_tendon(lt_n):
    '''tendon force'''
    return C[0, 0]*np.exp(kt*(lt_n-C[0, 1]))-C[0, 2]


def f_act(lm_n):
    '''active muscle force'''
    if (type(lm_n) is int) or (type(lm_n) is float):
        lm_n = np.ndarray(lm_n)[:, np.newaxis]
    if (type(lm_n) is np.ndarray) and (len(lm_n.shape) == 1):
        lm_n = lm_n[:, np.newaxis]
    siz = lm_n.shape[0]
    f = np.zeros((siz, 1))
    for i in range(siz):
        f[i, 0] = np.sum(np.multiply(B[0, :], np.exp(
            np.divide(-0.5*np.square(lm_n[i, 0]-B[1, :]), (B[2, :] + B[3, :]*lm_n[i, 0])))))
    return f


def f_pas(lm_n):
    '''passive muscle force'''
    return (np.exp(kpe*(lm_n-1)/e0)-1)/(np.exp(kpe)-1)


def f_v(vm_n):
    '''force-velocity curve'''
    return D[0, 0]*np.log(D[0, 1]*vm_n+D[0, 2]+np.sqrt((D[0, 1]*vm_n+D[0, 2])**2+1))+D[0, 3]


def inv_f_v(f):
    '''inverse of force-velocity curve, corrected from the original paper'''
    return 1/D[0, 1]*(np.sinh(1/D[0, 0]*(f - D[0, 3]))) - D[0, 2]/D[0, 1]


def f_m(a, lm_n, vm_n):
    '''total muscle force'''
    return f0*(a*f_act(lm_n)*f_v(vm_n)+f_pas(lm_n))


def ode_opensim(t, lm, lt, alpha, a):
    '''Thelen 2003 implementation'''
    return inv_f_v((f_tendon(lt)/np.cos(alpha)-f_pas(lm))/a*f_act(lm))

lt = 1.01
alpha = np.pi/8
a = 0.2
tf = 1
lm0 = 1.2

#plot_muscle_char(f_tendon, f_act, f_pas, f_v)
sol = solve_ivp(ode_opensim, (0, tf), y0=np.array([lm0]), args=(lt, alpha, a))
sol_interp = interp1d(sol.t, sol.y)
time_interp = np.linspace(0, tf, num=1000)
plt.plot(time_interp[:, np.newaxis], sol_interp(time_interp).T)
plt.show()