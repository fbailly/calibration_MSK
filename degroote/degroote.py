import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from utils import *


def f_tendon(lt_n):
    '''tendon force'''
    return C[0, 0]*np.exp(kt*(lt_n-C[0, 1]))-C[0, 2]


def f_act(lm_n):
    '''active muscle force'''
    if (type(lm_n) is int) or (type(lm_n) is float) :
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


def lm_ode_opensim(t, lm, act):
    '''Thelen 2003 implementation'''
    lt = lmt - np.sqrt(lm**2-(lm_opt*np.sin(alpha0))**2)  # (S23 of Degroote)
    cos_alpha = (lmt - lt)/lm  # (S18 of Degroote)
    return np.array([inv_f_v((f_tendon(lt/lt_sl)/cos_alpha-f_pas(lm[0]))/act*f_act(lm))]).squeeze()


def ft_ode_degroote(t, ft, act):
    lt_n = 1/kt*np.log(1/C[0, 0]*(ft+C[0, 2]))+C[0, 1]
    lm = np.sqrt((lm_opt*np.sin(alpha0))**2 + (lmt - lt_n*lt_sl)**2)
    cos_alpha = (lmt-lt_n*lt_sl)/lm
    fm = ft/cos_alpha
    fvm = (fm-f_pas(lm/lm_opt))/(act*f_act(np.array([lm/lm_opt])))
    vm = inv_f_v(fvm)
    vt_n = vmt - vm/cos_alpha
    dft = C[0, 0]*kt*np.exp(kt*(lt_n-C[0, 1]))*vt_n
    return dft[0]


lmt = 0.35  # musculo-tendon len
vmt =  0  # isostatic condition
lm_opt = 0.25  # optimal fiber len
lt_sl = 0.1  # tendon slack len
alpha0 = np.pi/4  # pennation angle at optimal fiber len
a = 0.1  # activation (input)
tf = 0.25  # simulation duration
lm0 = (lmt - lt_sl)/np.cos(alpha0)  # initial guess for the ivp problem (further noised)
ft0 = 0.5  # initial guess for the ivp problem (further noised)


# plot_muscle_char(f_tendon, f_act, f_pas, f_v)


plt.subplot(121)
for a_ in np.arange(0.1, 1.0, 0.1):
    sol = solve_ivp(lm_ode_opensim, (0, tf), y0=np.array([lm0+np.random.rand()*0.2]), args=(a_,), method='RK45')
    sol_interp = interp1d(sol.t, sol.y, kind='cubic')
    time_interp = np.linspace(0, tf, num=int(10000*tf))
    plt.plot(time_interp[:, np.newaxis], sol_interp(time_interp).T, label=f'a={a_:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Muscle length')
    plt.title('Normalized muscle length for varying activation levels\nOpensim-Thelen (2003)')
    plt.legend()

plt.subplot(122)
for a_ in np.arange(0.1, 1.0, 0.1):
    sol = solve_ivp(ft_ode_degroote, (0, tf), y0=np.array([ft0+np.random.rand()*0.5]), args=(a_,), method='RK45')
    sol_interp = interp1d(sol.t, sol.y, kind='cubic')
    time_interp = np.linspace(0, tf, num=int(10000*tf))
    plt.plot(time_interp[:, np.newaxis], sol_interp(time_interp).T, label=f'a={a_:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Tendon force')
    plt.title('Normalized tendon force for varying activation levels\nDegroote (2016)')
    plt.legend()
plt.suptitle('Opensim-Thelen ODE for muscle tendon equilibrium')

plt.figure()
plt.subplot(121)
for lm0 in np.arange(0.35, 0.6, 0.01):
    sol = solve_ivp(lm_ode_opensim, (0, tf), y0=np.array([lm0]), args=(a,), method='RK45')
    sol_interp = interp1d(sol.t, sol.y, kind='cubic')
    time_interp = np.linspace(0, tf, num=int(10000*tf))
    plt.plot(time_interp[:, np.newaxis], sol_interp(time_interp).T, label=f'lm0={lm0:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Muscle length')
    plt.title('Normalized muscle length for varying initial conditions of MT length\nOpensim-Thelen (2003)')
    plt.legend()

plt.subplot(122)
for ft0 in np.arange(0.2, 0.8, 0.02):
    sol = solve_ivp(ft_ode_degroote, (0, tf), y0=np.array([ft0]), args=(a,), method='RK45')
    sol_interp = interp1d(sol.t, sol.y, kind='cubic')
    time_interp = np.linspace(0, tf, num=int(10000*tf))
    plt.plot(time_interp[:, np.newaxis], sol_interp(time_interp).T, label=f'ft0={ft0:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Tendon force')
    plt.title('Normalized tendon force for varying intial conditions\nDegroote (2016)')
    plt.legend()
plt.suptitle('Opensim-Thelen ODE for muscle tendon equilibrium')

plt.show()
