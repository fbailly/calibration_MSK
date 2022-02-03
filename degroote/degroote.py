from scipy.integrate import solve_ivp
from utils import *
import seaborn as sns

def check_equilibrium(lm, act):
    vm = lm_ode_opensim(0, np.array([lm]), act)
    alpha = np.arcsin(lm_opt*np.sin(alpha0)/lm)
    lt = lmt - lm*np.cos(alpha)
    mf = (a*f_act(lm/lm_opt)*f_v(vm) + f_pas(lm/lm_opt))*np.cos(alpha)
    tf = f_tendon(lt/lt_sl)
    return mf, tf, vm


def search_equilibrium(act):
    vm_pos = 1e10
    lm_pos = np.array([0.5*lm0])
    while vm_pos > 1e-2:
        vm_pos = lm_ode_opensim(0, lm_pos, act)
        lm_pos += 0.001
        if np.isnan(vm_pos):
            vm_pos = 1e10
    lm_neg = np.array([1.5*lm0])
    vm_neg = -1e10
    while vm_neg < -1e-2:
        vm_neg = lm_ode_opensim(0, lm_neg, act)
        lm_neg -= 0.001
        if np.isnan(vm_neg):
            vm_pos = -1e10
    if np.isclose(lm_pos, lm_neg, rtol=1e-1, atol=1e-1):
        return (lm_pos+lm_neg)/2
    else:
        raise(AssertionError("equilibrium not reached"))


def plot_phase(act, color):
    eq_points = []
    for ac in act:
        eq_points += [search_equilibrium(ac)]
    lms = np.linspace(np.min(eq_points)-5e-4, np.max(eq_points)+5e-4, 200)
    for c, ac in enumerate(act):
        vms = check_equilibrium(lms, ac)[2]
        plt.quiver(lms[:-1], vms[:-1], np.sign(vms[:-1])*(lms[1:]-lms[:-1]), np.sign(vms[:-1])*(vms[1:]-vms[:-1]),
                scale_units='xy', angles='xy', scale=1, color=color[c], label=f"activation={ac:0.1f}")
        plt.plot(lms[np.argmin(np.abs(vms))], vms[np.argmin(np.abs(vms))], 'ro')
    plt.plot(lms, np.zeros(lms.shape), 'r')
    plt.legend()
    plt.xlabel("Muscle length")
    plt.ylabel("Muscle velocity")
    plt.ylim([-1, 1])
    plt.title("Phase space trajectory of Opensim-Thelen Hill muscle for varying activations")
    return


def f_tendon(lt_n):
    '''tendon force'''
    return C[0, 0]*np.exp(kt*(lt_n-C[0, 1]))-C[0, 2]


def f_act(lm_n):
    '''active muscle force'''
    if (type(lm_n) is int) or (type(lm_n) is float) or (type(lm_n) is np.float64):
        lm_n = np.array([lm_n])[:, np.newaxis]
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
    return np.array([inv_f_v((f_tendon(lt/lt_sl)/cos_alpha-f_pas(lm[0]/lm_opt))/(act*f_act(lm/lm_opt)))]).squeeze()


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


def dv_dA(lm, act):
    da = dA_dl(lm, act)
    lt = lmt - np.sqrt(lm**2-(lm_opt*np.sin(alpha0))**2)  # (S23 of Degroote)
    cos_alpha = (lmt - lt)/lm  # (S18 of Degroote)
    a = (f_tendon(lt/lt_sl)/cos_alpha-f_pas(lm/lm_opt))/(act*f_act(lm/lm_opt))
    return da/(D[0, 0]*D[0, 1])*np.cosh(1/D[0, 0]*(a - D[0, 3]))


def dA_dl(lm, act):
    gact = np.zeros((1, 3))
    dgact = np.zeros((1, 3))
    lt = lmt - np.sqrt(lm**2-(lm_opt*np.sin(alpha0))**2)  # (S23 of Degroote)
    cos_alpha = (lmt - lt)/lm  # (S18 of Degroote)
    for i in range(3):
        gact[:, i] = -0.5*(lm - B[1, i])**2 / 2*(B[2, i] + B[3, i] * lm0)
        dgact[:, i] = -(2*lm*B[2, i] + lm**2*B[3, i] - 2*B[1, i]*B[2, i] - B[3, i]*(B[1, i])**2)/(2*(B[2, i]+B[3, i]*lm)**2)
    b0dgact = np.multiply(B[0, :], dgact)
    dfact = np.sum(np.multiply(b0dgact, np.exp(gact)))
    dfpas = kpe/e0*np.exp(kpe*(lm-1)/e0)/(np.exp(kpe)-1)
    da_num = -dfpas*act*f_act(lm) - (f_tendon(lt/lt_sl)/cos_alpha-f_pas(lm/lm_opt))*act*dfact
    da_denum = (act*f_act(lm))**2
    return da_num/da_denum


def dv_dl(lm, act):
    dv_da = dv_dA(lm, act)
    da_dl = dA_dl(lm, act)
    return dv_da*da_dl


def check_dv_dl(lm, act):
    eps = 1e-8
    vm = lm_ode_opensim(0, np.array([lm]), act)
    dvm = lm_ode_opensim(0, np.array([lm+eps]), act)
    return (vm-dvm)/eps


lmt = 0.35  # musculo-tendon len
vmt = 0  # isostatic condition
lm_opt = 0.25  # optimal fiber len
lt_sl = 0.15  # tendon slack len
alpha0 = np.pi/4  # pennation angle at optimal fiber len
a = 0.5  # activation (input)
tf = 0.1  # simulation duration
lm0 = (lmt - lt_sl)/np.cos(alpha0)  # initial guess for the ivp problem (further noised)
ft0 = 0.5  # initial guess for the ivp problem (further noised)
ratio_interp = 1000
time_interp = np.linspace(0, tf, num=int(ratio_interp*tf))
# plot_muscle_char(f_tendon, f_act, f_pas, f_v)

print(dv_dl(lm0, a))
print(check_dv_dl(lm0, a))
print(f"Before solving, equilibirum is {check_equilibrium(lm0, a)}")
sol = solve_ivp(lm_ode_opensim, (0, tf), y0=np.array([lm0]), args=(a,), dense_output=True, method='RK23')
print(f"After solving, steady state is lm={sol.y[0, -1]}, equilibirum is {check_equilibrium(sol.y[0, -1], a)}")
plot_phase(np.linspace(0.1, 1, num=10), sns.color_palette())

plt.figure()
plt.subplot(121)
for a_ in np.arange(0.1, 1.0, 0.1):
    sol = solve_ivp(lm_ode_opensim, (0, tf), y0=np.array([lm0]), args=(a_,), dense_output=True, method='RK45')
    plt.plot(time_interp[:, np.newaxis], sol.sol(time_interp).T, label=f'a={a_:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Muscle length')
    plt.title('Normalized muscle length for varying activation levels\nOpensim-Thelen (2003)')
    plt.legend()


plt.subplot(122)
for a_ in np.arange(0.3, 1.0, 0.1):
    sol = solve_ivp(ft_ode_degroote, (0, tf), y0=np.array([ft0]), args=(a_,), dense_output=True, method='RK45')
    plt.plot(time_interp[:, np.newaxis], sol.sol(time_interp).T, label=f'a={a_:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Tendon force')
    plt.title('Normalized tendon force for varying activation levels\nDegroote (2016)')
    plt.legend()
plt.suptitle('Opensim-Thelen ODE for muscle tendon equilibrium')

plt.figure()
plt.subplot(121)
for lm0 in np.arange(0.3, 0.4, 0.01):
    sol = solve_ivp(lm_ode_opensim, (0, tf), y0=np.array([lm0]), args=(a,), dense_output=True, method='RK23')
    plt.plot(time_interp[:, np.newaxis], sol.sol(time_interp).T, label=f'lm0={lm0:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Muscle length')
    plt.title('Normalized muscle length for varying initial conditions of MT length\nOpensim-Thelen (2003)')
    plt.legend()

plt.subplot(122)
for ft0 in np.arange(0.2, 0.8, 0.05):
    sol = solve_ivp(ft_ode_degroote, (0, tf), y0=np.array([ft0]), args=(a,), dense_output=True, method='RK23')
    plt.plot(time_interp[:, np.newaxis], sol.sol(time_interp).T, label=f'ft0={ft0:.2f}')
    plt.xlabel('time (s)')
    plt.ylabel('Tendon force')
    plt.title('Normalized tendon force for varying intial conditions\nDegroote (2016)')
    plt.legend()
plt.suptitle('Opensim-Thelen ODE for muscle tendon equilibrium')

plt.show()
