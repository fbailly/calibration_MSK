from utils import *
import biorbd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from casadi import MX, Function, vertcat

def integrate_motion(ratio):
    t_full_s = np.linspace(0, duration, num=Ns * ratio + 1)
    states = np.zeros((2 * nq, ratio * Ns + 1))
    states[:, 0] = states_ref[:, 0]
    x = states_ref[:, 0]

    for n in range(Ns):
        t_inter = np.linspace(dt / ratio, dt, num=ratio)
        t_span = (0, dt)
        sol = solve_ivp(ode_scipy, t_span, x, dense_output=True, args=[cas_fun, n])
        states[:, 1 + n * ratio:1 + (n + 1) * ratio] = sol.sol(t_inter)
        x = sol['y'][:, -1]
    return states, t_full_s

def act_dynamics(t, x, n):
    q = x[:nq]
    dq = x[nq:2*nq]
    muscles_states = model.stateSet()
    act = muscles_ref[:, n]
    for k in range(nmus):
        muscles_states[k].setActivation(act[k])
    tau = model.muscularJointTorque(muscles_states, q, dq)
    ddq = biorbd.Model.ForwardDynamicsConstraintsDirect(model, q, dq, tau).to_MX()
    dq = model.computeQdot(q, dq).to_MX()
    return np.concatenate((dq, ddq))

def act_dynamics_cas():
    x = MX.sym('x', 2*nq)
    q = x[:nq]
    dq = x[nq:2*nq]
    muscles_states = model.stateSet()
    act = MX.sym('act', nmus)
    for k in range(nmus):
        muscles_states[k].setActivation(act[k])
    tau = model.muscularJointTorque(muscles_states, q, dq)
    ddq = biorbd.Model.ForwardDynamicsConstraintsDirect(model, q, dq, tau).to_mx()
    dq = model.computeQdot(q, dq).to_mx()
    cas_fun = Function('act_dyn_cas', [x, act], [vertcat(dq, ddq)])
    return cas_fun

def ode_scipy(t, x, cas_fun, n):
    dx = cas_fun(x, muscles_ref[:, n])
    return dx.toarray().squeeze()

# load model
model_path = "../dummy_model/dummy.bioMod"
model = biorbd.Model(model_path)
nq = model.nbQ()
nmus = model.nbMuscles()

fiso_max = [825, 920, 1200, 1125]
opt_len = [0.69, 0.72, 0.68, 0.73]
tendon_sl = [0.1, 0.1, 0.1, 0.1]

# reference motion
Ns = 32
duration = 1
dt = duration/Ns
ocp = generate_data(model_path, fiso_max, opt_len, tendon_sl, duration, Ns)
opts = {"linear_solver": "ma57", "tol": 1e-4, "print_level": 5}
sol = ocp.solve(solver_options=opts)
muscles_ref = sol.controls['muscles']
states_ref = sol.states['all']

# change muscle properties
set_fiso_max(model, fiso_max)
set_opt_len(model, opt_len)
set_tendon_sl(model, tendon_sl)

# integrate
cas_fun = act_dynamics_cas()
ratio = 100

states, t_full = integrate_motion(ratio)

# plot
plt.plot(t_full, states.T, label=['q0', 'q1', 'dq0', 'dq1'])
plt.gca().set_prop_cycle(None)
plt.plot(t_full[::ratio, ], states[:, ::ratio].T, 'x')

# change muscle properties
fiso_max_n = [825, 920, 1200, 1125]
opt_len_n = [0.69, 0.72, 0.68, 0.73]
tendon_sl_n = [0.2, 0.2, 0.2, 0.2]
set_fiso_max(model, fiso_max_n)
set_opt_len(model, opt_len_n)
set_tendon_sl(model, tendon_sl_n)

# integrate
ratio = 100
cas_fun = act_dynamics_cas()
states, t_full = integrate_motion(ratio)

# plot
plt.gca().set_prop_cycle(None)
plt.plot(t_full, states.T)
plt.gca().set_prop_cycle(None)
plt.plot(t_full[::ratio, ], states[:, ::ratio].T, 'o')

plt.legend()
plt.title(f"x -> tendon sl = {tendon_sl[0]}, o -> tendon sl = {tendon_sl_n[0]}")
plt.show()
