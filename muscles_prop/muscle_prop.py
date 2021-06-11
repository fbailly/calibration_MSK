from utils import *
import biorbd
import numpy.random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def act_dynamics(t, x):
    q = x[:nq]
    dq = x[nq:2*nq]
    muscles_states = model.stateSet()
    act = control(t)
    for k in range(nmus):
        muscles_states[k].setActivation(act[k])
    tau = model.muscularJointTorque(muscles_states, q, dq)
    ddq = biorbd.Model.ForwardDynamicsConstraintsDirect(model, q, dq, tau).to_array()
    dq = model.computeQdot(q, dq).to_array()
    return np.concatenate((dq, ddq))

# load model
model_path = "../dummy_model/dummy.bioMod"
model = biorbd.Model(model_path)
nq = model.nbQ()
nmus = model.nbMuscles()

# change properties
fiso_max = [825, 920, 1200, 1125]
opt_len = [0.69, 0.72, 0.68, 0.73]
tendon_sl = [0.2, 0.2, 0.2, 0.2]


# reference motion
Ns = 50
Duration = 1
ocp = generate_data(model_path, fiso_max, opt_len, tendon_sl, Duration, Ns)
opts = {"linear_solver": "ma57", "tol": 1e-4, "print_level": 5}
sol = ocp.solve(solver_options=opts)
muscles_ref = sol.controls['muscles']
states_ref = sol.states['all']
forces_ref = muscle_forces(sol.states['q'], sol.states['qdot'], sol.controls['muscles'], biorbd.Model(model_path))