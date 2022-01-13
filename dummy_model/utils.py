import bioviz
import biorbd_casadi as biorbd
from casadi import MX, Function
import numpy as np


def display_model(path):
    b = bioviz.Viz(path)
    b.exec()
    return


def set_fiso_max(model: biorbd.Model, values: MX):
    for i in range(model.nbMuscles()):
        model.muscle(i).setForceIsoMax(values[i])


def set_opt_len(model: biorbd.Model, values: MX):
    for i in range(model.nbMuscles()):
        model.muscle(i).characteristics().setOptimalLength(values[i])


def set_ins_point(model: biorbd.Model, values: MX):
    for i in range(model.nbMuscles()):
        ins_vec = MX.zeros(3)
        ins_vec[2] = values[i]
        model.muscle(i).position().setInsertionInLocal(ins_vec)


def set_tendon_sl(model: biorbd.Model, values: MX):
    for i in range(model.nbMuscles()):
        model.muscle(i).characteristics().setTendonSlackLength(values[i])


def numeric_muscle_force(model):
    q = MX.sym('q', model.nbQ(), 1)
    dq = MX.sym('dq', model.nbQ(), 1)
    act = MX.sym('act', model.nbMuscles(), 1)
    states = model.stateSet()
    for k in range(model.nbMuscles()):
        states[k].setActivation(act[k])
    cas_fun = Function('cas_fun', [act, q, dq], [model.muscleForces(states, q, dq).to_mx()])
    return cas_fun


def muscle_forces(q, qdot, act, model):
    nb_frames = q.shape[1]
    mus_forces = np.zeros((model.nbMuscles(), nb_frames))
    cas_fun = numeric_muscle_force(model)
    for frame in range(nb_frames):
        mus_forces[:, frame] = np.array(cas_fun(act[:, frame], q[:, frame], qdot[:, frame])).squeeze()
    return mus_forces


def set_param_opt(name):
    if name == 'fiso':
        param_opt_vec = [1, 0, 0]
    if name == 'opt_len':
        param_opt_vec = [0, 1, 0]
    if name == 'tendon':
        param_opt_vec = [0, 0, 1]
    if name == 'none':
        param_opt_vec = [0, 0, 0]
    return param_opt_vec, name