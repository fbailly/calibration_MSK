import bioviz
import biorbd
import numpy as np
from bioptim import (OdeSolver, OptimalControlProgram, ObjectiveFcn, ObjectiveList,
                     Node, DynamicsList, DynamicsFcn, BoundsList, InitialGuessList, QAndQDotBounds)

def display_model(path):
    b = bioviz.Viz(path)
    b.exec()
    return


def set_fiso_max(model: biorbd.Model, values):
    for i in range(model.nbMuscles()):
        model.muscle(i).setForceIsoMax(values[i])


def set_opt_len(model: biorbd.Model, values):
    for i in range(model.nbMuscles()):
        model.muscle(i).characteristics().setOptimalLength(values[i])


def set_tendon_sl(model: biorbd.Model, values):
    for i in range(model.nbMuscles()):
        model.muscle(i).characteristics().setTendonSlackLength(values[i])


def muscle_forces(q, qdot, act, model):
    nb_frames = q.shape[1]
    mus_forces = np.zeros((model.nbMuscles(), nb_frames))
    states = model.stateSet()
    for frame in range(nb_frames):
        for k in range(model.nbMuscles()):
            states[k].setActivation(act[:, frame][k])
        mus_forces[:, frame] = model.muscleForces(states, q[:, frame], qdot[:, frame])
    return mus_forces

def generate_data(
    model_path: str,
    fiso_max: list,
    opt_len: list,
    tendon_sl: list,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:

    # load model
    model = biorbd.Model(model_path)

    # change properties
    set_fiso_max(model, fiso_max)
    set_opt_len(model, opt_len)
    set_tendon_sl(model, tendon_sl)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker='1basetarget', second_marker='2seg2', weight=0
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[2, 3], node=Node.START, weight=1e4)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, node=Node.MID, target=np.array([-0.7, -0.9, 0, 0], ndmin=2).T, weight=1e4)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, node=Node.END, target=np.array([0.7, 0.9, 0, 0], ndmin=2).T, weight=1e4)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(model))
    x_bounds[0].min[:2*model.nbQ(), 0] = (0.7, 0.9, 0, 0)
    x_bounds[0].max[:2*model.nbQ(), 0] = (0.7, 0.9, 0, 0)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * model.nbQ() + [0] * model.nbQdot())

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    u_bounds = BoundsList()
    u_bounds.add(
        [muscle_min] * model.nbMuscleTotal(),
        [muscle_max] * model.nbMuscleTotal(),
    )

    u_init = InitialGuessList()
    u_init.add([muscle_init] * model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        n_threads=8
    )