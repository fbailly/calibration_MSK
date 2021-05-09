import bioviz
import biorbd
from casadi import MX,Function
import numpy as np
import matplotlib.pyplot as plt

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
    ParameterList,
    Bounds,
    InitialGuess,
    InterpolationType,
    Solver,
)

def display_model(path):
    b = bioviz.Viz(path)
    b.exec()
    return

def set_fiso_max(biorbd_model: biorbd.Model, values: MX):
    for i in range(biorbd_model.nbMuscles()):
        biorbd_model.muscle(i).setForceIsoMax(values[i])

def set_opt_len(biorbd_model: biorbd.Model, values: MX):
    for i in range(biorbd_model.nbMuscles()):
        biorbd_model.muscle(i).characteristics().setOptimalLength(values[i])

def set_ins_point(biorbd_model: biorbd.Model, values: MX):
    for i in range(biorbd_model.nbMuscles()):
        ins_vec = MX.zeros(3)
        ins_vec[2] = values[i]
        biorbd_model.muscle(i).position().setInsertionInLocal(ins_vec)

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
    muscle_forces = np.zeros((model.nbMuscles(), nb_frames))
    muscles_states = model.stateSet()
    cas_fun = numeric_muscle_force(model)
    for frame in range(nb_frames):
        muscle_forces[:, frame] = np.array(cas_fun(act[:, frame], q[:, frame], qdot[:, frame])).squeeze()
    return muscle_forces

def generate_data(
    biorbd_model_path: str,
    fiso_max: list,
    opt_len: list,
    ins_point: list,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:

    # load model
    biorbd_model = biorbd.Model(biorbd_model_path)

    # change properties
    set_fiso_max(biorbd_model, fiso_max)
    set_opt_len(biorbd_model, opt_len)
    set_ins_point(biorbd_model, ins_point)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker_idx=2, second_marker_idx=6, weight=0
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[2, 3], node=Node.START, weight=1e4)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, node=Node.MID, target=np.array([-0.7, -0.9, 0, 0], ndmin=2).T, weight=1e4)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, node=Node.END, target=np.array([0.7, 0.9, 0, 0], ndmin=2).T, weight=1e4)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:2*biorbd_model.nbQ(), 0] = (0.7, 0.9, 0, 0)
    x_bounds[0].max[:2*biorbd_model.nbQ(), 0] = (0.7, 0.9, 0, 0)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    u_bounds = BoundsList()
    u_bounds.add(
        [muscle_min] * biorbd_model.nbMuscleTotal(),
        [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    u_init = InitialGuessList()
    u_init.add([muscle_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
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

def prepare_ocp(
    biorbd_model_path: str,
    opt_fiso: bool,
    opt_opt_len: bool,
    opt_ins_point: bool,
    fiso_max: list,
    fiso_bounds: list,
    fiso_init: list,
    opt_len: list,
    opt_len_bounds: list,
    opt_len_init: list,
    ins_point: list,
    ins_point_bounds: list,
    ins_point_init: list,
    final_time: float,
    n_shooting: int,
    muscles_ref: np.ndarray,
    states_ref: np.ndarray,
    use_acados: bool,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:

    # load model
    biorbd_model = biorbd.Model(biorbd_model_path)

    # change properties
    set_fiso_max(biorbd_model, fiso_max)
    set_opt_len(biorbd_model, opt_len)
    set_ins_point(biorbd_model, ins_point)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL, target=muscles_ref, weight=50)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, target=states_ref, weight=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:2*biorbd_model.nbQ(), 0] = (0.5, 0.7, 0, 0)
    x_bounds[0].max[:2*biorbd_model.nbQ(), 0] = (0.9, 1.1, 0, 0)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    u_bounds = BoundsList()
    u_bounds.add(
        [muscle_min] * biorbd_model.nbMuscleTotal(),
        [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    u_init = InitialGuessList()
    u_init.add([muscle_init] * biorbd_model.nbMuscleTotal())

    # Parameters
    parameters = ParameterList()
    bounds_fiso = Bounds([fiso_bounds[0]]*biorbd_model.nbMuscles(),
                         [fiso_bounds[1]]*biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_fiso = InitialGuess(fiso_init)
    bounds_opt_len = Bounds([opt_len_bounds[0]]*biorbd_model.nbMuscles(),
                            [opt_len_bounds[1]]*biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_opt_len = InitialGuess(opt_len_init)
    bounds_ins_point = Bounds([ins_point_bounds[0]]*biorbd_model.nbMuscles(),
                              [ins_point_bounds[1]]*biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_ins_point = InitialGuess(ins_point_init)
    if opt_fiso:
        parameters.add(
            "forces_iso_max",
            set_fiso_max,
            initial_fiso,
            bounds_fiso,
            size=biorbd_model.nbMuscles(),
            scaling=np.array([1]),
            )
    if opt_opt_len:
        parameters.add(
        "optimal_length",
        set_opt_len,
        initial_opt_len,
        bounds_opt_len,
        size=biorbd_model.nbMuscles(),
        )
    if opt_ins_point:
        parameters.add(
        "insertion_point",
        set_ins_point,
        initial_ins_point,
        bounds_ins_point,
        size=biorbd_model.nbMuscles(),
        )
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        parameters=parameters,
        ode_solver=ode_solver,
        n_threads=8,
        use_sx=use_acados,
    )

def calibration_iter(model_path,
                     opt_fiso, opt_opt_len, opt_ins_point,
                     fiso_max, init_fiso_max, fiso_bounds,
                     opt_len, init_opt_len, opt_len_bounds,
                     ins_point, init_ins_point, ins_point_bounds,
                     Duration, Ns, muscles_ref, states_ref, use_acados=False):
    ocp = prepare_ocp(model_path,
                      opt_fiso, opt_opt_len, opt_ins_point,
                      fiso_max, fiso_bounds, init_fiso_max,
                      opt_len, opt_len_bounds, init_opt_len,
                      ins_point, ins_point_bounds, init_ins_point,
                      Duration, Ns, muscles_ref, states_ref, use_acados=use_acados)
    if use_acados:
        solver = Solver.ACADOS
        opts={}
    else:
        solver = Solver.IPOPT
        opts={"linear_solver": "ma57", "max_iter": 50, "print_level": 0}
    sol = ocp.solve(solver=solver, solver_options=opts)
    return sol

if __name__ == '__main__':

    # main settings
    Ns = 24
    Duration = 1
    use_acados = False
    n_iter = 10
    Ns_calib = 24
    Duration_calib = Duration*Ns_calib/Ns
    calib_gain = [1.3, 1.3, 1.8] # optimal gains
    # calib_gain = [1, 1, 1]

    # generate data with custom model
    model_path = "./dummy.bioMod"
    orig_fiso_max = [547, 620, 1000, 895]
    orig_opt_len = [0.69, 0.72, 0.68, 0.73]
    orig_ins_point = [0.5, 0.5, 0.5, 0.5]

    ocp = generate_data(model_path, orig_fiso_max, orig_opt_len, orig_ins_point, Duration, Ns)
    opts = {"linear_solver": "ma57", "tol": 1e-4, "print_level": 0}
    sol = ocp.solve(solver_options=opts)
    muscles_ref = sol.controls['muscles'][:, :Ns_calib]
    states_ref = sol.states['all'][:, :Ns_calib+1]
    forces_ref = muscle_forces(sol.states['q'], sol.states['qdot'], sol.controls['muscles'], biorbd.Model(model_path))
    # start with wrong model
    # fiso_max = [547, 620, 1000, 895]
    fiso_max = [700, 700, 700, 700]
    # opt_len = [0.7, 0.7, 0.7, 0.7]
    opt_len = [0.73, 0.66, 0.73, 0.68]
    ins_point = [0.5, 0.5, 0.5, 0.5]
    # ins_point = [0.46, 0.52, 0.47, 0.53]

    # recursively match data with optimized model
    fiso_vec = np.zeros((4, n_iter))
    opt_len_vec = np.zeros((4, n_iter))
    ins_point_vec = np.zeros((4, n_iter))
    success_vec = np.zeros((1, n_iter))

    init_fiso_max = fiso_max.copy()
    fiso_bounds = [300, 1200]
    init_opt_len = opt_len.copy()
    opt_len_bounds = [0.6, 0.8]
    init_ins_point = ins_point.copy()
    ins_point_bounds = [0.4, 0.7]
    opt_fiso = False
    opt_opt_len = True
    opt_ins_point = False

    for i in range(n_iter):
        sol = calibration_iter(model_path, opt_fiso, opt_opt_len, opt_ins_point,
                               fiso_max, init_fiso_max, fiso_bounds,
                               opt_len, init_opt_len, opt_len_bounds,
                               ins_point, init_ins_point, ins_point_bounds,
                               Duration_calib, Ns_calib, muscles_ref, states_ref, use_acados)
        success_vec[0, i] = sol.status
        if opt_fiso:
            fiso_vec[:, i] = fiso_max
            fiso_max = list(fiso_max + calib_gain[0]*(sol.parameters['forces_iso_max'].squeeze() - fiso_max))
            # init_fiso_max = fiso_max
            opt_len_vec[:, i] = opt_len
            ins_point_vec[:, i] = ins_point
            print(f"{i}th iter")
            print(f"{sol.parameters['forces_iso_max']} N")
            print(f"Original model forces isomax : {orig_fiso_max} N")
            opt_fiso = False
            opt_ins_point = False
            opt_opt_len = True
        elif opt_opt_len:
            opt_len_vec[:, i] = opt_len
            opt_len = list(opt_len + calib_gain[1]*(sol.parameters['optimal_length'].squeeze() - opt_len))
            fiso_vec[:, i] = fiso_max
            ins_point_vec[:, i] = ins_point
            # init_opt_len = opt_len
            print(f"{i}th iter")
            print(f"{sol.parameters['optimal_length']} m")
            print(f"Original model optimal length : {orig_opt_len} m")
            opt_fiso = True
            opt_ins_point = False
            opt_opt_len = False
        elif opt_ins_point:
            ins_point_vec[:, i] = ins_point
            ins_point = list(ins_point + calib_gain[2]*(sol.parameters['insertion_point'].squeeze() - ins_point))
            fiso_vec[:, i] = fiso_max
            opt_len_vec[:, i] = opt_len
            # init_opt_len = opt_len
            print(f"{i}th iter")
            print(f"{sol.parameters['insertion_point']} m")
            print(f"Original model insertion point : {orig_ins_point} m")
            opt_fiso = False
            opt_ins_point = False
            opt_opt_len = True

        if np.allclose(opt_len, orig_opt_len, 1e-1, 1e-1) \
                and np.allclose(fiso_max, orig_fiso_max, 1e-1, 1e-1) \
                and np.allclose(ins_point, orig_ins_point, 1e-2, 1e-2):
            break

    forces_est = muscle_forces(sol.states['q'], sol.states['qdot'], sol.controls['muscles'], biorbd.Model(model_path))

    force_err = np.sqrt(np.sum((forces_ref-forces_est)**2, 1)/(Ns+1))
    force_err_percent = force_err/np.mean(forces_est, 1)*100
    fiso_err = np.sqrt(np.sum((np.array(fiso_max) - np.array(orig_fiso_max))**2)/4)
    fiso_err_percent = np.sqrt((np.array(fiso_max) - np.array(orig_fiso_max))**2)/np.array(orig_fiso_max)*100
    opt_len_err = np.sqrt(np.sum((np.array(opt_len) - np.array(orig_opt_len))**2)/4)
    opt_len_err_percent = np.sqrt((np.array(opt_len) - np.array(orig_opt_len))**2)/np.array(orig_opt_len)*100
    print(f"mean RMSE on muscle force : {np.mean(force_err)}")
    print(f"Percentage error on muscle force: {np.mean(force_err_percent)}")
    print(f"RMSE on force iso max : {fiso_err}")
    print(f"Percentage error on force iso max : {np.mean(fiso_err_percent)}")
    print(f"RMSE on opt len : {opt_len_err}")
    print(f"Percentage error on opt len : {np.mean(opt_len_err_percent)}")

    # sol.animate()
    sol.graphs()
    iters = np.linspace(0, i, num=i+1)
    plt.subplot(411)
    plt.plot(iters, fiso_vec[:, :i+1].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_fiso_max, ndmin=2).T, i+1, 1).T, 'o--', label='True values')
    plt.legend()
    plt.title('Forces isomax calibration')
    plt.xlabel('iterations')
    plt.ylabel('F_isomax')
    plt.subplot(412)
    plt.plot(iters, opt_len_vec[:, :i+1].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_opt_len, ndmin=2).T, i+1, 1).T, 'o--', label='True values')
    plt.title('Optimal length calibration')
    plt.xlabel('iterations')
    plt.ylabel('Opt_length')
    plt.subplot(413)
    plt.plot(iters, ins_point_vec[:, :i+1].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_ins_point, ndmin=2).T, i+1, 1).T, 'o--', label='True values')
    plt.title('Insertion point calibration')
    plt.xlabel('iterations')
    plt.ylabel('Ins_point')
    plt.subplot(414)
    plt.plot(iters, success_vec[:, :i+1].T, label='Convergence')
    plt.title('Convergence iter (0 = success / 1 = max_iter)')
    plt.xlabel('iterations')
    plt.ylabel('Convergence')
    plt.show()