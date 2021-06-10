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
    muscle_forces = np.zeros((model.nbMuscles(), nb_frames))
    muscles_states = model.stateSet()
    cas_fun = numeric_muscle_force(model)
    for frame in range(nb_frames):
        muscle_forces[:, frame] = np.array(cas_fun(act[:, frame], q[:, frame], qdot[:, frame])).squeeze()
    return muscle_forces


def generate_data(
    model_path: str,
    fiso_max: list,
    opt_len: list,
    ins_point: list,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:

    # load model
    model = biorbd.Model(model_path)

    # change properties
    set_fiso_max(model, fiso_max)
    set_opt_len(model, opt_len)
    set_ins_point(model, ins_point)

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

def prepare_ocp(
    model_path: str,
    opt_fiso: bool, opt_opt_len: bool, opt_ins_point: bool, opt_tendon_sl: bool,
    fiso_max: list, fiso_bounds: list, fiso_init: list,
    opt_len: list, opt_len_bounds: list, opt_len_init: list,
    ins_point: list, ins_point_bounds: list, ins_point_init: list,
    tendon_sl: list, tendon_sl_bounds: list, tendon_sl_init: list,
    final_time: float,
    n_shooting: int,
    muscles_ref: np.ndarray,
    states_ref: np.ndarray,
    use_acados: bool,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:

    # load model
    model = biorbd.Model(model_path)

    # change properties
    set_fiso_max(model, fiso_max)
    set_opt_len(model, opt_len)
    set_ins_point(model, ins_point)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL, target=muscles_ref, weight=50)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, target=states_ref, weight=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(model))
    x_bounds[0].min[:2*model.nbQ(), 0] = (0.5, 0.7, 0, 0)
    x_bounds[0].max[:2*model.nbQ(), 0] = (0.9, 1.1, 0, 0)

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

    # Parameters
    parameters = ParameterList()
    bounds_fiso = Bounds([fiso_bounds[0]]*model.nbMuscles(),
                         [fiso_bounds[1]]*model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_fiso = InitialGuess(fiso_init)
    bounds_opt_len = Bounds([opt_len_bounds[0]]*model.nbMuscles(),
                            [opt_len_bounds[1]]*model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_opt_len = InitialGuess(opt_len_init)
    bounds_ins_point = Bounds([ins_point_bounds[0]]*model.nbMuscles(),
                              [ins_point_bounds[1]]*model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_ins_point = InitialGuess(ins_point_init)
    bounds_tendon_sl = Bounds([tendon_sl_bounds[0]]*model.nbMuscles(),
                              [tendon_sl_bounds[1]]*model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_tendon_sl = InitialGuess(tendon_sl_init)
    if opt_fiso:
        parameters.add(
            "forces_iso_max",
            set_fiso_max,
            initial_fiso,
            bounds_fiso,
            size=model.nbMuscles(),
            scaling=np.array([1]),
            )
    if opt_opt_len:
        parameters.add(
        "optimal_length",
        set_opt_len,
        initial_opt_len,
        bounds_opt_len,
        size=model.nbMuscles(),
        )
    if opt_ins_point:
        parameters.add(
        "insertion_point",
        set_ins_point,
        initial_ins_point,
        bounds_ins_point,
        size=model.nbMuscles(),
        )
    if opt_tendon_sl:
        parameters.add(
        "tendon_slack_len",
        set_tendon_sl,
        initial_tendon_sl,
        bounds_tendon_sl,
        size=model.nbMuscles(),
        )
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
        parameters=parameters,
        ode_solver=ode_solver,
        n_threads=8,
        use_sx=use_acados,
    )

def calibration_iter(model_path,
                     opt_fiso, opt_opt_len, opt_ins_point, opt_tendon_sl,
                     fiso_max, init_fiso_max, fiso_bounds,
                     opt_len, init_opt_len, opt_len_bounds,
                     tendon_sl, init_tendon_sl, tendon_sl_bounds,
                     ins_point, init_ins_point, ins_point_bounds,
                     Duration, Ns, muscles_ref, states_ref, use_acados=False):
    ocp = prepare_ocp(model_path,
                      opt_fiso, opt_opt_len, opt_ins_point, opt_tendon_sl,
                      fiso_max, fiso_bounds, init_fiso_max,
                      opt_len, opt_len_bounds, init_opt_len,
                      tendon_sl, tendon_sl_bounds, init_tendon_sl,
                      ins_point, ins_point_bounds, init_ins_point,
                      Duration, Ns, muscles_ref, states_ref, use_acados=use_acados)
    if use_acados:
        solver = Solver.ACADOS
        opts={"print_level": 0}
    else:
        solver = Solver.IPOPT
        opts={"linear_solver": "ma57", "max_iter": 50, "print_level": 0, "hessian_approximation": "exact"}
    sol = ocp.solve(solver=solver, solver_options=opts)
    return sol

if __name__ == '__main__':

    # main settings
    Ns = 24
    Duration = 1
    use_acados = False
    n_iter = 100
    Ns_calib = 24
    Duration_calib = Duration*Ns_calib/Ns
    # calib_gain = [1.3, 1.3, 1.8] # optimal gains
    calib_gain = [1.5, 1.5, 1, 1]

    # generate data with custom model
    model_path = "./dummy.bioMod"
    orig_fiso_max = [547, 620, 1000, 895]
    orig_opt_len = [0.69, 0.72, 0.68, 0.73]
    orig_ins_point = [0.5, 0.5, 0.5, 0.5]
    orig_tendon_sl = [0.04, 0.04, 0.04, 0.04]
    ocp = generate_data(model_path, orig_fiso_max, orig_opt_len, orig_ins_point, Duration, Ns)
    opts = {"linear_solver": "ma57", "tol": 1e-4, "print_level": 0}
    sol = ocp.solve(solver_options=opts)
    muscles_ref = sol.controls['muscles'][:, :Ns_calib]
    states_ref = sol.states['all'][:, :Ns_calib+1]
    forces_ref = muscle_forces(sol.states['q'], sol.states['qdot'], sol.controls['muscles'], biorbd.Model(model_path))
    # start with wrong model
    # fiso_max = [547, 620, 1000, 895]
    fiso_max = [600, 600, 950, 850]
    # opt_len = [0.69, 0.72, 0.68, 0.73]
    opt_len = [0.725, 0.66, 0.73, 0.68]
    ins_point = [0.5, 0.5, 0.5, 0.5]
    # ins_point = [0.475, 0.515, 0.48, 0.52]
    # tendon_sl = [0.04, 0.04, 0.04, 0.04]
    tendon_sl = [0.01, 0.06, 0.02, 0.07]

    # recursively match data with optimized model
    fiso_vec = np.zeros((4, n_iter+1))
    opt_len_vec = np.zeros((4, n_iter+1))
    ins_point_vec = np.zeros((4, n_iter+1))
    tendon_sl_vec = np.zeros((4, n_iter+1))
    success_vec = np.ones((1, n_iter))

    init_fiso_max = fiso_max.copy()
    fiso_bounds = [500, 1100]
    init_opt_len = opt_len.copy()
    opt_len_bounds = [0.6, 0.8]
    init_ins_point = ins_point.copy()
    ins_point_bounds = [0.4, 0.7]
    init_tendon_sl = tendon_sl.copy()
    tendon_sl_bounds = [0.005, 0.07]

    g_opt_fiso = True
    g_opt_opt_len = True
    g_opt_ins_point = False
    g_opt_tendon_sl = True

    fiso_vec[:, 0] = fiso_max
    opt_len_vec[:, 0] = opt_len
    ins_point_vec[:, 0] = ins_point
    tendon_sl_vec[:, 0] = tendon_sl

    if g_opt_tendon_sl:
        tmp_opt = 'tendon'
        opt_fiso = False
        opt_opt_len = False
        opt_ins_point = False
        opt_tendon_sl = True
    elif g_opt_opt_len:
        opt_fiso = False
        opt_opt_len = True
        opt_ins_point = False
        opt_tendon_sl = False
        tmp_opt = 'opt_len'
    elif g_opt_fiso:
        opt_fiso = True
        opt_opt_len = False
        opt_ins_point = False
        opt_tendon_sl = False
        tmp_opt = 'fiso'

    break_loop = False

    for i in range(n_iter):
        sol = calibration_iter(model_path, opt_fiso, opt_opt_len, opt_ins_point, opt_tendon_sl,
                               fiso_max, init_fiso_max, fiso_bounds,
                               opt_len, init_opt_len, opt_len_bounds,
                               ins_point, init_ins_point, ins_point_bounds,
                               tendon_sl, init_tendon_sl, tendon_sl_bounds,
                               Duration_calib, Ns_calib, muscles_ref, states_ref, use_acados)
        success_vec[0, i] = sol.status

        if tmp_opt == 'fiso':
            fiso_vec[:, i+1] = sol.parameters['forces_iso_max'].squeeze()
            fiso_max = list(fiso_max + calib_gain[0]*(sol.parameters['forces_iso_max'].squeeze() - fiso_max))
            init_fiso_max = fiso_max
            opt_len_vec[:, i+1] = opt_len
            ins_point_vec[:, i+1] = ins_point
            tendon_sl_vec[:, i+1] = tendon_sl
            print(f"{i}th iter")
            print(f"{sol.parameters['forces_iso_max']} N")
            print(f"Original model forces isomax : {orig_fiso_max} N")
            opt_fiso = False
            opt_ins_point = False
            opt_opt_len = False
            opt_tendon_sl = True
            tmp_opt = 'tendon'

        elif tmp_opt == 'opt_len':
            opt_len_vec[:, i+1] = sol.parameters['optimal_length'].squeeze()
            opt_len = list(opt_len + calib_gain[1]*(sol.parameters['optimal_length'].squeeze() - opt_len))
            fiso_vec[:, i+1] = fiso_max
            ins_point_vec[:, i+1] = ins_point
            tendon_sl_vec[:, i+1] = tendon_sl
            init_opt_len = opt_len
            print(f"{i}th iter")
            print(f"{sol.parameters['optimal_length']} m")
            print(f"Original model optimal length : {orig_opt_len} m")

            opt_opt_len = False
            if g_opt_fiso:
                tmp_opt = 'fiso'
                opt_fiso = True
                opt_tendon_sl = False
            else:
                tmp_opt = 'tendon'
                opt_tendon_sl = True
                opt_fiso = False
        elif tmp_opt == 'ins_pt':
            ins_point_vec[:, i+1] = sol.parameters['insertion_point'].squeeze()
            ins_point = list(ins_point + calib_gain[2]*(sol.parameters['insertion_point'].squeeze() - ins_point))
            fiso_vec[:, i] = fiso_max
            opt_len_vec[:, i] = opt_len
            init_ins_point = ins_point
            print(f"{i}th iter")
            print(f"{sol.parameters['insertion_point']} m")
            print(f"Original model insertion point : {orig_ins_point} m")
            opt_fiso = False
            opt_ins_point = False
            opt_opt_len = True
        elif tmp_opt == 'tendon':
            tendon_sl_vec[:, i+1] = sol.parameters['tendon_slack_len'].squeeze()
            tendon_sl = list(tendon_sl + calib_gain[3]*(sol.parameters['tendon_slack_len'].squeeze() - tendon_sl))
            fiso_vec[:, i+1] = fiso_max
            opt_len_vec[:, i+1] = opt_len
            ins_point_vec[:, i+1] = ins_point
            init_tendon_sl = tendon_sl
            print(f"{i}th iter")
            print(f"{sol.parameters['tendon_slack_len']} m")
            print(f"Original model tendon slack len : {orig_tendon_sl} m")
            opt_fiso = False
            opt_ins_point = False
            opt_opt_len = True
            opt_tendon_sl = False
            tmp_opt = 'opt_len'

        conv_tol = 1e-3

        if g_opt_opt_len:
            break_loop = np.allclose(opt_len, orig_opt_len, conv_tol, conv_tol)
        else:
            break_loop = True

        if g_opt_fiso:
            break_loop = np.allclose(fiso_max, orig_fiso_max, conv_tol, conv_tol)*break_loop
        else:
            break_loop = True*break_loop

        if g_opt_tendon_sl:
            break_loop = np.allclose(tendon_sl, orig_tendon_sl, conv_tol, conv_tol)*break_loop
        else:
            break_loop = True*break_loop

        if break_loop:
            break

    forces_est = muscle_forces(sol.states['q'], sol.states['qdot'], sol.controls['muscles'], biorbd.Model(model_path))

    force_err = np.sqrt(np.sum((forces_ref-forces_est)**2, 1)/(Ns+1))
    force_err_percent = force_err/np.mean(forces_est, 1)*100
    fiso_err = np.sqrt(np.sum((np.array(fiso_max) - np.array(orig_fiso_max))**2)/4)
    fiso_err_percent = np.sqrt((np.array(fiso_max) - np.array(orig_fiso_max))**2)/np.array(orig_fiso_max)*100
    opt_len_err = np.sqrt(np.sum((np.array(opt_len) - np.array(orig_opt_len))**2)/4)
    opt_len_err_percent = np.sqrt((np.array(opt_len) - np.array(orig_opt_len))**2)/np.array(orig_opt_len)*100
    tendon_sl_err = np.sqrt(np.sum((np.array(tendon_sl) - np.array(orig_tendon_sl))**2)/4)
    tendon_sl_err_percent = np.sqrt((np.array(tendon_sl) - np.array(orig_tendon_sl))**2)/np.array(orig_tendon_sl)*100
    print(f"mean RMSE on muscle force : {np.mean(force_err)}")
    print(f"Percentage error on muscle force: {np.mean(force_err_percent)}")
    print(f"RMSE on force iso max : {fiso_err}")
    print(f"Percentage error on force iso max : {np.mean(fiso_err_percent)}")
    print(f"RMSE on opt len : {opt_len_err}")
    print(f"Percentage error on opt len : {np.mean(opt_len_err_percent)}")
    print(f"RMSE on tendon sl : {tendon_sl_err}")
    print(f"Percentage error on tendon sl : {np.mean(tendon_sl_err_percent)}")

    # sol.animate()
    # sol.graphs()
    plt.rcParams['font.size'] = '18'
    iters = np.linspace(0, i+1, num=i+2)
    plt.subplot(321)
    plt.plot(iters, fiso_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_fiso_max, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.legend()
    plt.title(f'Forces isomax calibration. Optimized={g_opt_fiso}')
    plt.xlabel('iterations')
    plt.ylabel('F_isomax')
    plt.subplot(322)
    plt.plot(iters, opt_len_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_opt_len, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.title(f'Optimal length calibration. Optimized={g_opt_opt_len}')
    plt.xlabel('iterations')
    plt.ylabel('Opt_length')
    plt.subplot(323)
    plt.plot(iters, ins_point_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_ins_point, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.title(f'Insertion point calibration. Optimized={g_opt_ins_point}')
    plt.xlabel('iterations')
    plt.ylabel('Ins_point')
    plt.subplot(324)
    plt.plot(iters, tendon_sl_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_tendon_sl, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.title(f'Tendon slack len calibration. Optimized={g_opt_tendon_sl}')
    plt.xlabel('iterations')
    plt.ylabel('slack len')
    plt.subplot(325)
    plt.plot(iters[1:], success_vec[:, :i+1].T, label='Convergence')
    plt.title('Convergence iter (0 = success / 1 = max_iter)')
    plt.xlabel('iterations')
    plt.ylabel('Convergence')
    plt.show()