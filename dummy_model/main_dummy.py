import biorbd
from casadi import MX,Function
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

def prepare_ocp(
    model_path: str,
    param_opt_vec: list,
    fiso_max: list, fiso_bounds: list, fiso_init: list,
    opt_len: list, opt_len_bounds: list, opt_len_init: list,
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
    set_tendon_sl(model, tendon_sl)

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
    bounds_tendon_sl = Bounds([tendon_sl_bounds[0]]*model.nbMuscles(),
                              [tendon_sl_bounds[1]]*model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    initial_tendon_sl = InitialGuess(tendon_sl_init)
    if param_opt_vec[0]:
        parameters.add("forces_iso_max", set_fiso_max, initial_fiso, bounds_fiso,
                       size=model.nbMuscles(), scaling=np.array([500]),)
    if param_opt_vec[1]:
        parameters.add("optimal_length", set_opt_len, initial_opt_len, bounds_opt_len,
                       size=model.nbMuscles(), scaling=np.array([0.5]))
    if param_opt_vec[2]:
        parameters.add("tendon_slack_len", set_tendon_sl, initial_tendon_sl, bounds_tendon_sl,
                       size=model.nbMuscles(), scaling=np.array([0.05]))
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
                     param_opt_vec,
                     fiso_max, init_fiso_max, fiso_bounds,
                     opt_len, init_opt_len, opt_len_bounds,
                     tendon_sl, init_tendon_sl, tendon_sl_bounds,
                     Duration, Ns, muscles_ref, states_ref, use_acados=False):
    ocp = prepare_ocp(model_path,
                      param_opt_vec,
                      fiso_max, fiso_bounds, init_fiso_max,
                      opt_len, opt_len_bounds, init_opt_len,
                      tendon_sl, tendon_sl_bounds, init_tendon_sl,
                      Duration, Ns, muscles_ref, states_ref, use_acados=use_acados)
    if use_acados:
        solver = Solver.ACADOS
        opts={"print_level": 0}
    else:
        solver = Solver.IPOPT
        opts={"linear_solver": "ma57", "max_iter": 100, "print_level": 0, "hessian_approximation": "exact"}
    sol = ocp.solve(solver=solver, solver_options=opts)
    return sol

if __name__ == '__main__':

    # main settings
    Ns = 32
    Duration = 1
    use_acados = False
    n_iter = 20
    Ns_calib = 32
    Duration_calib = Duration*Ns_calib/Ns
    # calib_gain = [1.3, 1.3, 1.8] # optimal gains
    calib_gain = [1, 1, 1]

    # generate data with custom model
    model_path = "./dummy.bioMod"
    orig_fiso_max = [825, 920, 1200, 1125]
    orig_opt_len = [0.69, 0.72, 0.68, 0.73]
    orig_tendon_sl = [0.2, 0.2, 0.2, 0.2]
    ocp = generate_data(model_path, orig_fiso_max, orig_opt_len, orig_tendon_sl, Duration, Ns)
    opts = {"linear_solver": "ma57", "tol": 1e-4, "print_level": 5}
    sol = ocp.solve(solver_options=opts)
    muscles_ref = sol.controls['muscles'][:, :Ns_calib]
    states_ref = sol.states['all'][:, :Ns_calib+1]
    forces_ref = muscle_forces(sol.states['q'], sol.states['qdot'], sol.controls['muscles'], biorbd.Model(model_path))[:, :Ns_calib+1]
    # start with wrong model
    fiso_max = [825, 920, 1200, 1125]
    # fiso_max = [600, 600, 1050, 1000]
    # opt_len = [0.69, 0.72, 0.68, 0.73]
    opt_len = [0.725, 0.66, 0.73, 0.68]
    # tendon_sl = [0.2, 0.2, 0.2, 0.2]
    tendon_sl = [0.22, 0.18, 0.21, 0.17]

    # recursively match data with optimized model
    fiso_vec = np.zeros((4, n_iter+2))
    opt_len_vec = np.zeros((4, n_iter+2))
    tendon_sl_vec = np.zeros((4, n_iter+2))
    success_vec = np.ones((1, n_iter+1))

    init_fiso_max = fiso_max.copy()
    fiso_bounds = [500, 1300]
    init_opt_len = opt_len.copy()
    opt_len_bounds = [0.6, 0.8]
    init_tendon_sl = tendon_sl.copy()
    tendon_sl_bounds = [0.1, 0.5]

    g_opt_fiso = False
    g_opt_opt_len = True
    g_opt_tendon_sl = True  # False, True, "LOW_FREQ"
    freq_tendon_calib = 10

    fiso_vec[:, 0] = fiso_max
    opt_len_vec[:, 0] = opt_len
    tendon_sl_vec[:, 0] = tendon_sl

    if g_opt_opt_len:
        param_opt_vec, param_opt = set_param_opt('opt_len')
    elif g_opt_tendon_sl:
        param_opt_vec, param_opt = set_param_opt('tendon')
    elif g_opt_fiso:
        param_opt_vec, param_opt = set_param_opt('fiso')

    # param_opt_vec, param_opt  =set_param_opt('none')
    break_loop = False

    for i in range(n_iter+1):

        if g_opt_tendon_sl == 'LOW_FREQ':
            if (i+1) % freq_tendon_calib == 0:
                param_opt_vec, param_opt = set_param_opt('tendon')

        sol = calibration_iter(model_path, param_opt_vec,
                               fiso_max, init_fiso_max, fiso_bounds,
                               opt_len, init_opt_len, opt_len_bounds,
                               tendon_sl, init_tendon_sl, tendon_sl_bounds,
                               Duration_calib, Ns_calib, muscles_ref, states_ref, use_acados)
        success_vec[0, i] = sol.status

        if param_opt == 'fiso':
            fiso_vec[:, i+1] = sol.parameters['forces_iso_max'].squeeze()
            fiso_max = list(fiso_max + calib_gain[0]*(sol.parameters['forces_iso_max'].squeeze() - fiso_max))
            init_fiso_max = fiso_max
            opt_len_vec[:, i+1] = opt_len_vec[:, i]
            tendon_sl_vec[:, i+1] = tendon_sl_vec[:, i]
            print(f"{i}th iter")
            print(f"{sol.parameters['forces_iso_max']} N")
            print(f"Original model forces isomax : {orig_fiso_max} N")
            if g_opt_tendon_sl == True:
                param_opt_vec, param_opt = set_param_opt('tendon')
            elif g_opt_opt_len:
                param_opt_vec, param_opt = set_param_opt('opt_len')
            else:
                pass

        elif param_opt == 'opt_len':
            opt_len_vec[:, i+1] = sol.parameters['optimal_length'].squeeze()
            opt_len = list(opt_len + calib_gain[1]*(sol.parameters['optimal_length'].squeeze() - opt_len))
            fiso_vec[:, i+1] = fiso_vec[:, i]
            tendon_sl_vec[:, i+1] = tendon_sl_vec[:, i]
            init_opt_len = opt_len
            print(f"{i}th iter")
            print(f"{sol.parameters['optimal_length']} m")
            print(f"Original model optimal length : {orig_opt_len} m")
            if g_opt_fiso:
                param_opt_vec, param_opt = set_param_opt('fiso')
            elif g_opt_tendon_sl:
                param_opt_vec, param_opt = set_param_opt('tendon')
            else :
                pass

        elif param_opt == 'tendon':
            tendon_sl_vec[:, i+1] = sol.parameters['tendon_slack_len'].squeeze()
            tendon_sl = list(tendon_sl + calib_gain[2]*(sol.parameters['tendon_slack_len'].squeeze() - tendon_sl))
            fiso_vec[:, i+1] = fiso_vec[:, i]
            opt_len_vec[:, i+1] = opt_len_vec[:, i]
            init_tendon_sl = tendon_sl
            print(f"{i}th iter")
            print(f"{sol.parameters['tendon_slack_len']} m")
            print(f"Original model tendon slack len : {orig_tendon_sl} m")
            if g_opt_opt_len:
                param_opt_vec, param_opt = set_param_opt('opt_len')
            elif g_opt_fiso:
                param_opt_vec, param_opt = set_param_opt('fiso')
            else:
                pass
        conv_tol = 1e-2

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
    sol.graphs()
    plt.rcParams['font.size'] = '18'
    iters = np.linspace(0, i+1, num=i+2)
    plt.subplot(221)
    plt.plot(iters, fiso_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_fiso_max, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.legend()
    plt.title(f'Forces isomax calibration. Optimized={g_opt_fiso}')
    plt.xlabel('iterations')
    plt.ylabel('F_isomax')
    plt.subplot(222)
    plt.plot(iters, opt_len_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_opt_len, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.title(f'Optimal length calibration. Optimized={g_opt_opt_len}')
    plt.xlabel('iterations')
    plt.ylabel('Opt_length')

    plt.subplot(223)
    plt.plot(iters, tendon_sl_vec[:, :i+2].T, 'x-', label='Calibration')
    plt.gca().set_prop_cycle(None)
    plt.plot(iters, np.repeat(np.array(orig_tendon_sl, ndmin=2).T, i+2, 1).T, 'o--', label='True values')
    plt.title(f'Tendon slack len calibration. Optimized={g_opt_tendon_sl}')
    plt.xlabel('iterations')
    plt.ylabel('slack len')
    plt.subplot(224)
    plt.plot(iters[1:], success_vec[:, :i+1].T, label='Convergence')
    plt.title('Convergence iter (0 = success / 1 = max_iter)')
    plt.xlabel('iterations')
    plt.ylabel('Convergence')
    plt.show()