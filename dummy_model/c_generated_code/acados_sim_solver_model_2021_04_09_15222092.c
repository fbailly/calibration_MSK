/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */
// standard
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/print.h"


// example specific
#include "model_2021_04_09_15222092_model/model_2021_04_09_15222092_model.h"
#include "acados_sim_solver_model_2021_04_09_15222092.h"


// ** global data **
sim_config  * model_2021_04_09_15222092_sim_config;
sim_in      * model_2021_04_09_15222092_sim_in;
sim_out     * model_2021_04_09_15222092_sim_out;
void        * model_2021_04_09_15222092_sim_dims;
sim_opts    * model_2021_04_09_15222092_sim_opts;
sim_solver  * model_2021_04_09_15222092_sim_solver;


external_function_param_casadi * sim_impl_dae_fun;
external_function_param_casadi * sim_impl_dae_fun_jac_x_xdot_z;
external_function_param_casadi * sim_impl_dae_jac_x_xdot_u_z;



int model_2021_04_09_15222092_acados_sim_create()
{
    // initialize
    int nx = 8;
    int nu = 4;
    int nz = 0;

    
    double Tsim = 0.041666666666666664;

    
    sim_impl_dae_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));
    sim_impl_dae_fun_jac_x_xdot_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));
    sim_impl_dae_jac_x_xdot_u_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));

    // external functions (implicit model)
    sim_impl_dae_fun->casadi_fun  = &model_2021_04_09_15222092_impl_dae_fun;
    sim_impl_dae_fun->casadi_work = &model_2021_04_09_15222092_impl_dae_fun_work;
    sim_impl_dae_fun->casadi_sparsity_in = &model_2021_04_09_15222092_impl_dae_fun_sparsity_in;
    sim_impl_dae_fun->casadi_sparsity_out = &model_2021_04_09_15222092_impl_dae_fun_sparsity_out;
    sim_impl_dae_fun->casadi_n_in = &model_2021_04_09_15222092_impl_dae_fun_n_in;
    sim_impl_dae_fun->casadi_n_out = &model_2021_04_09_15222092_impl_dae_fun_n_out;
    external_function_param_casadi_create(sim_impl_dae_fun, 0);

    sim_impl_dae_fun_jac_x_xdot_z->casadi_fun = &model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z;
    sim_impl_dae_fun_jac_x_xdot_z->casadi_work = &model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_work;
    sim_impl_dae_fun_jac_x_xdot_z->casadi_sparsity_in = &model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_sparsity_in;
    sim_impl_dae_fun_jac_x_xdot_z->casadi_sparsity_out = &model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_sparsity_out;
    sim_impl_dae_fun_jac_x_xdot_z->casadi_n_in = &model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_n_in;
    sim_impl_dae_fun_jac_x_xdot_z->casadi_n_out = &model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_n_out;
    external_function_param_casadi_create(sim_impl_dae_fun_jac_x_xdot_z, 0);

    // external_function_param_casadi impl_dae_jac_x_xdot_u_z;
    sim_impl_dae_jac_x_xdot_u_z->casadi_fun = &model_2021_04_09_15222092_impl_dae_jac_x_xdot_u_z;
    sim_impl_dae_jac_x_xdot_u_z->casadi_work = &model_2021_04_09_15222092_impl_dae_jac_x_xdot_u_z_work;
    sim_impl_dae_jac_x_xdot_u_z->casadi_sparsity_in = &model_2021_04_09_15222092_impl_dae_jac_x_xdot_u_z_sparsity_in;
    sim_impl_dae_jac_x_xdot_u_z->casadi_sparsity_out = &model_2021_04_09_15222092_impl_dae_jac_x_xdot_u_z_sparsity_out;
    sim_impl_dae_jac_x_xdot_u_z->casadi_n_in = &model_2021_04_09_15222092_impl_dae_jac_x_xdot_u_z_n_in;
    sim_impl_dae_jac_x_xdot_u_z->casadi_n_out = &model_2021_04_09_15222092_impl_dae_jac_x_xdot_u_z_n_out;
    external_function_param_casadi_create(sim_impl_dae_jac_x_xdot_u_z, 0);

    

    // sim plan & config
    sim_solver_plan plan;
    plan.sim_solver = IRK;

    // create correct config based on plan
    model_2021_04_09_15222092_sim_config = sim_config_create(plan);

    // sim dims
    model_2021_04_09_15222092_sim_dims = sim_dims_create(model_2021_04_09_15222092_sim_config);
    sim_dims_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims, "nx", &nx);
    sim_dims_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims, "nu", &nu);
    sim_dims_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims, "nz", &nz);


    // sim opts
    model_2021_04_09_15222092_sim_opts = sim_opts_create(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims);
    int tmp_int = 4;
    sim_opts_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_opts, "num_stages", &tmp_int);
    tmp_int = 1;
    sim_opts_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_opts, "num_steps", &tmp_int);
    tmp_int = 5;
    sim_opts_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_opts, "newton_iter", &tmp_int);
    bool tmp_bool = false;
    sim_opts_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_opts, "jac_reuse", &tmp_bool);



    // sim in / out
    model_2021_04_09_15222092_sim_in  = sim_in_create(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims);
    model_2021_04_09_15222092_sim_out = sim_out_create(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims);
    sim_in_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims,
               model_2021_04_09_15222092_sim_in, "T", &Tsim);

    // model functions
    model_2021_04_09_15222092_sim_config->model_set(model_2021_04_09_15222092_sim_in->model,
                 "impl_ode_fun", sim_impl_dae_fun);
    model_2021_04_09_15222092_sim_config->model_set(model_2021_04_09_15222092_sim_in->model,
                 "impl_ode_fun_jac_x_xdot", sim_impl_dae_fun_jac_x_xdot_z);
    model_2021_04_09_15222092_sim_config->model_set(model_2021_04_09_15222092_sim_in->model,
                 "impl_ode_jac_x_xdot_u", sim_impl_dae_jac_x_xdot_u_z);

    // sim solver
    model_2021_04_09_15222092_sim_solver = sim_solver_create(model_2021_04_09_15222092_sim_config,
                                               model_2021_04_09_15222092_sim_dims, model_2021_04_09_15222092_sim_opts);

    /* initialize parameter values */
    

    /* initialize input */
    // x
    double x0[8];
    for (int ii = 0; ii < 8; ii++)
        x0[ii] = 0.0;

    sim_in_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims,
               model_2021_04_09_15222092_sim_in, "x", x0);


    // u
    double u0[4];
    for (int ii = 0; ii < 4; ii++)
        u0[ii] = 0.0;

    sim_in_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims,
               model_2021_04_09_15222092_sim_in, "u", u0);

    // S_forw
    double S_forw[96];
    for (int ii = 0; ii < 96; ii++)
        S_forw[ii] = 0.0;
    for (int ii = 0; ii < 8; ii++)
        S_forw[ii + ii * 8 ] = 1.0;


    sim_in_set(model_2021_04_09_15222092_sim_config, model_2021_04_09_15222092_sim_dims,
               model_2021_04_09_15222092_sim_in, "S_forw", S_forw);

    int status = sim_precompute(model_2021_04_09_15222092_sim_solver, model_2021_04_09_15222092_sim_in, model_2021_04_09_15222092_sim_out);

    return status;
}


int model_2021_04_09_15222092_acados_sim_solve()
{
    // integrate dynamics using acados sim_solver
    int status = sim_solve(model_2021_04_09_15222092_sim_solver,
                           model_2021_04_09_15222092_sim_in, model_2021_04_09_15222092_sim_out);
    if (status != 0)
        printf("error in model_2021_04_09_15222092_acados_sim_solve()! Exiting.\n");

    return status;
}


int model_2021_04_09_15222092_acados_sim_free()
{
    // free memory
    sim_solver_destroy(model_2021_04_09_15222092_sim_solver);
    sim_in_destroy(model_2021_04_09_15222092_sim_in);
    sim_out_destroy(model_2021_04_09_15222092_sim_out);
    sim_opts_destroy(model_2021_04_09_15222092_sim_opts);
    sim_dims_destroy(model_2021_04_09_15222092_sim_dims);
    sim_config_destroy(model_2021_04_09_15222092_sim_config);

    // free external function
    external_function_param_casadi_free(sim_impl_dae_fun);
    external_function_param_casadi_free(sim_impl_dae_fun_jac_x_xdot_z);
    external_function_param_casadi_free(sim_impl_dae_jac_x_xdot_u_z);

    return 0;
}


int model_2021_04_09_15222092_acados_sim_update_params(double *p, int np)
{
    int status = 0;
    int casadi_np = 0;

    if (casadi_np != np) {
        printf("model_2021_04_09_15222092_acados_sim_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    sim_impl_dae_fun[0].set_param(sim_impl_dae_fun, p);
    sim_impl_dae_fun_jac_x_xdot_z[0].set_param(sim_impl_dae_fun_jac_x_xdot_z, p);
    sim_impl_dae_jac_x_xdot_u_z[0].set_param(sim_impl_dae_jac_x_xdot_u_z, p);

    return status;
}

/* getters pointers to C objects*/
sim_config * model_2021_04_09_15222092_acados_get_sim_config()
{
    return model_2021_04_09_15222092_sim_config;
};

sim_in * model_2021_04_09_15222092_acados_get_sim_in()
{
    return model_2021_04_09_15222092_sim_in;
};

sim_out * model_2021_04_09_15222092_acados_get_sim_out()
{
    return model_2021_04_09_15222092_sim_out;
};

void * model_2021_04_09_15222092_acados_get_sim_dims()
{
    return model_2021_04_09_15222092_sim_dims;
};

sim_opts * model_2021_04_09_15222092_acados_get_sim_opts()
{
    return model_2021_04_09_15222092_sim_opts;
};

sim_solver  * model_2021_04_09_15222092_acados_get_sim_solver()
{
    return model_2021_04_09_15222092_sim_solver;
};

