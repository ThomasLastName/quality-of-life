
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import cvxpy as cvx
import numpy as np
from quality_of_life.my_base_utils import my_warn

#
# ~~~ Try to find a vector x satistfying Ax \geq b
def linear_feasibility_program( A, b, solver=cvx.ECOS, verbose=True ):
    m,n = A.shape               # ~~~ get the number of rows and columns of A
    assert b.shape==(m,1)       # ~~~ safety feature
    x = cvx.Variable((n,1))     # ~~~ define the optimization variable x
    constraints = [A @ x >= b]  # ~~~ define the constraints of the problem
    objective = cvx.Minimize(1) # ~~~ objective function can be anything for linear feasibility problem
    problem = cvx.Problem(objective, constraints)   # ~~~ put it all together into a complete minimization program
    problem.solve(solver=solver)              # ~~~ try to solve it
    if problem.status==cvx.OPTIMAL:
        return x.value
    elif verbose:
        my_warn("The constraint set was found to be infeasible")

#
# ~~~ Choose v to minimize v^TQv + v^Tg subject to Av \geq b
def minimize_quadradic_with_affine_constraints( Q, g, A, b, solver=cvx.ECOS, verbose=True ):
    m,n = A.shape               # ~~~ get the number of rows and columns of A
    assert b.shape==(m,1)       # ~~~ safety feature
    assert Q.shape==(n,n)       # ~~~ safety feature
    # assert Q>>0               # ~~~ this important safety feature is omitted
    v = cvx.Variable((n,1))     # ~~~ define the optimization variable x
    constraints = [A @ v >= b]  # ~~~ define the constraints of the problem
    objective = cvx.Minimize( cvx.quad_form(v,Q) + v.T@g )  # ~~~ the objective function to be minimized
    problem = cvx.Problem(objective,constraints)    # ~~~ put it all together into a complete minimization program
    problem.solve(solver=solver)                    # ~~~ try to solve it
    if problem.status==cvx.OPTIMAL:
        return v.value
    elif verbose:
        try:
            linear_feasibility_program(A,b)
        except:
            my_warn(f"The constraint set is feasiible, but the solver {solver} failed, anyway")

#
# ~~~ Verify the assumptions of a quadratically constrained quadratic program (QCQP)
def verify_QCQP_assumptions(
        #
        # ~~~ Objective(x) = x.T@H_o@x + 2*c_o.T@x + d_o
        H_o,
        c_o,
        d_o,
        #
        # ~~~ Constrain x.T@H_i@x + 2*c_i.T@x + d_i \leq 0 for each i
        H_I = list(),   # ~~~ H_I is the list of the H_i's
        c_I = list(),   # ~~~ c_I is the list of the c_i's
        d_I = list(),   # ~~~ d_I is the list of the d_i's
        #
        # ~~~ Constrain x.T@H_j@x + 2*c_j.T@x + d_j == 0 for each j
        H_J = list(),   # ~~~ H_J is the list of the H_j's
        c_J = list(),   # ~~~ c_J is the list of the c_j's
        d_J = list(),   # ~~~ d_J is the list of the d_j's
    ):
    #
    # ~~~ Verify several type assumptions
    n_inequality_constraints = len(H_I)
    n_equality_constraints   = len(H_J)
    assert len(c_I)==n_inequality_constraints,  f"Please verify that len(H_I)==len(c_I)==len(d_I). Presnetly, len(c_I)={len(c_I)} but len(H_I)={n_inequality_constraints}"
    assert len(d_I)==n_inequality_constraints,  f"Please verify that len(H_I)==len(c_I)==len(d_I). Presnetly, len(d_I)={len(d_I)} but len(H_I)={n_inequality_constraints}"
    assert len(c_J)==n_equality_constraints,    f"Please verify that len(H_J)==len(c_J)==len(d_J). Presnetly, len(c_J)={len(c_J)} but len(H_J)={n_inequality_constraints}"
    assert len(d_J)==n_equality_constraints,    f"Please verify that len(H_J)==len(c_J)==len(d_J). Presnetly, len(d_J)={len(d_J)} but len(H_J)={n_inequality_constraints}"
    #
    # ~~~ Verify several shape assumptions
    assert c_o.ndim == 1
    n_primal_variables = len(c_o)
    assert H_o.shape == (n_primal_variables, n_primal_variables)
    for i in range(n_inequality_constraints):
        assert H_I[i].shape == (n_primal_variables, n_primal_variables)
        assert c_I[i].shape == (n_primal_variables,)
    for j in range(n_equality_constraints):
        assert H_J[j].shape == (n_primal_variables, n_primal_variables)
        assert c_J[j].shape == (n_primal_variables,)
    #
    # ~~~ Enforce symmetry assumptions
    any_non_symmetric = False
    if not np.array_equal(H_o,H_o.T):
        H_o = (H_o + H_o.T)/2
        any_non_symmetric = True
    for i in range(n_inequality_constraints):
        if not np.array_equal( H_I[i], H_I[i].T ):
            H_I[i] = ( H_I[i] + H_I[i].T )/2
            any_non_symmetric = True
    for j in range(n_equality_constraints):
        if not np.array_equal( H_J[j], H_J[j].T ):
            H_J[j] = ( H_J[j] + H_J[j].T )/2
            any_non_symmetric = True
    if any_non_symmetric:
        my_warn("One or more of the supplied matrices was not symmetric. The symmetric part (A+A.T)/2 of any such matrix A will be used instead.")
    #
    # ~~~ Sanity check
    x = np.random.normal(size=(n_primal_variables,))
    _ = np.inner(x,H_o@x) + np.inner(c_o,x) + d_o
    for i in range(n_inequality_constraints):
        _ = np.inner(x,H_I[i]@x) + np.inner(c_I[i],x) + d_I[i]
    for j in range(n_equality_constraints):
        _ = np.inner(x,H_J[j]@x) + np.inner(c_J[j],x) + d_J[j]
    #
    # ~~~ Return the processed problem data
    return H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J

#
# ~~~ Implement equation (9) of https://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec12_gh.pdf
def solve_dual_of_QCQP(
        #
        # ~~~ Objective(x) = x.T@H_o@x + 2*c_o.T@x + d_o
        H_o,
        c_o,
        d_o,
        #
        # ~~~ Constrain x.T@H_i@x + 2*c_i.T@x + d_i \leq 0 for each i
        H_I = list(),   # ~~~ H_I is the list of the H_i's
        c_I = list(),   # ~~~ c_I is the list of the c_i's
        d_I = list(),   # ~~~ d_I is the list of the d_i's
        #
        # ~~~ Constrain x.T@H_j@x + 2*c_j.T@x + d_j == 0 for each j
        H_J = list(),   # ~~~ H_J is the list of the H_j's
        c_J = list(),   # ~~~ c_J is the list of the c_j's
        d_J = list(),   # ~~~ d_J is the list of the d_j's
        solver = cvx.SCS,
        *args,
        **kwargs
    ):
    H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J = verify_QCQP_assumptions( H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J )
    n_inequality_constraints = len(H_I)
    n_equality_constraints = len(H_J)
    n_primal_variables = len(c_o)
    #
    # ~~~ Define the semi-definite program according to equation (9) of https://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec12_gh.pdf
    lamb = cvx.Variable( n_inequality_constraints, nonneg=True ) if n_inequality_constraints>0 else None
    eta  = cvx.Variable( n_equality_constraints ) if n_equality_constraints>0 else None
    gamma = cvx.Variable(1)
    quadratic_part  =  H_o + sum(lamb[i]*H_I[i] for i in range(n_inequality_constraints)) - sum(eta[j]*H_J[j] for j in range(n_equality_constraints))
    linear_part     =  c_o + sum(lamb[i]*c_I[i] for i in range(n_inequality_constraints)) - sum(eta[j]*c_J[j] for j in range(n_equality_constraints))
    constant_part   =  d_o + sum(lamb[i]*d_I[i] for i in range(n_inequality_constraints)) - sum(eta[j]*d_J[j] for j in range(n_equality_constraints))
    M = cvx.vstack([
            cvx.hstack([ quadratic_part, cvx.reshape( linear_part, (n_primal_variables,1) ) ]),
            cvx.reshape( cvx.hstack([ linear_part, constant_part-gamma ]), (1,n_primal_variables+1) )
        ])
    problem = cvx.Problem( cvx.Maximize(gamma), [M>>0] )
    problem.solve( solver=solver, *args, **kwargs )
    return problem, gamma, lamb, eta
