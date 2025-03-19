
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import cvxpy as cvx
import numpy as np
from tqdm.auto import tqdm
from quality_of_life.my_base_utils import my_warn, support_for_progress_bars


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
        #
        # ~~~ Other
        solver = cvx.SCS,
        max_fw_iter = None,
        debug = False,
        *args,
        **kwargs
    ):
    H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J = verify_QCQP_assumptions( H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J )
    n_inequality_constraints = len(H_I)
    n_equality_constraints = len(H_J)
    n_primal_variables = len(c_o)
    if max_fw_iter is None:
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
        #
        # ~~~ Solve it
        problem = cvx.Problem( cvx.Maximize(gamma), [M>>0] )
        problem.solve( solver=solver, *args, **kwargs )
        if eta is not None:
            return ( problem, gamma.value, lamb.value, eta.value ) if not debug else ( problem, gamma, lamb, eta )
        else:
            return ( problem, gamma.value, lamb.value ) if not debug else ( problem, gamma, lamb )
    else:
        raise NotImplementedError
        A = np.concatenate([ np.stack(H_I), np.stack(H_J) ])
        b = np.concatenate([ np.stack(c_I), np.stack(c_J) ])
        c = np.concatenate([ np.stack(d_I), np.stack(d_J) ])
        Q = H_o
        r = c_o
        s = d_o
        lr = 0.00001
        eps = 1e-5
        alpha = 0.5
        ALPHA = 1.1
        t = 0
        theta = np.random.normal( size=(n_inequality_constraints + n_equality_constraints,) )
        for j in range(n_inequality_constraints):
            theta[j] = theta[j]**2
        calq = Q + (theta.reshape(-1,1,1)*A).sum(axis=0)    # ~~~ Q(\theta)     = I + \sum_{j=1}^{k-1} \theta_j A_j
        beta = r + (theta.reshape(-1,1)*b).sum(axis=0)      # ~~~ \beta(\theta) = r + \sum_{j=1}^{k-1} \theta_j b_j
        z = np.linalg.solve( calq, beta )
        with support_for_progress_bars():
            pbar = tqdm( desc="Solving the Dual Problem Using Frank-Wolfe", total=max_fw_iter, ascii=' >=' )
            for _ in range(max_fw_iter):
                #
                # ~~~ Compute the gradient
                g = (A@z)@z + b@z + c
                g *= lr
                #
                # ~~~ Perform the Frank-Wolfe sub-routine
                keep_trying = True
                while keep_trying:
                    fw_var = cvx.Variable( n_inequality_constraints + n_equality_constraints )
                    objective = cvx.Maximize( g@fw_var )
                    constraints = [
                            Q + sum( fw_var[j]*A[j] for j in range(n_inequality_constraints+n_equality_constraints) ) >> eps*np.eye(n_primal_variables)
                        ] + [
                            fw_var[j] >= 0 for j in range(n_inequality_constraints)
                        ]
                    problem = cvx.Problem(objective, constraints)
                    duality_gap = problem.solve( solver=solver, *args, **kwargs )/lr
                    if duality_gap == float("inf"):
                        lr *= alpha
                        g  *= alpha
                        # print(f"decreasing learning rate to {lr}")
                    else:
                        lr *= ALPHA
                        alpha = 2/(t+2)
                        # print(problem.status)
                        theta = (1-alpha)*theta + alpha*fw_var.value
                        t += 1
                        keep_trying = False
                        # print(f"increasing learning rate to {lr}")
                calq = Q + (theta.reshape(-1,1,1)*A).sum(axis=0)    # ~~~ Q(\theta)     = I + \sum_{j=1}^{k-1} \theta_j A_j
                beta = r + (theta.reshape(-1,1)*b).sum(axis=0)      # ~~~ \beta(\theta) = r + \sum_{j=1}^{k-1} \theta_j b_j
                z = np.linalg.solve( calq, beta )
                dual_objective_value = np.inner(z,beta) + np.inner(theta,c) + s
                pbar.set_postfix({
                        "dual": f"{dual_objective_value:<4.4f}",
                        "gap" : f"{duality_gap:<4.4f}"
                    })
                pbar.update()
        # calQ  =  H_o + sum(lamb[i]*H_I[i] for i in range(n_inequality_constraints)) - sum(eta[j]*H_J[j] for j in range(n_equality_constraints))
        # beta  =  c_o + sum(lamb[i]*c_I[i] for i in range(n_inequality_constraints)) - sum(eta[j]*c_J[j] for j in range(n_equality_constraints))
        # cnst  =  d_o + sum(lamb[i]*d_I[i] for i in range(n_inequality_constraints)) - sum(eta[j]*d_J[j] for j in range(n_equality_constraints))
        # z = np.linalg.solve( calQ, beta )
        # dual_objective_value = np.inner(z,beta) + cnst
        return None

#
# ~~~ Implement the stated relaxation of (10) of https://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec12_gh.pdf
def solve_rank_relaxation_of_QCQP(
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
        #
        # ~~~ Other
        solver = cvx.SCS,
        debug = False,
        *args,
        **kwargs
    ):
    H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J = verify_QCQP_assumptions( H_o, c_o, d_o, H_I, c_I, d_I, H_J, c_J, d_J )
    n_inequality_constraints = len(H_I)
    n_equality_constraints = len(H_J)
    n_primal_variables = len(c_o)
    X = cvx.Variable( (n_primal_variables,n_primal_variables), PSD=True )
    x = cvx.Variable( n_primal_variables )
    objective = cvx.Minimize( cvx.trace(H_o@X) + 2*cvx.sum(cvx.multiply(c_o,x)) + d_o )
    constraints = [ Schur_complement(X,x) ]
    for i in range(n_inequality_constraints): constraints.append( cvx.trace(H_I[i]@X) + 2*cvx.sum(cvx.multiply(c_I[i],x)) + d_I[i] <= 0 )
    for j in range(n_equality_constraints): constraints.append( cvx.trace(H_J[j]@X) + 2*cvx.sum(cvx.multiply(c_J[j],x)) + d_J[j] == 0 )
    problem = cvx.Problem( objective, constraints )
    problem.solve( solver=solver, *args, **kwargs )
    return (problem, X.value, x.value) if not debug else (problem,X,x)

#
# ~~~ Form the constraint X \geq x@x.T
def Schur_complement( X, x ):
    m,n = X.shape
    assert m==n and x.shape==(n,)
    return cvx.bmat([
            [ X,                      cvx.reshape(x, (n, 1)) ],
            [ cvx.reshape(x, (1, n)), np.array([[1.]])       ]
        ]) >> 0
