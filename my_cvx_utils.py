
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import cvxpy as cvx
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

