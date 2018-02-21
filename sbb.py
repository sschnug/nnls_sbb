import numpy as np

"""
FUNCTION
"""
def project(x):
    x = np.clip(x, 0, np.inf, out=x)
    return x

def func(A, x, b):
    return 0.5*np.linalg.norm(A.dot(x) - b)**2

def get_binding_set(x, grad):
    return np.where( (x==0) & (grad>0) )  # TODO -> tolerances

def calc_grad(A, x, b, non_binding=None):
    Ax = A.dot(x) - b
    grad = A.T.dot(Ax)
    if non_binding is not None:
        grad[non_binding] = 0.0
    return grad

def calc_grad_precomputed(AtA, x, b, non_binding=None):
    grad = AtA.dot(x) - A.T.dot(b)
    if non_binding is not None:
        grad[non_binding] = 0.0
    return grad

#@profile
def sbb(A, b, x0=None, precompute_AtA=False, M_inner=50, beta=0.05, mu=0.95, sigma=0.01, maxit=100, pg_tol=1e-4):
    """ Init """
    M, N = A.shape
    indices_full = np.arange(N)

    if precompute_AtA:
        AtA = A.T.dot(A)

    # Iterates
    if x0 is None:
        x0 = np.zeros(N) + 1e-4
    x_prev = x0[:]
    x = x0[:]

    # Gradients
    grad_prev = None
    if precompute_AtA:
        grad_prev = calc_grad_precomputed(AtA, x, b)
    else:
        grad_prev = calc_grad(A, x, b)
    grad = grad_prev[:]

    # Objectives
    obj_prev = func(A, x, b)

    for i in range(1, maxit+1):
        print(i)
        if check_termination(x_prev, grad_prev, pg_tol):
            return x_prev

        grad = None
        for j in range(1, M_inner+1):
            print('.', end='', flush=True)
            binding = get_binding_set(x_prev, grad_prev)
            non_binding = np.setdiff1d(indices_full, binding, assume_unique=True)

            grad = None
            if precompute_AtA:
                grad = calc_grad_precomputed(AtA, x, b, binding)
            else:
                grad = calc_grad(A, x, b, binding)

            if precompute_AtA:
                tmp_0 = AtA.dot(grad)
            else:
                tmp_0 = A.T.dot(A.dot(grad))

            tmp_1 = grad.dot(tmp_0)
            alpha = 0

            if j % 2 == 0:
                upper = np.linalg.norm(grad)**2
                alpha = upper / tmp_1
            else:
                lower = np.linalg.norm(tmp_0)**2
                alpha = tmp_1 / lower

            x = project(x - beta * alpha * grad)

        obj_new = func(A, x, b)
        if (obj_prev - obj_new) >= (sigma * grad_prev.dot(x_prev - x)):
            x_prev = x[:]
            obj_prev = obj_new
            grad_prev = grad[:]
        else:
            beta *= mu

        print(' -> obj: ', obj_prev)

    return x_prev

def check_termination(x, grad, tolg):
    # PG-norm limit
    gp = np.where( (x!=0.0) | (grad<0.0))
    grad_gp = grad[gp]
    pg = np.linalg.norm(grad_gp, ord=np.inf)
    #print('pg: ', pg)
    if pg < tolg:
        return 8

""" TEST """
import cvxpy as cvx
from scipy.optimize import minimize
from time import perf_counter as pc

def solve_scs(A, b):
    M, N = A.shape
    x = cvx.Variable(N)
    constraints = [x >= 0]
    objective = cvx.Minimize(0.5 * cvx.square(cvx.norm(A*x - b, 2)))
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.SCS, verbose=True, use_indirect=True, max_iters=100000)
    return x.value

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
# X_train, y_train = load_svmlight_file("real-sim")
# X_train, y_train = load_svmlight_file("rcv1_train.binary")
# X_train, y_train = load_svmlight_file("news20.binary")
X_train, y_train = load_svmlight_file("webspam_wc_normalized_unigram.svm")

A_sparse = csr_matrix(X_train)
b = y_train
print('A.shape: ', A_sparse.shape)
print('A nnz: ', A_sparse.nnz)

# SCS
# print('SCS')
# start = pc()
# x = solve_scs(A_sparse, b)
# end = pc()
# xobj = 0.5*np.linalg.norm(A_sparse.dot(x) - b)**2
# print('x: ', x[:5])
# print('obj: ', xobj)
# print('secs: ', end-start)

# SBB
print('SBB')
start = pc()
x = sbb(A_sparse, b, np.zeros(A_sparse.shape[1]), maxit=5, M_inner=200)
end = pc()
xobj = 0.5*np.linalg.norm(A_sparse.dot(x) - b)**2
print('x: ', x[:5])
print('obj: ', xobj)
print('secs: ', end-start)

# SCIPY
f_wrap = lambda x: func(A_sparse, x, b)
jac_wrap = lambda x: calc_grad(A_sparse, x, b)

start = pc()
x = minimize(f_wrap, np.zeros(A_sparse.shape[1]), jac=jac_wrap,
                bounds=[(0, np.inf) for i in range(A_sparse.shape[1])],
                options={'disp': True, 'maxiter': 10}).x
end = pc()
xobj = 0.5*np.linalg.norm(A_sparse.dot(x) - b)**2
print('scipy l-bfgs-b: ', x[:5])
print('obj: ', xobj)
print('secs: ', end-start)
