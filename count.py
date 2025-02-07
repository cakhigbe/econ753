import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from scipy.special import factorial

# poisson negative log-likelihood
def poissonllh(beta, x, naffairs):
    """
    inputs:
    - beta : initial guess
    - x: characteristics
    - naffairs: y - number of affairs in the past year
    return:
    - llh
    """
    xlambda = np.exp(x @ beta) # dot product
    llh = np.sum(naffairs * np.log(xlambda) - xlambda - np.log(factorial(naffairs)))
    return -llh # negative for minimization

# gradient
def poissongrad(beta, x, naffairs):
    """
    inputs:
    - beta : initial guess
    - x: characteristics
    - naffairs: y - number of affairs in the past year
    return:
    - grad
    """
    xlambda = np.exp(x @ beta)
    grad = x.T @ (naffairs - xlambda) # d lnL / d beta
    return -grad

# BHHH algorithm
def bhhh(beta, x, naffairs, tol=1e-6, max_iter=1000):
    """
    inputs:
    - x: characteristics
    - naffairs: y - number of affairs in the past year
    - beta: initial guess
    - max_iter: maximum number of iterations
    - tol: convergence tolerance
    return:
    - beta_new: estimated parameters
    - i: number of iterations
    - hessian_init: initial hessian approx
    - hessian_fin: hessian at est params
    """
    beta_new = beta
    hessian_init =x.T @ np.diag(np.exp(x @ beta)) @ x
    for i in range(max_iter):
        xlambda = np.exp(x @ beta)
        # calculate score function
        residuals = naffairs - xlambda
        score = x * residuals[:, np.newaxis] 
        grad = np.sum(score, axis=0) # gradient = sum of scores
        # approximate hessian using bhhh
        info_matrix = score.T @ score
        update = np.linalg.solve(info_matrix, grad) # d = -I(beta) * gradllh
        # update beta
        beta_new += update
        if np.linalg.norm(update) < tol:
            hessian_fin = info_matrix
            return beta_new, i, hessian_init, hessian_fin
    print("fail to converge")
    return beta_new, i
    
# NLLS optimization
def nlls(beta, x, naffairs, tol=1e-6, max_iter=1000):
    """
    inputs:
    - x: characteristics
    - naffairs: y - number of affairs in the past year
    - beta: initial guess
    - max_iter: maximum number of iterations
    - tol: convergence tolerance
    return:
    - beta_new: estimated parameters
    - i: number of iterations
    """
    beta_new = beta
    for i in range(max_iter):
        xlambda = np.exp(x @ beta)
        # calculate score function
        residuals = naffairs - xlambda
        J = x * xlambda[:, np.newaxis] # jacobian
        grad = J.T @ residuals # grad = J.T @ (y - xlambda)
        # approximate hessian using nlls
        H = J.T @ J
        update = np.linalg.solve(H, grad)
        # update beta
        beta_new += update
        if np.linalg.norm(update) < tol:
            return beta_new, i
    print("fail to converge")
    return beta_new, i

# import dataset
data = pd.read_csv('psychtoday.csv')
naffairs = data.iloc[:,0].values # y is the count of affairs in the past year
x = data.iloc[:,1:].values # all other

# initial guess
beta = np.zeros(x.shape[1])

# estimate using BFGS
bfgs_num_start = time.perf_counter()
result_num = minimize(poissonllh, beta, args=(x, naffairs), method='BFGS')
bfgs_num_time = time.perf_counter() - bfgs_num_start
beta_bfgs_num = result_num.x

# estimate using BFGS with analytical gradient
bfgs_ana_start = time.perf_counter()
result_ana = minimize(poissonllh, beta, args=(x, naffairs), method='BFGS', jac=poissongrad)
bfgs_ana_time = time.perf_counter() - bfgs_ana_start
beta_bfgs_ana = result_ana.x

# estimate using BFGS with analytical gradient
nm_start = time.perf_counter()
result_nm = minimize(poissonllh, beta, args=(x, naffairs), method='Nelder-Mead')
nm_time = time.perf_counter() - nm_start
beta_nm = result_nm.x

# estimate using BHHH
bhhh_start = time.perf_counter()
beta_bhhh, bhhh_iter, bhhh_hessian_init, bhhh_hessian_fin = bhhh(beta, x, naffairs)
bhhh_time = time.perf_counter() - bhhh_start

# estimate using NLLS
nlls_start = time.perf_counter()
beta_nlls, nlls_iter = nlls(beta, x, naffairs)
nlls_time = time.perf_counter() - nlls_start

# compile results
methods = ['BFGS Numerical', 'BFGS Analytical', 'Nelder-Mead', 'BHHH', 'NLLS']
betas = [beta_bfgs_num, beta_bfgs_ana, beta_nm, beta_bhhh, beta_nlls]
iterations = [result_num.nit, result_ana.nit, result_nm.nit, bhhh_iter, nlls_iter]
func_evals = [result_num.nfev, result_ana.nfev, result_nm.nfev, bhhh_iter, nlls_iter]
times = [bfgs_num_time, bfgs_ana_time, nm_time, bhhh_time, nlls_time]

# display results
count_table = []
parameters = ['Constant', 'Age', 'Years Married', 'Religiousness', 'Occupation', 'Marriage Rating']
# Constant | Age |Years married | Religiousness (1-5) | Occupation (1-7) | Marriage Rating (1-5)
for m, b, i, f, t in zip(methods, beta, iterations, func_evals, times):
    row = {
        'Method': m,
        'Iterations': i,
        'Function Evaluations': f,
        'Time': t
    }

    for param, val in zip(parameters, beta):
        row[param] = val
    
    count_table.append(row)

results = pd.DataFrame(count_table)
print(results)

# calculate eigenvalues for bhhh hessian
eigen_init = np.linalg.eigvals(bhhh_hessian_init)
eigen_fin = np.linalg.eigvals(bhhh_hessian_fin)

print("\n Eigenvalues Initial Hessian Approximation:", eigen_init)
print("\n Eigenvalues Final Hessian Approximation:", eigen_fin)
print(result_num.nfev)
