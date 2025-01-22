import numpy as np

# system of equations
def f(x):
    F = np.zeros(2)
    F[0] = x[0]**2 + x[1]**2 - 1
    F[1] = x[0] - x[1]
    return F

def broyden(f, x0, tol=1e-8, max_iter=100):
    """
    inputs:
    - f: function defines the system of eqs
    - x0: initial guess
    return:
    - x: approx solution
    - i+1: number of iterations
    """
    x = np.array(x0, dtype=float) # convert x0 to numpy array
    n = len(x)
    # inital jacobian approximation
    B = np.eye(n)
    for i in range(max_iter):
        # evaluate at current guess
        F = f(x) 
        # solve B delta x = -F for update step
        delta = np.linalg.solve(B, -F) 
        x_new = x + delta # update
        if np.linalg.norm(delta, ord=2) < tol:
            return x_new, i+1
        # update jacobian with broyden's method
        F_new = f(x_new) # evaluate at new guess
        y = F_new - F
        B += np.outer((y - B @ delta), delta) / np.dot(delta, delta) # update B using broyden's formula
        x = x_new
    raise ValueError("fail to converge")

# initial guess
x0 = [2.0, 1.0]

# solve using broyden
sol_broyden, iter_broyden = broyden(f, x0)
print("Broyden Solution:", sol_broyden)
print("Iterations:", iter_broyden)
    