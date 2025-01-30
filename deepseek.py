import numpy as np

# Define the system of equations
def f(x):
    return np.array([
        x[0]**2 + x[1]**2 - 1,  # F[1]
        x[0] - x[1]             # F[2]
    ])

# Broyden's method
def broyden(f, x0, tol=1e-10, max_iter=100):
    n = len(x0)
    x = x0.copy()
    F = f(x)
    B = np.eye(n)  # Initial approximation of the Jacobian (identity matrix)

    for k in range(max_iter):
        # Check for convergence
        if np.linalg.norm(F) < tol:
            print(f"Converged in {k} iterations.")
            return x

        # Solve for the update step: B * dx = -F
        dx = np.linalg.solve(B, -F)

        # Update x
        x_new = x + dx

        # Compute new F
        F_new = f(x_new)

        # Update the approximation of the Jacobian (Broyden's update)
        y = F_new - F
        B = B + np.outer((y - B @ dx), dx) / (dx @ dx)

        # Update x and F for the next iteration
        x = x_new
        F = F_new

    print("Maximum iterations reached. Solution may not have converged.")
    return x

# Initial guess
x0 = np.array([2.0, 1.0])

# Solve the system
solution = broyden(f, x0)

print("Solution:", solution)