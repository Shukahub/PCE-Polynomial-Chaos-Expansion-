import numpy as np
import chaospy as cp

# Step 1: Generate training data
N_samples = 100
X = np.random.uniform(-1, 1, (2, N_samples))  # 2D inputs in [-1,1]

# Define true function: outputs = 78-dim vector
Y = np.zeros((78, N_samples))
for i in range(78):
    Y[i, :] = np.sin(X[0]) + np.log(1 + X[1]**2) + i * X[0] * X[1]

# Step 2: Fit PCE (2nd order, 6 terms)
dist = cp.Iid(cp.Uniform(-1, 1), 2)
poly = cp.orth_ttr(2, dist)  # total degree 2
coeff = cp.fit_regression(poly, X, Y)

# Step 3: Print Fortran-style coefficients
print("! PCE coefficients for Fortran")
for i in range(78):
    line = ", ".join([f"{coeff[k,i]:.8f}d0" for k in range(6)])
    print("  " + line + ", &")
