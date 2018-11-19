import cvxpy as cp
import numpy as np

N = 4096
n = 2048
k = 1000

theta = np.zeros(shape=(N, 1))
sparse_locs = np.random.randint(0, N, size=(k, ))
theta[sparse_locs] = 1
print(np.linalg.norm(theta))
phi = np.identity(N)

x = np.matmul(phi, theta)
print(np.linalg.norm(x))
A = np.random.normal(loc=0, scale=1, size=(n, N))
y = np.matmul(A, x)
y = np.reshape(y, (n, 1))


x_bar = cp.Variable((N, 1))

objective = cp.Minimize(cp.norm(x_bar, p=1))
constraints = [y - A * x_bar <= 0.1, y - A * x_bar >= -.1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(np.linalg.norm(x - x_bar.value))

import matplotlib.pyplot as plt

plt.plot(x)
plt.plot(x_bar.value)
plt.show()
