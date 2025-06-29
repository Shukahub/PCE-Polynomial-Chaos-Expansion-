import numpy as np
from numpy.polynomial.polynomial import polyvander2d

# 用户参数（可改）
N_samples = 500
n_outputs = 78

# 输入采样范围（可改）
x1 = np.random.uniform(-1, 1, N_samples)
x2 = np.random.uniform(-1, 1, N_samples)

# 构造训练输入矩阵
X = np.vstack([x1, x2])  # shape (2, N)

# 定义目标真实函数（可修改）
def true_function(x1, x2, i):
    return np.sin(x1) + np.log(1 + x2**2) + i * x1 * x2

# 构造输出矩阵：Y[i, sample_id]
Y = np.zeros((n_outputs, N_samples))
for i in range(n_outputs):
    Y[i, :] = true_function(x1, x2, i)

# 多项式基向量构造: [1, x1, x2, x1^2, x1*x2, x2^2]
V = np.vstack([
    np.ones(N_samples),
    x1,
    x2,
    x1**2,
    x1 * x2,
    x2**2
]).T  # Shape: (N_samples, 6)

# 最小二乘拟合每个输出
coeff = np.linalg.lstsq(V, Y.T, rcond=None)[0]  # Shape: (6, 78)

# 输出 Fortran 格式
print("data coeff / &")
for i in range(n_outputs):
    line = ", ".join([f"{coeff[k,i]:.8f}d0" for k in range(6)])
    suffix = "," if i < n_outputs - 1 else " "
    print(f"  {line}{suffix} &")
print("/")
