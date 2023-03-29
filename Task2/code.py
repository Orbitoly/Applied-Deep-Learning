from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def target(x, A, b):
    return A.T @ A @ x - A.T @ b


def gradient_descent(A, x0, b, epsilon=0.1, delta=1e-5):
    calc_error = lambda x: np.linalg.norm(A @ x - b) ** 2

    errors = np.array([])
    x = x0
    errors = np.append(errors, calc_error(x))

    while 2 * (np.linalg.norm(target(x, A, b)) ** 2) > delta:
        print(x)
        x = x - 2 * epsilon * target(x, A, b)
        errors = np.append(errors, calc_error(x))

    return x, errors


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# split randomly using numpy

A = diabetes_X#np.eye(2)
b = diabetes_y#np.array([1, 1])
# x0 = #np.array([0.5, 0.5])
x0 = np.ones(A.shape[1])
x, errors = gradient_descent(A, x0, b, epsilon=0.1, delta=1e-5)
print(x)
#plot errors and show
plt.plot(errors[:100])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Task 1: Error vs Iteration')

plt.show()

#split data 80 20


# indices = np.random.permutation(len(diabetes_X))
# diabetes_X_train = diabetes_X[indices[:-20]]
# diabetes_y_train = diabetes_y[indices[:-20]]
# diabetes_X_test = diabetes_X[indices[-20:]]
# diabetes_y_test = diabetes_y[indices[-20:]]
# add intercept
