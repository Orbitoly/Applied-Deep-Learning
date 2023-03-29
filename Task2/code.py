from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def target(x, A, b):
    return A.T @ A @ x - A.T @ b

def calc_error(x, A, b):
    return np.linalg.norm(A @ x - b) ** 2

def gradient_descent(A, x0, b, epsilon=0.1, delta=1e-5):

    errors = np.array([])
    x = x0
    errors = np.append(errors, calc_error(x, A, b))

    while 2 * (np.linalg.norm(target(x, A, b)) ** 2) > delta:
        x = x - 2 * epsilon * target(x, A, b)
        errors = np.append(errors, calc_error(x, A, b))

    return x, errors


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

A = diabetes_X
b = diabetes_y
# x0 = #np.array([0.5, 0.5])
x0 = np.ones(A.shape[1])
x, errors = gradient_descent(A, x0, b, epsilon=0.1, delta=1e-5)

#plot errors and show
#plt.plot(errors[:100])
#plt.xlabel('Iteration')
#plt.ylabel('Error')
#plt.title('Task 1: Error vs Iteration')

#plt.show()


def split_data(X, y, ratio = 0.2):
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-int(ratio * len(X))]]
    y_train = y[indices[:-int(ratio * len(X))]]
    X_test = X[indices[-int(ratio * len(X)):]]
    y_test = y[indices[-int(ratio * len(X)):]]
    return X_train, y_train, X_test, y_test

x0 = np.ones(A.shape[1])
X_train, y_train, X_test, y_test = split_data(diabetes_X, diabetes_y, ratio = 0.2)

x, errors = gradient_descent(X_train, x0, y_train, epsilon=0.1, delta=1e-5)

print(x)
test_errors = calc_error(x, X_test, y_test)
print(test_errors)