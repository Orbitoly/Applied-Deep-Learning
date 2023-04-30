from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
#seed
np.random.seed(1)
def target(x, A, b):
    return A.T @ A @ x - A.T @ b

def calc_error(x, A, b):
    return np.linalg.norm(A @ x - b) ** 2

#load data
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
A = diabetes_X
b = diabetes_y

#------------------Part 1-------------------
def gradient_descent_1(A, x0, b, epsilon=0.1, delta=1e-5):
    errors = np.array([])
    x = x0
    errors = np.append(errors, calc_error(x, A, b))
    while 2 * (np.linalg.norm(target(x, A, b)) ** 2) > delta:
        x = x - 2 * epsilon * target(x, A, b)
        errors = np.append(errors, calc_error(x, A, b))
    return x, errors

def test_task1():
    x0 = np.ones(A.shape[1])
    x, errors = gradient_descent_1(A, x0, b, epsilon=0.1, delta=1e-5)
    
    # plot errors and show
    plt.plot(errors[:50])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Task 1: Error vs Iteration')
    plt.show()


#------------------Part 2-------------------
def gradient_descent_2(x0, A_train, b_train, A_test, b_test, epsilon=0.1, delta=1e-5):

    errors_train = np.array([])
    errors_test = np.array([])
    x = x0
    errors_train = np.append(errors_train, calc_error(x, A_train, b_train))
    errors_test = np.append(errors_test, calc_error(x, A_test, b_test))

    while 2 * (np.linalg.norm(target(x, A_train, b_train)) ** 2) > delta:
        x = x - 2 * epsilon * target(x, A_train, b_train)
        errors_train = np.append(errors_train, calc_error(x, A_train, b_train))
        errors_test = np.append(errors_test,calc_error(x, A_test, b_test))
    return x, errors_train, errors_test


def split_data(X, y, ratio = 0.2):
    X_train = X[:-int(ratio * len(X))]
    y_train = y[:-int(ratio * len(X))]
    X_test = X[-int(ratio * len(X)):]
    y_test = y[-int(ratio * len(X)):]
    return X_train, y_train, X_test, y_test

def test_task2():
    x0 = np.ones(A.shape[1])
    X_train, y_train, X_test, y_test = split_data(diabetes_X, diabetes_y, ratio = 0.2)
    x, error_train, error_test = gradient_descent_2(x0, X_train, y_train, X_test, y_test, epsilon=0.1, delta=1e-5)
    
    # plot errors and show
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(error_train[:50])
    ax1.set_title('Train Errors')
    ax2.plot(error_test[:50])
    ax2.set_title('Test Errors')
    fig.suptitle('Task 2: Errors Functions')
    plt.show()

#------------------Part 3-------------------
def split_data_random(X, y, ratio = 0.2):
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-int(ratio * len(X))]]
    y_train = y[indices[:-int(ratio * len(X))]]
    X_test = X[indices[-int(ratio * len(X)):]]
    y_test = y[indices[-int(ratio * len(X)):]]
    return X_train, y_train, X_test, y_test

def test_task3():
    x0 = np.ones(A.shape[1])
    X_train, y_train, X_test, y_test = split_data_random(diabetes_X, diabetes_y, ratio = 0.2)
    n = 10
    x, error_train, error_test = gradient_descent_2(x0, X_train, y_train, X_test, y_test, epsilon=0.1, delta=1e-5)
    error_train_avg = error_train
    error_test_avg = error_test

    #split the data randomly 10 times, and calculate the average
    for i in range(n-1):
        X_train, y_train, X_test, y_test = split_data_random(diabetes_X, diabetes_y, ratio=0.2)

        x, error_train, error_test = gradient_descent_2(x0, X_train, y_train, X_test, y_test, epsilon=0.1, delta=1e-5)
        error_train_avg = error_train_avg + error_train
        error_test_avg = error_test_avg + error_test

    error_train_avg = 1/n * error_train_avg
    error_test_avg = 1/n * error_test_avg

    # plot errors and show
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(error_train_avg[:50])
    ax1.set_title('Train Errors')
    ax2.plot(error_test_avg[:50])
    ax2.set_title('Test Errors')
    fig.suptitle('Task 3: Errors Functions')
    plt.show()

test_task3()