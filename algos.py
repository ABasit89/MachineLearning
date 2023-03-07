import numpy as np
# single step perceptron


def single_step_perceptron(feature_vector, label, theta, theta_0):
    if label*(np.dot(feature_vector, theta)+theta_0) <= 0:
        theta = theta + label*feature_vector
        theta_0 = theta_0 + label
    return (theta, theta_0)


def perceptron(feature_matrix, label_matrix, T):
    theta = np.zeros(shape=feature_matrix.shape[1])
    theta_0 = 0
    for t in range(T):
        for i in range(feature_matrix.shape[0]):
            theta, theta_0 = single_step_perceptron(feature_matrix[i], label_matrix[i], theta, theta_0)
    return (theta, theta_0)