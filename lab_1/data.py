import numpy as np
import random

def sample_gmm_2d(K, C, N):
    X = np.zeros((K*N, 2), dtype=float)
    Y = np.zeros((K*N), dtype=float)
    for i in range(K):
        mean = np.random.uniform(-10, 10)
        sigma = np.random.uniform(0, 5)
        c_i = random.sample(range(C), 1)
        for j in range(N):
            X[i*N+j] = np.random.uniform(mean, sigma, 2)
            Y[i*N+j] = c_i[0]
    return (X, Y)
