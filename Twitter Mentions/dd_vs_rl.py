import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.covariance import MinCovDet

### HELPER FUNCTIONS ###

def subspace_dist(A, B):
    '''
    Inputs: 
        A, B - Two subspaces (columns of each matrix are basis for the subspace)
    Outputs: The distance between the two subspaces, defined as the Frobenius norm of the sine of principal angles
    '''
    return np.linalg.norm(np.sin(subspace_angles(A, B)))

def post_change_subspace(alpha, n, k):
    '''
    Inputs: 
        alpha - index of node with highest node-centrality
        n- num of nodes
        k - dimension of subspace to use
    Outputs: Expected leading eigenvector
    '''
    eig = 0.0*np.ones((n, 1))
    eig[alpha, 0] += 1
    return eig


### Load Data ###

companies = ['AAPL', 'AMZN', 'CRM', 'CVS', 'FB', 'GOOG', 'IBM', 'KO', 'PFE', 'UPS']
n = len(companies) # number of nodes
min_shape = 15831 # minimum length of all files (used this much data from each node)
Y = np.zeros((n, min_shape))
timestamps = None # to store timestamps corresponding to each index

for i in range(n):
    company = companies[i]
    filename = f"Twitter_volume_{company}.csv"
    df = pd.read_csv(filename, sep=',')
    df = df[:min_shape]
    Y[i, :] = df['value'].to_numpy()
    if timestamps is None:
        timestamps = df['timestamp'].to_numpy()

### Split data and get nominal covariance estimate ###

day = 288 # num of data points per day
Y_train = Y[:, :4*day] # train on first 4 days of data
Y_test = Y[:, 4*day:14*day] # test on next 10 days

for i in range(10):
    Y_test[i, :] = Y_test[i, :] - np.mean(Y_test[i, :])

timestamps_train = timestamps[:4*day]
timestamps_test = timestamps[4*day:14*day]

# initial_cov = MinCovDet(random_state=0).fit(Y_train.T).covariance_

initial_cov = 1/Y_train.shape[0] * Y_train @ Y_train.T # initial prediction of graph signal covariance

### Setup ###

m = Y_test.shape[1]
k = 1 # Number of eigenvectors to consider
window_size = 12*3 # 3 hour of data
wA, U = np.linalg.eigh(initial_cov)
U0 = U[:, -k:] # Initial subspace

t_cp = 1756 + (12*8) # Apple special event 2015-03-09 08:02:53

alphas = np.array(range(n)) # Possible parameter values (alpha = index of node with highest eigenvector centrality)
U1 = {a:post_change_subspace(a, n, k) for a in alphas} # Dict mapping alpha -> subspace(alpha)

cs = [0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68]
for c in cs:
    print(c)
    cusum = [0]
    for i in range(m - window_size + 1):
        window = Y_test[:, i:i+window_size] # Signals in the current window
        C_hat = (1/window_size) * np.dot(window, window.T) # Empirical covariance
        wCs, Us = np.linalg.eigh(C_hat)
        U_hat = Us[:, -k:] # observed subspace
        alpha_hat = max(alphas, key=lambda a: np.linalg.norm(U1[a].T @ U_hat)) # Parameter estimate
        Lt = subspace_dist(U0, U1[alpha_hat]) - subspace_dist(U_hat, U1[alpha_hat]) - c
        cusum.append(max(0, cusum[-1] + Lt))

    rl = []
    dd = []
    thresholds = np.linspace(0, 35, 50)
    for threshold in thresholds:
        # ARL calculation
        false_alarms = len([i for i in range(t_cp) if cusum[i] > threshold])
        rl.append(false_alarms / t_cp)
        # rl.append(next((i for i in range(t_cp) if cusum[i] > threshold), t_cp)) # first index where cusum > threshold
        # EDD calculation
        first_i = next((i for i in range(t_cp, m - window_size + 1) if cusum[i] > threshold), m) # first index where cusum > threshold
        dd.append(first_i - t_cp)

    # plt.plot(rl, dd)
    # plt.title("Detection delay vs False Alarm Rate - Our Model")
    # plt.ylabel("Detection delay")
    # plt.xlabel("False Alarm Rate")
    # plt.show()

    np.save("tw_param_fa_" + str(c), rl)
    np.save("tw_param_dd_" + str(c), dd)