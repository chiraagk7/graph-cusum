import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.linalg import subspace_angles

### HELPER FUNCTIONS ###

def subspace_dist(A, B):
    '''
    Inputs: 
        A, B - Two subspaces (columns of each matrix are basis for the subspace)
    Outputs: The distance between the two subspaces, defined as the Frobenius norm of the sine of principal angles
    '''
    return np.linalg.norm(np.sin(subspace_angles(A, B)))

def post_change_subspace(alpha, k, p0, p1, n):
    '''
    Inputs: 
        alpha - parameter for proportion of nodes in C1, 
        k - dimension of subspace
        p0, p1 - probabilities for edges within/between communities
        n - num of nodes
    Outputs: Leading k-dimensional subspace of expected post-change graph
    '''
    # Construct expected adjacency matrix
    n1 = round(n*alpha) # number of nodes in C1
    A_expected = p0*np.ones((n, n))
    A_expected[:n1, :n1] = p1 # Expected value for the upper left block is p1, rest are p0

    # Compute leading subspace
    wA, U = np.linalg.eigh(A_expected)
    return U[:, -k:]


def edd_vs_arl_param(trials, c):
    '''
    Input: 
        trials - Number of trials to run
        c - correction parameter to use
    Output: edd and arl vectors averaged over the number of trials
    '''
    edd_avg = np.zeros(20)
    arl_avg = np.zeros(20)

    for t in range(trials):

        print("Starting Trial " + str(t))

        ### INITIAL GRAPH: ERDOS-RENYI(n, p) ###

        n = 100 # number of nodes
        p0 = 2*math.log(n)/n # initial connection probability

        G0 = nx.generators.random_graphs.erdos_renyi_graph(n, p0)
        A0 = nx.to_numpy_matrix(G0)

        ### POST-CHANGE GRAPH: SBM (2 COMMUNITIES) ###

        p1 = 5*p0 # prob of edge within the emerging faction (community emerges on top of underlying E-R model)
        alpha = 0.15 # proportion of nodes in C1 (range from 0 to 1)

        size = round(n*alpha) # Size of faction
        A1 = A0.copy()
        for i in range(size):
            for j in range(i):
                if random.random() < p1:
                    A1[i, j] = 1
                    A1[j, i] = 1
                else:
                    A1[i, j] = 0
                    A1[j, i] = 0

        # plt.imshow(A0)
        # plt.title("Initial Adjacency Matrix")
        # plt.show()
        # plt.imshow(A1)
        # plt.title("Post-Change Adjacency Matrix")
        # plt.show()

        ### GENERATE SIGNALS ###

        # Define graph filter
        poly = lambda x: x**2
        H0 = poly(A0)
        H1 = poly(A1)

        # Signal parameters
        m = 10000 # number of signals to generate

        # Generate signals
        W = np.random.multivariate_normal(np.zeros(n), np.eye(n), m).T # white noise
        Y_arl = np.dot(H0, W) # pre-change only (for ARL calculation)
        Y_edd = np.dot(H1, W) # post-change only (for EDD calculation)


        ### DETECTION SETUP ###

        k = 2 # Number of eigenvectors to consider
        window_size = 25 # For covariance estimates

        wA, U = np.linalg.eigh(A0)
        U0 = U[:, -k:] # Initial subspace

        alphas = 0.01 * np.array(range(10, 91)) # Possible parameter values (discretized range from 0.25-0.75)
        U1 = {a:post_change_subspace(a, k, p0, p1, n) for a in alphas} # Dict alpha: subspace(alpha)

        cusum_arl = [0]
        for i in range(m - window_size + 1):
            window = Y_arl[:, i:i+window_size] # Signals in the current window
            C_hat = (1/window_size) * np.dot(window, window.T) # Empirical covariance
            wCs, Us = np.linalg.eigh(C_hat)
            U_hat = Us[:, -k:] # observed subspace
            alpha_hat = max(alphas, key=lambda a: np.linalg.norm(U1[a].T @ U_hat)) # Parameter estimate
            Lt = subspace_dist(U0, U1[alpha_hat]) - subspace_dist(U_hat, U1[alpha_hat]) - c
            cusum_arl.append(max(0, cusum_arl[-1] + Lt))

        cusum_edd = [0]
        for i in range(m - window_size + 1):
            window = Y_edd[:, i:i+window_size] # Signals in the current window
            C_hat = (1/window_size) * np.dot(window, window.T) # Empirical covariance
            wCs, Us = np.linalg.eigh(C_hat)
            U_hat = Us[:, -k:] # observed subspace
            alpha_hat = max(alphas, key=lambda a: np.linalg.norm(U1[a].T @ U_hat)) # Parameter estimate
            Lt = subspace_dist(U0, U1[alpha_hat]) - subspace_dist(U_hat, U1[alpha_hat]) - c
            cusum_edd.append(max(0, cusum_edd[-1] + Lt))


        thresholds = np.linspace(0.0, 1.5, 20)
        edd = []
        arl = []

        for threshold in thresholds:
            # ARL calculation
            rl = next((i for i in range(m-window_size+1) if cusum_arl[i] > threshold), m) # first index where cusum > threshold
            arl.append(rl)
            # EDD calculation
            dd = next((i for i in range(m-window_size+1) if cusum_edd[i] > threshold), m) # first index where cusum > threshold
            edd.append(dd)
        
        print(arl)
        print(edd)
        edd_avg += edd
        arl_avg += arl

    arl_avg /= trials
    edd_avg /= trials
    return arl_avg, edd_avg


for c in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]:
    print("c = " + str(c))
    arl, edd = edd_vs_arl_param(trials=25, c=c)
    np.save("ef_param_arl_" + str(c), arl)
    np.save("ef_param_edd_" + str(c), edd)



# plt.plot(arl, edd)
# plt.xscale('log')
# plt.xlabel("Average Run Length")
# plt.ylabel("Expected Detection Delay")
# plt.title("EDD vs ARL for Emerging Faction")
# plt.show()