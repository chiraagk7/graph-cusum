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


def edd_vs_arl_no_param(trials, c):
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
        p = 2*math.log(n)/n

        G0 = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        A0 = nx.to_numpy_matrix(G0)

        ### POST-CHANGE GRAPH: SBM (2 COMMUNITIES) ###

        m_param = 1
        G1 = nx.generators.random_graphs.barabasi_albert_graph(n, m_param)
        A1 = nx.to_numpy_matrix(G1)
        
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
        W = np.random.multivariate_normal(np.zeros(n), np.eye(n), m).T
        Y_arl = np.dot(H0, W) # no change only (for ARL calculation)
        Y_edd = np.dot(H1, W) # change only (for EDD calculation)


        ### DETECTION SETUP ###

        k = 1 # Number of eigenvectors to consider

        wA, U = np.linalg.eigh(A0)
        U0 = U[:, -k:] # Initial subspace

        cusum_arl = [0]
        for i in range(m):
            x = Y_arl[:, i] / np.linalg.norm(Y_arl[:, i]) # Current signal, normalized
            Lt = subspace_dist(U0, x) - c
            cusum_arl.append(max(0, cusum_arl[-1] + Lt))

        cusum_edd = [0]
        for i in range(m):
            x = np.atleast_2d(Y_edd[:, i] / np.linalg.norm(Y_edd[:, i])) # Current signal, normalized
            Lt = subspace_dist(U0, x) - c
            cusum_edd.append(max(0, cusum_edd[-1] + Lt))


        thresholds = np.linspace(0.0, 1, 20)
        edd = []
        arl = []

        for threshold in thresholds:
            # ARL calculation
            rl = next((i for i in range(m) if cusum_arl[i] > threshold), m) # first index where cusum > threshold
            arl.append(rl)
            # EDD calculation
            dd = next((i for i in range(m) if cusum_edd[i] > threshold), m) # first index where cusum > threshold
            edd.append(dd)
        
        print(arl)
        print(edd)
        edd_avg += edd
        arl_avg += arl

    arl_avg /= trials
    edd_avg /= trials
    return arl_avg, edd_avg


if __name__ == "__main__":
    cs = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

    for c in cs:
        arl, edd = edd_vs_arl_no_param(trials=20, c=c)
        np.save("nc_blind_arl_" + str(c), arl)
        np.save("nc_blind_edd_" + str(c), edd)
    
    # plt.plot(arl, edd)
    # plt.xscale('log')
    # plt.xlabel("Average Run Length")
    # plt.ylabel("Expected Detection Delay")
    # plt.title("EDD vs ARL for Node Centrality")
    # plt.show()





