import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.linalg import subspace_angles
import csv

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

def save_gamma_hats_csv(param_hats):
    '''
    Saves csv of estimate post-change parameter values over time

    Inputs: 
        param_hats - the estimated parameter for the post-change subspace
    Returns nothing
    '''
    data = np.array([range(len(param_hats)), param_hats]).T
    file = open('ef_gamma_hats.csv', 'w', newline ='')
    with file: 
        header = ['# Index', 'Gamma_hat'] 
        writer = csv.writer(file) 
        writer.writerow(header)
        writer.writerows(data)
    
def save_cusum_csv(cusum, cs):
    '''
    Save csv of cusum value over time for different values of c

    Inputs:
        cusum - dictionary of lists of cusum values over time (keys are each c in cs)
        cs - list of correction parameters, corresponding to each item in "cusum"
    Returns nothing
    '''

    data = [range(len(cusum[cs[0]]))]
    for c in cs:
        data.append(cusum[c])
    data = np.array(data).T
    
    headers = [str(c) for c in cs] # convert cs to string to use as headers

    file = open('ef_cusum.csv', 'w', newline ='')
    with file: 
        header = ['# Index'] + headers 
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def save_adjacency_csv(A, name):
    '''
    Save csv of adjacency matrix with given name (A0 or A1)
    Returns nothing
    '''
    np.savetxt('ef_' + name + '.csv', A, delimiter=",")

if __name__ == "__main__":
    ### INITIAL GRAPH: ERDOS-RENYI(n, p) ###

    n = 100 # number of nodes
    p0 = 2*math.log(n)/n # connection probability

    G0 = nx.generators.random_graphs.erdos_renyi_graph(n, p0)
    A0 = nx.to_numpy_matrix(G0)

    ### POST-CHANGE GRAPH: SBM (2 COMMUNITIES) ###

    p1 = 5*p0 # prob of edge within the emerging faction (community emerges on top of E-R model)
    alpha = 0.25 # proportion of nodes in C1 (range from 0 to 1)

    size = round(n*alpha) # Size of faction
    A1 = A0.copy()
    for i in range(size):
        for j in range(i):
            if random.random() < p1: # reflip with 5x higher connection probability
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

    save_adjacency_csv(A0, 'A0')
    save_adjacency_csv(A1, 'A1')

    ### GENERATE SIGNALS ###

    # Define graph filter
    poly = lambda x: x**2
    H0 = poly(A0)
    H1 = poly(A1)

    # Signal parameters
    m = 1000 # number of signals to generate
    t_cp = 600 # index of change point

    # Generate signals
    W = np.random.multivariate_normal(np.zeros(n), np.eye(n), m).T
    Y0 = np.dot(H0, W[:, 0:t_cp])
    Y1 = np.dot(H1, W[:, t_cp:m+1])
    Y = np.concatenate((Y0, Y1), axis=1)

    ### DETECTION SETUP ###

    k = 2 # Number of eigenvectors to consider
    cusum = {} # dictionary from c -> cusum score over time
    cs = [0.0, 0.02, 0.04] # values of c to try
    window_size = 50 # For covariance estimates

    wA, U = np.linalg.eigh(A0)
    U0 = U[:, -k:] # Initial subspace

    alphas = 0.01 * np.array(range(10, 91)) # Possible parameter values (discretized range from 0.25-0.75)
    U1 = {a:post_change_subspace(a, k, p0, p1, n) for a in alphas} # Dict alpha: subspace(alpha)




    ### DETECTION ###

    alpha_hats = []
    alphas_saved = False # to avoid duplicate saving alpha_hat estimates 

    for c in cs:
        cusum[c] = [0]
        for i in range(m - window_size + 1):
            window = Y[:, i:i+window_size] # Signals in the current window
            C_hat = (1/window_size) * np.dot(window, window.T) # Empirical covariance
            wCs, Us = np.linalg.eigh(C_hat)
            U_hat = Us[:, -k:] # observed subspace
            alpha_hat = max(alphas, key=lambda a: np.linalg.norm(U1[a].T @ U_hat)) # Parameter estimate
            if not alphas_saved:
                alpha_hats.append(alpha_hat)
            Lt = subspace_dist(U0, U1[alpha_hat]) - subspace_dist(U_hat, U1[alpha_hat]) - c
            cusum[c].append(max(0, cusum[c][-1] + Lt))
        alphas_saved = True


    # plt.plot(alpha_hats)
    # plt.xlabel('Index')
    # plt.ylabel('Predicted Value')
    # plt.title('Predicted Parameter Value Over Time')
    # plt.axvline(x=t_cp-window_size, color='r')
    # plt.show()

    save_gamma_hats_csv(alpha_hats)

    # for c in cs:
    #     plt.plot(cusum[c])

    # plt.legend(cs)
    # plt.xlabel('Index')
    # plt.ylabel('CUSUM Score')
    # plt.title('Running CUSUM Statistic')
    # plt.axvline(x=t_cp-window_size, color='r')
    # plt.show()

    save_cusum_csv(cusum, cs)





    


