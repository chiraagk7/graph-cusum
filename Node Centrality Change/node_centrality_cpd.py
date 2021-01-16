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

def post_change_subspace(alpha, n, k):
    '''
    Inputs: 
        alpha - index of node with highest node-centrality
        n - num of nodes
        k - dimension of subspace to use
    Outputs: Expected leading eigenvector
    '''
    eig = 0.0*np.ones((n, 1))
    eig[alpha, 0] += 1
    return eig

    
def save_gamma_hats_csv(param_hats):
    '''
    Save csv of estimate post-change parameter values over time
    Returns nothing
    '''
    data = np.array([range(len(param_hats)), param_hats]).T
    file = open('nc_gamma_hats.csv', 'w', newline ='')
    with file: 
        header = ['# Index', 'Gamma_hat'] 
        writer = csv.writer(file) 
        writer.writerow(header)
        writer.writerows(data)
    
def save_cusum_csv(cusum, cs):
    '''
    Save csv of cusum value over time for different values of c
    Returns nothing
    '''

    data = [range(len(cusum[cs[0]]))]
    for c in cs:
        data.append(cusum[c])
    data = np.array(data).T
    headers = [str(c) for c in cs]

    file = open('nc_cusum.csv', 'w', newline ='')
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
    p = 2*math.log(n)/n

    G0 = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    A0 = nx.to_numpy_matrix(G0)

    ### POST-CHANGE GRAPH: SBM (2 COMMUNITIES) ###
    m_param = 1
    G1 = nx.generators.random_graphs.barabasi_albert_graph(n, m_param)
    ec = nx.eigenvector_centrality(G1)
    centrality = [ec[i] for i in range(n)]
    max_centrality_node = np.argmax(centrality)

    A1_old = nx.to_numpy_matrix(G1)

    perm = np.random.permutation(100)
    alpha = np.where(perm == max_centrality_node)[0][0]
    print(alpha)
    A1 = np.zeros(A1_old.shape)
    for i in range(n):
        for j in range(i+1):
            A1[i, j] = A1_old[perm[i], perm[j]]
            A1[j, i] = A1_old[perm[i], perm[j]]

    # plt.imshow(A0)
    # plt.title("Initial Adjacency Matrix")
    # plt.show()
    # plt.imshow(A1)
    # plt.title("Post-Change Adjacency Matrix")
    # plt.show()


    # ### GENERATE SIGNALS ###

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

    # ### DETECTION SETUP ###

    k = 1 # Number of eigenvectors to consider
    cusum = {} # dictionary from c -> cusum score over time
    cs = [0.0, 0.05, 0.10] # values of c to try
    window_size = 50 # For covariance estimates

    wA, U = np.linalg.eigh(A0)
    U0 = U[:, -k:] # Initial subspace

    alphas = np.array(range(100)) # Possible parameter values (alpha = index of node with highest eigenvector centrality)
    U1 = {a:post_change_subspace(a, n, k) for a in alphas} # Dict alpha: subspace(alpha)




    # ### DETECTION ###
    alpha_hats = []
    alphas_saved = False

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
    # plt.axvline(x=t_cp, color='r')
    # plt.show()

    save_gamma_hats_csv(alpha_hats)

    # for c in cs:
    #     plt.plot(cusum[c])

    # plt.legend(cs)
    # plt.xlabel('Index')
    # plt.ylabel('CUSUM Score')
    # plt.title('Running CUSUM Statistic')
    # plt.axvline(x=t_cp, color='r')
    # plt.show()

    save_cusum_csv(cusum, cs)






