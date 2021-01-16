import numpy as np
import matplotlib.pyplot as plt 
import csv

def save_csv(c, fa, dd):
    '''
    Save csv of data for different values of c, false alarm rates, detection delays
    Returns nothing
    '''

    data = [fa, dd]
    data = np.array(data).T
    
    file = open(f'tw_fa_dd_param_{c}.csv', 'w', newline ='')
    with file: 
        header = ['FA Rate',  'Detection Delay'] 
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == "__main__":
    # Example Usage

    # cs1 = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92]
    # for c in cs1:
    #     fa = np.load('SC_blind_fa_' + str(c) + '.npy')
    #     dd = np.load('SC_blind_dd_' + str(c) + '.npy')
    #     save_csv(c, fa, dd)

    # cs2 = [0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68]
    # for c in cs2:
    #     fa = np.load('SC_param_fa_' + str(c) + '.npy')
    #     dd = np.load('SC_param_dd_' + str(c) + '.npy')
    #     save_csv(c, fa, dd)
    pass

