import numpy as np
import matplotlib.pyplot as plt 
import csv

def save_csv(c, edd, arl):
    '''
    Save csv of data for different values of c
    Returns nothing
    '''

    data = [arl, edd]
    data = np.array(data).T
    
    file = open(f'nc_arl_edd_param_{c}.csv', 'w', newline ='') # change name to "blind" as needed
    with file: 
        header = ['ARL',  'EDD'] 
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    ## Example Usage

    # cs1 = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
    # for c in cs1:
    #     arl = np.load('no_param_arl_' + str(c) + '.npy')
    #     edd = np.load('no_param_edd_' + str(c) + '.npy')
    #     save_csv(c, edd, arl)

    # cs2 = [0.03, 0.04, 0.05, 0.06, 0.07]
    # for c in cs2:
    #     arl = np.load('param_arl_' + str(c) + '.npy')
    #     edd = np.load('param_edd_' + str(c) + '.npy')
    #     save_csv(c, edd, arl)
    pass

