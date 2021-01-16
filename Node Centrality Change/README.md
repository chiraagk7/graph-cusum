# Overview

This experiment aims to detect a change in node centrality from a "flat" distribution (i.e. an Erdős–Rényi graph) to a power-law distribution (a Barabási-Albert graph)

# Files

- `node_centrality_cpd.py`- saves a .csv of the CUSUM statistic value for each tested value of the correction paramter c. Parameters such as number of nodes, pre-change connection probability, B-A graph parameters, and
signal-generating filter can be varied. Corresponds to Fig. 2A.
- `edd_vs_arl.py` - saves the average run length under pre-change and post-change conditions for various threshold values and values of c. Corresponds to Fig. 2B ("param" case)
- `edd_vs_arl_no_param.py` - saves the average run length under pre-change and post-change conditions for various threshold values and values of c. Corresponds to Fig. 2B ("blind" case)
- `convert_to_csv.py` - utility function to save ARL as .csv (convert from .npy)
