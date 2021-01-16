# Overview

This experiment aims to detect the emergence of a highly-connected community which develops on top of an underlying Erdős–Rényi graph. 

# Files

- `emerging_faction.py`- saves a .csv of the CUSUM statistic value for each tested value of the correction paramter c. Parameters such as number of nodes, pre- and post-change connection probabilities,
signal-generating filter, and window size can be varied.
- `edd_vs_arl.py` - saves the average run length under pre-change and post-change conditions for various threshold values and values of c. Corresponds to Fig. 2B ("param" case)
- `edd_vs_arl_no_param.py` - saves the average run length under pre-change and post-change conditions for various threshold values and values of c. Corresponds to Fig. 2B ("blind" case)
- `convert_to_csv.py` - utility function to save ARL as .csv
