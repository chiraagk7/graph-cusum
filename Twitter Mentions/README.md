# Overview

This experiment aims to detect changes in Twitter mentions for large companies. We use the realTweets/ dataset which can be found [here](https://github.com/numenta/NAB/tree/master/data).
The Twitter data must be downloaded in the local directory for the below files to run correctly. The ground truth change point is 2015-03-09 08:02:53. This is the first time point after 8 AM on 3/9/2015,
the day when Apple had a special event about Apple Watch, new Macbook, new iOS, etc.

# Files

- `dd_vs_rl.py` - saves the detection delay and false alarm rate for various threshold values and values of c. Corresponds to Fig. 2D ("param" case)
- `dd_vs_rl_no_param.py` - saves the detection delay and false alarm rate for various threshold values and values of c. Corresponds to Fig. 2D ("blind" case)
- `convert_to_csv.py` - utility function to save detection delay, false alarm rate as .csv
