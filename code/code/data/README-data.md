# Generating dataset

First, download the MFTC datset.

## How to generate train set and test set?
1. Preprocessing code can be found in cleaners.py.
2. Dataset for each domain can be split using generate_train_test_data() in generate_train_test_data.py (Set fract = 0.9).  Don't forget to modify file names throughout the function as you want.


## How to generate supervised dataset?
1. Run a generate_sup_duality_half() in generate_supervised_duality.py. 
2. Make sure output filenames are set as you like in generate_sup_duality_outside_foundation(), generate_sup_duality_within_foundation() and generate_sup_duality_half().  
3. Combine the two output files.

## How to generate unsupervised dataset?
1. Run generate_unsup() from geneerate_unsupervised.py by providing the appropriate filename. 

