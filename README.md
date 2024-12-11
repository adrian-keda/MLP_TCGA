# MLP_TCGA

This repository has the code used for the hyperparameter tuning and development of a Multilayer Perceptron with one (see corresponding code in **1Layer** folder) and two hidden layers (see **2Layer** folder).

Each folder has a snakefile, a python script and a config file used for the hyperparameter tuning. The config file is used to define which chunks from the random selections file ( *Genes.20000.combination.for.ML.set.tsv.gz*  ) are used to perform the grid search. In case of 1 layer MLP, there is also a file for developing the model with the best hyperparameters. Currently, 2 layer MLP is in development.