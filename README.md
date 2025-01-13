# DML-Heston-DiffPCA
 
In this project, Differential Machine Learning (DML) was applied to price European Options, using the computationally efficient version of Heston model. In addition, Differential Principle Components Analysis (Diff-PCA) was designed to improve the performance.

Motivation:
Our goal is to train a feedforward neural network that takes in Heston parameters and prices a European option. However in other applications, the first derivatives (Delta) is often useful in hedging applications. Its computation is efficient by virtue of Automatic Differentiation, but it is not involved in the training. The original scheme of DML therefore takes into account the first order differentials, however it remains a theoretical question whether higher order differentials can be examined under the same principle. 

Our contribution is therefore to extend the training scheme to incorporate the second order differentials. Under this scheme, any higher order differentials (and any asset price models) can easily be designed using this scheme. We also recognize the exponential computational surge in this results and therefore introduce the Differential PCA, aimed to reduce the computational cost while maintaining a reasonable performance. 

In this repository, you will find:
- `data_generation.ipynb`: demonstrates the construction of the dataset. The original data are given by [Asridi] (https://github.com/asridi/DML-Calibration-Heston-Model). We here focus on the construction of second order differentials and Diff-PCA
- `model_training.ipynb`: establishes and trains different models:
    1.  the benchmark model (trained with no differentials)
    2.  Model trained with 1st order differentials
    3.  Model trained with both 1st and 2nd order differentials
    4.  Model trained with both 1st and 2nd order differentials, with Diff-PCA
    
- `model_testing.ipynb`: conducts a brief performance analysis on the models.

- `data` folder: contains Heston parameters, their differentials in csv files with `d2_` initials. They satisfy the Feller conditions and are provided from https://github.com/asridi/DML-Calibration-Heston-Model. The complete datset is in `dataset_100K_feller.csv`. In this demonstration, we only considered 10K datapoints. The data and their differentials (1st and 2nd order) are contained in `feller_d2.csv`.

- `model` folder: contains the trained models
- `results` folder: contains graphics and training records
