# Metric-Learning-Strategies-for-the-Classification-of-T-Tauri-Stars
## Files:  
-   README.MD: This file
-   MMC_Less_Feats.py: The main file with the implemented code
-   FI_ind_18.txt: Example file with indices of features order by highest importance
-   allCaracTraining.dat: Example file with data to run the algorithm
-   Informe.txt: Example of the output file with summary of classifiers.  
    Structure:  
    {Classifier}_acc{#of features}    {Hyperparameter: value}    {accuracy_score} {cohen_kappa_score}    
    {recall_score}(1st feature)  
    {recall_score}(2nd feature)  
    ...  
    ..  
    .
## Directories:
-   img: Images of data visualized in 3 dimensions
-   vid: Animations of images on img (outputs of Main code)
-   classification: graphs of classification results (outputs of Main code)

## Instructions of Main code
-   This code is divided in 7 sections signaled with '#%%'.
-   1.Loading Data and preprocessing:
    Here the data file (eg. allCaracTraining.dat) and indices file (eg. I_ind_18.txt) are loaded. The data is preprocessed by dropping desired classes and feature and being scaled.
-   2.Functions:
    The functions to run the algorithm are defined.
-   3.Metric Optimization:
    The optimization for the metric is done for set of features, starting by defining the sets and followed by minimizing the objective function.
-   4.Applying Metrics:
    Obtained metrics are saved to an output file called 'metricas.txt' and then are applied to the data to transform it.
-   5.Clustering Accuracies calculation:
    This section simply takes the transformed data and calculates the clustering accuracy A_c for each set.
-   6.Plots:
    In this section, the original data and the transformed sets are plotted on their first 3 dimensions. An .mp4 animation like those found on directory 'vid' is created.
-   7.Classification:
    Finally, classifiers of types DT, KNN and RF are defined and trained for each set of data. The resultant classifiers are saved in .pkl files, a summary is written  to the file 'metricas.txt' and performance graphs like those on directory 'classification' are created.
