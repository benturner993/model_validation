# Objective
The object of this code repository is to automate and simplify the scoring of classification models.

# What does it do?
For any input csv input with *id* and *predictions*, this code will automatically generate:
    
* Confusion matrix
* Accuracy score
* Null accuracy score
* Precision score
* Recall score
* F1 score
* AUROC
* MCC
* MSE
* X-Graph

# Instructions
To use:
* Place .exe file in the same directory as your .csv file
* Run the .exe file
* Where prompted, input the names of your models, the csv and the predictions columns (in probability format)
* Wait for the process to finish
* The output will be a .txt file and a .png graph in your working directory

A worked example:
* Place .exe file in the same directory as *model_predictions.csv*
* Run the .exe file
* Where prompted, input: ModelA, ModelB, model_predictions.csv, pred1_prob, pred2_prob, actual
* Wait for the process to finish
* The output will be a ModelA_ModelB_results.txt file and a ModelA_ModelB_graph.png graph in your working directory 

# Notes
At present this code is only created for use with classification models.
For any errors please contact Ben.