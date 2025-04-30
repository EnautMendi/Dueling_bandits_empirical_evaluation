# Dueling Bandits Strategies Empirical Evaluation
## Data

The data for this study can be obtained from: 
- Full experiment results and logs of the [Non-condorcet Set-up](full_experiment/)
- Full experiment results and logs of the [Condorcet Set-up](full_experiment_condorcet/)
- The files with the summary of the results showing the mean and std of all the metrics for all strategies and the plots present in the paper [Results folder](results/)

## Experiment
To ensure data integrity and suitability for analysis, the following steps were taken:

1. All the strategies were implemented in [strategies.py](strategies.py) following the same base structure.
     
2. The experiment was executed by running the [main.py](main.py) script, which contains the strategies to be executed together with both setup initializations and hyperparameters as the number of arms, iterations, and runs.

3. The data obtained from the experiment was processed by following the steps on [results_analysis.ipynb](results_analysis.ipynb), where the main results were summarized by getting the mean and std, and the plots were generated using the log folders.
