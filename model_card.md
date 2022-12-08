# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model created as part of Udacity ML DevOps nanodegree by Prashant Singh

## Intended Use
This ML model is used to predict the salary class of an individual based off a number of variables like education, age, etc. The data have both continuous and categorial variables. The main purpose of the model is to predict the salary class of individuals.

We used a Random forest classifier(ensemble method) as models like linear regression, SVM tend to underfit on our current data which leads to a slightly low prediction accuracy which eventually results in a poor model performance. 

## Training Data
We used 80% of the data for training purposes.

## Evaluation Data
We sliced 20% of total data for testing purposes.

## Model parameters in use 
max_depth = 5 
random_state = 101
n_estimator = 10

## Metrics
Metrics and Performance on overall dataset: Precision: 0.859 Recall: 0.754 FBeta: 0.803

## Ethical Considerations
The model is not bias-proof and there's definitely a scope for improvement in this area. In order to test for bias, one can make use of Aequitas package. 

## Caveats and Recommendations
Some slices performed relatively worse than the other slices as logged in the slice_outut.txt file. More data for some of the variable slices may lead to a better metric score.
