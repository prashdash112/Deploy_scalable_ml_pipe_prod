# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model created as part of Udacity ML DevOps nanodegree by Prashant Singh

## Intended Use
This model is used to predict the salary of an individual based off a number of depended variables like education, age, etc. The data have both continuous and categorial variables. The categorial variables are split into slices and their respective performace metrices are also measured and logged in slice.txt file.

## Training Data
We used 80% of the data for training purposes.

## Evaluation Data
We sliced 20% of total data for testing purposes.

## Metrics
Metrics and Performance on overall dataset: Precision: 0.859 Recall: 0.754 FBeta: 0.803

## Ethical Considerations
The model is not bias-proof and there's definitely a scope for improvement in this area. 

## Caveats and Recommendations
Some slices performed relatively worse than the other slices as logged in the slice_outut.txt file. More data for some of the variable slices may lead to a better metric score.
