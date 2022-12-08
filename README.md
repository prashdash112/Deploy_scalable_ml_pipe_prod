## Project name: Deploy_scalable_ml_pipe_prod
In this project, we've build a machine learning classifier that classifies the individuals in 2 salary groups based on several values like age, education, etc. We've created an api based on this model and deployed it on heroku so users can post request and get predicted values. 

### Github repo link: https://github.com/prashdash112/Deploy_scalable_ml_pipe_prod


### This Project includes the below mentioned componenets:
* Setting up GitHub Actions.
* Creating Random forest machine learning model.
* Wrote unit tests to test the ml model.
* Wrote function that computes model metrics on slices of the data and saves to slice_output.txt.
* Wrote a model card to brief the model performance,purpose and metrices.
* Created a REST API with FastAPI to cater the model to other users.
* Wrote unit tests to test the api client.
* Deployed the ml app to Heroku.
* Queried the live API for both post and get methods.

### Github actions
Added github actions to perform continuous integration via continuous pytest and flake8 testing. 

### ML model
Developed a random forest classifier to classify the individuals into 2 salary groups.

### Testing ML model
Wrote unit tests to see whether the model is returning the expected data types or not. It also performs a check on model parameters like max_depth.

### Slice metrices 
A script that calculates all metrices on slices of data for e.g a slice can be a dataset for all distinct values of education field.

### Model Card
A model card is a model documentation that have all details and nuances related to out deployed model.

### FastApi
Wrote a rest api using fast api services to cater the model to other users.

### Api test
Wrote several unit tests using the test_client class of fastApi to test the api see whether the connection status is "OK" or not.

### Deployment 
Deployed the Rest API to a cloud service provider: heroku. At this stage, the model is in production state with both CI/CD involved and stitched together. 

### Querying the live API
A user can query the API using the request_post.py file. User needs to run the command:
'''
python request_post.py
'''
