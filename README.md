# MLzoomcamp midterm project

## Project parts:

### Problem description

I'm gonna be working with the Credit score classification dataset:

Link to the dataset on Kaggle : [Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)  
(I'm using only the training part)

Initial dataset is stored here: [data/initial_dataset.csv](https://github.com/Stayermax/MLzoomcamp_midterm_project/blob/main/data/initial_dataset.csv)

* This dataset consists of customer records for one year for the time period between January and August.
* Customer records have a lot of fields such as name, occupation, age, payment behaviour and many more.
* **We want to predict customer's credit score** which can be one out of three options: **Poor**, **Standard** or **Good**.
* Dataset can include data about the same customer a few times, and his/her credit score can change with time.
* Some of the features are more important and useful for credit score prediction and we gonna try to find them. 

**So once again our task is to predict customer credit score based on customer parameters.**

### EDA + Model training
**It can take up to 15 minutes to run the whole notebook, my models require some time for training.**  

The whole EDA process fully described in the [Jupyter notebook](https://github.com/Stayermax/MLzoomcamp_midterm_project/blob/main/notebook.ipynb)
In a few words:

#### Data analysis
1. When we initially read the dataset most of the columns have mixed types, so we clean them in some different ways.
   * For example some columns 'like annual_income' or 'monthly_inhand_salary' should be numeric, but have values like 505_ or 16.3_, so we delete underscores and cust them to numeric
2. We find nulls in dataset and patch them in different ways.
   * Many clients have records for all 8 months of the dataset, so if we are lacking some data for client for month of July, for instace ssn, but we know it for the rest of the month it's really easy to restore it.
3. We detect outliers and path them as well.
   * Some fields had very unusual values that are contradicting common sense (for example num_of_loan had value -100, which is impossible, since number of loans can't be negative), so we fixed it by averaging num_of_loan for the customer.

#### Features engineering
1. We drop some columns like id, name and customer_id
2. Out of categorical values we find the most unimportant features using mutual information metric and drop them as well (like ssn_area or payment_behaviour)
3. Fully prepared dataset is stored here: [data/prepared_dataset.csv](https://github.com/Stayermax/MLzoomcamp_midterm_project/blob/main/data/prepared_dataset.csv)

#### Splitting the data
We split the data into train, validation and test in proportions 60/20/20

#### Model training 
We train 5 models:
1. Dummy model (always predicts Standard credit rating)
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. XGboost

For each of the models we also fine tune hyperparameters on validation set in order to improve the performance.

Final results on test set (metric is averaged accuracy between 3 clasees):
1. Dummy model: 53%
2. Logistic Regression: 63.48%
3. Decision Tree: 71.45%
4. Random Forest: 81.12%
5. XGboost: 82.1%

## Exporting notebook to the script

The whole training process can be reproduced using [train.py](https://github.com/Stayermax/MLzoomcamp_midterm_project/blob/main/train.py) file.  
This file all the same operations we did in the notebook with the initial dataset and then trains XGboost model on preprocessed data.  
Ready model is saved in [classifier_model.pkl](https://github.com/Stayermax/MLzoomcamp_midterm_project/blob/main/classifier_model.pkl) file.

## Reproducibility

I run my code from MAC using conda and it's possible you'll have issues with xgboost library.
But hopefully not. 
All installations you need are the following:

      pip3 install pandas
      pip3 install numpy
      pip3 install seaborn
      pip3 install matplotlib
      pip3 install scikit-learn
      pip3 install matplotlib
      pip3 install tqdm
      pip3 install fastapi

for xgboost try one of the following:

      pip3 install xgboost
   
or

      conda install py-xgboost

## Model deployment

My model is deployed via FastApi. You can run it using the following command:

    uvicorn prediction_service:app --reload --port 8000 --host localhost

You can open swagger of this service here: 
      
      localhost:8000/credit_rating_serivce/api/docs

This service has two endpoints:  
#### **Random sample endpoint:**  
      
      GET localhost:8000/credit_rating_serivce/api/random_sample

This GET endpoint returns you a randomly sampled dictionary with actually expected label in the following format:
   
   {"actual_credit_score": actual_credit_score, "customer_data": customer_data_dict}

#### **Prediction endpoint:**  


      POST localhost:8000/credit_rating_serivce/api/predict_dict

This POST endpoint accepts a dictionary (that you can take from the first endpoint) and predicts a credit score. 

## Dependency and environment management

But still let's try to run it:
1. Create conda environment:
      
   conda env create -f environment.yml -n vitaly_venv python=3.10

2. Import my libraries and install them:

    ...

## Containerization

In order to build and run prediction service in Docker container:
    
    docker build . -t prediction_service
    docker run -d -p 8000:8000 prediction_service

The service gonna be accessingle from here:
   
      localhost:8000/credit_rating_serivce/api/docs

## Cloud deployment 

I deployed this service to the server of my company, so it won't be accessible from outside.  
But here is the video of how I did it: