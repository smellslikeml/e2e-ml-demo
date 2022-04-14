# Databricks notebook source
# MAGIC %md
# MAGIC # End to End ML Demo with Databricks and MLflow
# MAGIC 
# MAGIC This notebook will walk through a sample usecase of predicting customer churn. It features using MLflow to track and log our experiments and serve our final model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Training Data
# MAGIC 
# MAGIC For this example, we'll be working customer data. 
# MAGIC The data is stored in the Delta Lake format.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC We'll read in our bronze table and do some basic transformations, including performing one-hot encoding and cleaning up the column names, to create a silver table.

# COMMAND ----------

# MAGIC %run ./includes/Lakehouse-Setup

# COMMAND ----------

# Read into Spark
df = spark.table("ml_demo_churn.bronze_customers")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can always switch to `pyspark.pandas` if you prefer pandas syntax while leveraging spark under the hood!

# COMMAND ----------

import pyspark.pandas as pd

def compute_churn_features(data):
  
  
  data = data.to_pandas_on_spark()
  
  # OHE
  data = pd.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],dtype = 'int64')
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Convert churnString into boolean value
  churn_values = {"Yes": 1.0, "No": 0.0}
  data['churn'] = data["churnString"].map(lambda x: churn_values[x])

  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC After applying our transformations, we'll save the resulting dataframe into a silver Delta Lake table.

# COMMAND ----------

# Set paths
database_name = 'ml_demo_churn'
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
silver_tbl_path = '/home/{}/ml_demo_churn/silver/'.format(user)
silver_tbl_name = 'silver_customers'

# COMMAND ----------

# Write out silver-level data to Delta lake
trainingDF = compute_churn_features(df).to_spark()

trainingDF.write.format('delta').mode('overwrite').save(silver_tbl_path)

# Drop table if exists (for demo)
spark.sql('''
   DROP TABLE IF EXISTS `{}`.{}
   '''.format(database_name,silver_tbl_name))

# Create silver table
spark.sql('''
   CREATE TABLE `{}`.{}
   USING DELTA 
   LOCATION '{}'
   '''.format(database_name,silver_tbl_name,silver_tbl_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ml_demo_churn.silver_customers

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizations
# MAGIC 
# MAGIC Create quick visualizations and plots simply by toggling the display buttons shown under the results! Here, we'll make a fast bar chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   gender,
# MAGIC   churnString,
# MAGIC   COUNT(churnString)
# MAGIC FROM
# MAGIC   ml_demo_churn.bronze_customers
# MAGIC GROUP BY
# MAGIC   gender,
# MAGIC   churnString;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: Compute and write features to Feature Store!
# MAGIC We can also build a feature store on top of data and then use it to train a model and deploy both the model and features to production. After executing the next cell, the table will be visible and searchable in the [Feature Store](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#feature-store) -- try it!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```python
# MAGIC #remove %md to run
# MAGIC from databricks.feature_store import feature_table
# MAGIC from databricks.feature_store import FeatureStoreClient
# MAGIC 
# MAGIC fs = FeatureStoreClient()
# MAGIC 
# MAGIC churn_features_df = compute_churn_features(df)
# MAGIC 
# MAGIC churn_feature_table = fs.create_feature_table(
# MAGIC   name='ml_demo_churn.silver_features',
# MAGIC   keys='customerID',
# MAGIC   schema=churn_features_df.spark.schema(),
# MAGIC   description='These features are derived from the ml_demo_churn.bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
# MAGIC )
# MAGIC 
# MAGIC fs.write_table(df=churn_features_df.to_spark(), name='ml_demo_churn.silver_features', mode='overwrite')
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment Tracking
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" width="800px"/>
# MAGIC 
# MAGIC This notebook walks through a basic Machine Learning example. A resulting model from one of the models will be deployed using MLflow APIs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training XGBoost model with Hyperparameter Search
# MAGIC 
# MAGIC The modeling here is simplistic training an XGBoost classifier. We've expanded this example to include [hyperopt](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) for distributed asynchronous hyperparameter optimization.
# MAGIC 
# MAGIC MLflow is library-agnostic. You can use it with any machine learning library, and in any programming language, since all functions are accessible through a [REST API](https://mlflow.org/docs/latest/rest-api.html#rest-api) and [CLI](https://mlflow.org/docs/latest/cli.html#cli).

# COMMAND ----------

# helper packages
import pandas as pd
import numpy as np
import time
import warnings

# modeling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# hyperparameter tuning
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from hyperopt.pyll import scope

# model/grid search tracking
import mlflow

# COMMAND ----------

# typically not advised but doing this to minimize excessive messaging 
# during the grid search
warnings.filterwarnings("ignore") 

# COMMAND ----------

training_pd = spark.read.table("ml_demo_churn.silver_customers").toPandas()

X = training_pd.drop(["churn", "churnString", "customerID"], axis=1)
y = training_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert our data to a DMatrix object, an XGBoost internal data structure optimized for memory efficiency and training speed
train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)

# COMMAND ----------

# define our search space
search_space = {
  'learning_rate': hp.loguniform('learning_rate', -7, 0),
  'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
  'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
  'subsample': hp.uniform('subsample', 0.5, 1),
  'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
  'gamma': hp.loguniform('gamma', -10, 10),
  'alpha': hp.loguniform('alpha', -10, 10),
  'lambda': hp.loguniform('lambda', -10, 10),
  'objective': 'binary:logistic',
  'eval_metric': 'auc'
}

# COMMAND ----------

def fit_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog(silent=True)
  
  # However, we can log additional information by using an MLFlow tracking context manager 
  with mlflow.start_run(nested=True):
    
    # convert our data to a DMatrix object, an XGBoost internal data structure optimized for memory efficiency and training speed
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)

    # Train model and record run time
    start_time = time.time()
    booster = xgb.train(params=params, dtrain=train, num_boost_round=5000, evals=[(test, "test")], early_stopping_rounds=100, verbose_eval=False)
    
    # train
    run_time = time.time() - start_time
    mlflow.log_metric('runtime', run_time)
    
    # Record AUC as primary loss for Hyperopt to minimize
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -auc_score, 'booster': booster.attributes()}

# COMMAND ----------

spark_trials = SparkTrials(parallelism=12)

# COMMAND ----------

with mlflow.start_run(run_name='xgboost_search'):
  best_params = fmin(
    fn=fit_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=10,
    rstate=np.random.default_rng(123),
    trials=spark_trials
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters.
# MAGIC 
# MAGIC During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.<br>
# MAGIC Thereafter, the registry will let data scientists track multiple versions over the course of model progression as they assign each version with a lifecycle stage:
# MAGIC - Staging
# MAGIC - Production
# MAGIC - Archived
# MAGIC <br>
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg" width="1000px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Deployment
# MAGIC Using MLFlow APIs, models can be deployed:
# MAGIC - In batch or streaming pipelines in Databricks with Python functions as Spark or Pandas UDFs
# MAGIC - As REST Endpoints using built-in MLflow Model Serving
# MAGIC - As Python functions in AWS SageMaker
# MAGIC - As Docker images and deployed on external infrastructure
# MAGIC 
# MAGIC And many other options! Here, we demonstrate creating a REST endpoint with MLflow Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC You can select any tracked model and load as a spark udf/pandas udf to predict on a dataframe.

# COMMAND ----------

import mlflow
logged_model = 'runs:/aaa9fa1197de4421abeb22e209d56dda/model' #replace with your model id

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
inference_data = spark.read.format("delta").load("/home/salma.mayorquin@databricks.com/ml_demo_churn/eval/").toPandas() #replace path

inference_data['predictions'] = loaded_model.predict(inference_data)
display(inference_data)

# COMMAND ----------

import mlflow
logged_model = 'runs:/aaa9fa1197de4421abeb22e209d56dda/model' #replace with your model run id

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

inference_data = spark.read.format("delta").load("/home/salma.mayorquin@databricks.com/ml_demo_churn/eval/") #replace path
columns = inference_data.columns

# Predict on a Spark DataFrame.
display(inference_data.withColumn('predictions', loaded_model(*columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC Once a model has been registered, you can construct the `model_uri` with its dedicated model name and stage label.

# COMMAND ----------

model_name = 'ML_churn_demo'
stage = 'Staging'

staged_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{stage}")

display(inference_data.withColumn('predictions', staged_model(*columns)))
