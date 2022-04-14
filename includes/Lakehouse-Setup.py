# Databricks notebook source
# MAGIC %md
# MAGIC ## Building a Lakehouse with Delta Lake
# MAGIC 
# MAGIC This notebook will demonstrate how we can set up our Lakehouse with Delta Lake to progressively transform data into tables for downstream uses. 
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/02/telco-accel-blog-2-new.png" width=1012/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this case we'll grab a CSV from the web, but we could also use Python or Spark to read data from databases or cloud storage.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load into Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC #### Path configs

# COMMAND ----------

# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Set config for database name, file paths, and table names
database_name = 'ml_demo_churn'

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
driver_to_dbfs_path = 'dbfs:/home/{}/ml_demo_churn/Telco-Customer-Churn.csv'.format(user)
dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# Paths for various Delta tables
bronze_tbl_path = '/home/{}/ml_demo_churn/bronze/'.format(user)
silver_tbl_path = '/home/{}/ml_demo_churn/silver/'.format(user)
gold_tbl_path = '/home/{}/ml_demo_churn/gold/'.format(user)
#automl_tbl_path = '/home/{}/ml_demo_churn/automl-silver/'.format(user)
ml_preds_path = '/home/{}/ml_demo_churn/preds/'.format(user)

bronze_tbl_name = 'bronze_customers'
silver_tbl_name = 'silver_customers'
gold_tbl_name = 'gold_customers'
ml_preds_tbl_name = 'ml_preds'

# Delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))
# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+gold_tbl_name, ignore_errors=True)
shutil.rmtree('/dbfs'+ml_preds_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read and display

# COMMAND ----------

# Define schema
schema = StructType([
  StructField('customerID', StringType()),
  StructField('gender', StringType()),
  StructField('seniorCitizen', DoubleType()),
  StructField('partner', StringType()),
  StructField('dependents', StringType()),
  StructField('tenure', DoubleType()),
  StructField('phoneService', StringType()),
  StructField('multipleLines', StringType()),
  StructField('internetService', StringType()), 
  StructField('onlineSecurity', StringType()),
  StructField('onlineBackup', StringType()),
  StructField('deviceProtection', StringType()),
  StructField('techSupport', StringType()),
  StructField('streamingTV', StringType()),
  StructField('streamingMovies', StringType()),
  StructField('contract', StringType()),
  StructField('paperlessBilling', StringType()),
  StructField('paymentMethod', StringType()),
  StructField('monthlyCharges', DoubleType()),
  StructField('totalCharges', DoubleType()),
  StructField('churnString', StringType())
  ])

# Read CSV, write to Delta and take a look
bronze_df = spark.read.format('csv').schema(schema).option('header','true')\
               .load(driver_to_dbfs_path)

bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Bronze

# COMMAND ----------

# Create bronze table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,bronze_tbl_name,bronze_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can query our table!

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ml_demo_churn.bronze_customers

# COMMAND ----------

# MAGIC %md
# MAGIC No only can we read data in batch mode, Structured Streaming is a powerful capability for building end-to-end continuous applications. At a high-level, it offers the following features:
# MAGIC 
# MAGIC 1. __Output tables are always consistent__ with all the records in a prefix (partition) of the data, we will process and count in order.
# MAGIC 1. __Fault tolerance__ is handled holistically by Structured Streaming, including in interactions with output sinks.
# MAGIC 1. Ability to handle __late and out-of-order event-time data__. 
# MAGIC 
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/Streaming/continuous_streaming/continuous-apps-1024x366.png" alt="" width="1000"/>
# MAGIC 
# MAGIC Common stream __technologies__ include:
# MAGIC 
# MAGIC - Sockets
# MAGIC - Java Message Service (JMS) applications, including RabbitMQ, ActiveMQ, IBM MQ, etc.
# MAGIC - Apache Kafka

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Transaction Log
# MAGIC We also now have full visibility into our table by accessing the transaction log

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY ml_demo_churn.bronze_customers

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time Travel
# MAGIC 
# MAGIC Since we're keeping a history of changes over time, we can also travel back in time to a previous verion of our table! There's lots of different ways to explore your history as well. 
# MAGIC 
# MAGIC For example, by timestamp:
# MAGIC 
# MAGIC In Python:
# MAGIC ```python
# MAGIC df = spark.read \
# MAGIC   .format("delta") \
# MAGIC   .option("timestampAsOf", "2019-01-01") \
# MAGIC   .load("/path/to/my/table")
# MAGIC   ```
# MAGIC SQL syntax:
# MAGIC ```sql
# MAGIC SELECT count(*) FROM my_table TIMESTAMP AS OF "2019-01-01"
# MAGIC SELECT count(*) FROM my_table TIMESTAMP AS OF date_sub(current_date(), 1)
# MAGIC SELECT count(*) FROM my_table TIMESTAMP AS OF "2019-01-01 01:30:00.000"
# MAGIC ```
# MAGIC 
# MAGIC or by version number:
# MAGIC Python syntax:
# MAGIC ```python
# MAGIC df = spark.read \
# MAGIC   .format("delta") \
# MAGIC   .option("versionAsOf", "5238") \
# MAGIC   .load("/path/to/my/table")
# MAGIC 
# MAGIC df = spark.read \
# MAGIC   .format("delta") \
# MAGIC   .load("/path/to/my/table@v5238")
# MAGIC ```
# MAGIC SQL syntax:
# MAGIC ```sql
# MAGIC SELECT count(*) FROM my_table VERSION AS OF 5238
# MAGIC SELECT count(*) FROM my_table@v5238
# MAGIC SELECT count(*) FROM delta.`/path/to/my/table@v5238`
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Handling Schemas: Enforcement and Evolution
# MAGIC Most of the time, we want strict schemas to be adhered to so that downstream teams and processes that rely on that schema don't break. But sometimes the data coming in is changing rapidly and we need to evolve our schemas instead of strictly enforcing them. Delta supports both [shema enforcement and evolution](https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html). Let's dive into each
# MAGIC 
# MAGIC ###### Schema Enforcement
# MAGIC Schema enforcement, also known as schema validation, is a safeguard in Delta Lake that ensures data quality by **rejecting writes to a table that do not match the table’s schema**. 
# MAGIC 
# MAGIC ###### Schema Evolution
# MAGIC You can choose to change your schema by simply adding an extra line of code! 
# MAGIC ```python
# MAGIC (
# MAGIC   new_schema_df
# MAGIC   .write
# MAGIC   .mode("append")
# MAGIC   .option("mergeSchema", "true") # <- 1 LOC to allow for schema evolution
# MAGIC   .saveAsTable("ml_demo_churn.bronze_customers")
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Create Gold

# COMMAND ----------

@udf("boolean")
def convert_to_bool(val):
  if val == "Yes":
    return bool(True)
  else:
    return bool(False)
  
@udf("int")
def convert_to_int(val):
  if val == "Month-to-month":
    return 0
  elif val == "One year":
    return 1
  else:
    return 2
  

  
df_gold = spark.table("ml_demo_churn.bronze_customers") \
                         .drop() \
                         .withColumn("Churn", convert_to_bool(col("churnString"))) \
                         .withColumn("SeniorCitizen", convert_to_bool(col("seniorCitizen"))) \
                         .withColumn("Partner", convert_to_bool(col("partner"))) \
                         .withColumn("Dependents", convert_to_bool(col("dependents"))) \
                         .withColumn("PaperlessBilling", convert_to_bool(col("paperlessBilling"))) \
                         .withColumn("Contract", convert_to_int(col("contract"))) \
                         .withColumnRenamed("multipleLines", "MultipleLines") \
                         .withColumnRenamed("internetService", "InternetService") \
                         .withColumnRenamed("onlineSecurity", "OnlineSecurity") \
                         .withColumnRenamed("deviceProtection", "DeviceProtection") \
                         .withColumnRenamed("techSupport", "TechSupport") \
                         .withColumnRenamed("streamingTV", "StreamingTV") \
                         .withColumnRenamed("streamingMovies", "StreamingMovies") \
                         .withColumnRenamed("monthlyCharges", "MonthlyCharges") \
                         .withColumnRenamed("totalCharges", "TotalCharges") \
                         .select("customerID", 
                                 "gender", 
                                 "SeniorCitizen", 
                                 "Partner", 
                                 "Dependents", 
                                 "tenure", 
                                 "PhoneService",
                                 "MultipleLines",
                                 "InternetService",
                                 "OnlineSecurity",
                                 "DeviceProtection",
                                 "TechSupport",
                                 "StreamingTV",
                                 "StreamingMovies",
                                 "Contract",
                                 "PaperlessBilling",
                                 "PaymentMethod",
                                 "MonthlyCharges",
                                 "TotalCharges",
                                 "Churn")
  
display(df_gold)

# COMMAND ----------

df_gold.write.format('delta').mode('overwrite').save(gold_tbl_path)

spark.sql('''
   CREATE TABLE `{}`.{}
   USING DELTA 
   LOCATION '{}'
   '''.format(database_name,gold_tbl_name,gold_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Constraints
# MAGIC We'd like more assurances around gold tables. We can `ADD CONTRAINT`s to tables to ensure data integrity for downstream teams. More on [Delta Constraints](https://docs.databricks.com/delta/delta-constraints.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC ALTER TABLE  ml_demo_churn.gold_customers ADD CONSTRAINT validContract CHECK (Contract > -1 and Contract < 3);
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC #### Table ACLs
# MAGIC 
# MAGIC It is very simple to govern who has access to all of these tables using sql syntax. More on Governance here.
# MAGIC ```
# MAGIC GRANT USAGE ON DATABASE ml_demo_churn TO finance;
# MAGIC GRANT CREATE ON DATABASE ml_demo_churn TO finance;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Huge Performance Improvements: Data Skipping, ZORDER, OPTIMIZE, Caching

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Data Skipping
# MAGIC Delta collects column statistics (min and max) to skip reading as many files as possible when querying with a `WHERE` clause.
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2018/07/image7.gif">
# MAGIC 
# MAGIC By default, Delta Lake on Databricks collects statistics on the first 32 columns defined in your table schema. (You can change this value using the table property `dataSkippingNumIndexedCols`.)
# MAGIC 
# MAGIC The above information is used for Data skipping information and is collected automatically when you write data into a Delta table. Delta Lake on Databricks takes advantage of this information (minimum and maximum values) at query time to provide faster queries. 
# MAGIC 
# MAGIC **NOTE:** This is another feature that you don't need to configure manually; the feature is **automatically** activated whenever possible. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### ZORDER
# MAGIC To optimize performance **even more**, you can use **Z-Ordering** which takes advantage of the information gathered for Data Skipping. 
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2018/07/Screen-Shot-2018-07-30-at-2.03.55-PM.png">
# MAGIC 
# MAGIC ##### What is Z-Ordering? 
# MAGIC 
# MAGIC Z-Ordering is a technique to **colocate related information** in the same set of files (dimensionality reduction). 
# MAGIC 
# MAGIC As mentioned above, This co-locality is automatically used by Delta Lake on Databricks data-skipping algorithms to dramatically reduce the amount of data that needs to be read. 
# MAGIC 
# MAGIC To Z-Order data, you specify the columns to order on in the `ZORDER BY` clause:

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE ml_demo_churn.gold_customers ZORDER BY customerID, tenure

# COMMAND ----------

# MAGIC %md 
# MAGIC #### OPTIMIZE 
# MAGIC Delta Lake on Databricks can improve the speed of read queries from a table by **coalescing small files into larger ones**. 
# MAGIC 
# MAGIC <img src="https://github.com/Corbin-A/images/blob/main/databricks/1manydelta/optimize.png?raw=true">
# MAGIC 
# MAGIC You trigger compaction by running the `OPTIMIZE` command.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Caching
# MAGIC When you use Databricks, all the tables are saved to a S3/ADLS/GCS bucket in YOUR account. When you spin up a cluster, you are launching VM's in YOUR account. So when a cluster needs to read data, it needs to get it from COS. The Delta cache accelerates subsequent data reads by **caching copies of the remote files in the nodes’ local storage** using a fast intermediate data format. Successive reads of the same data are then performed locally, which results in significantly improved reading speed.
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_images/databricks-architecture.png">
# MAGIC 
# MAGIC **NOTE:** When the Delta cache is enabled, data that has to be fetched from a remote source is **automatically** added to the cache after it is accessed the first time. This means, you don't have to do anything to cache the data you're currently using. 
# MAGIC 
# MAGIC However, if you want to preload data into the Delta cache beforehand, you can use the `CACHE` command:

# COMMAND ----------

# MAGIC %sql 
# MAGIC CACHE SELECT * FROM ml_demo_churn.gold_customers

# COMMAND ----------

# MAGIC %md
# MAGIC #### VACUUM
# MAGIC As you can imagine, keeping large table histories takes up more and more space in S3/ADLSg2/GCS. It is very important to VACUUM your Delta tables to delete history that is older than you desire to keep (and pay cloud storage fees for). This is done through the VACUUM command.
# MAGIC 
# MAGIC **NOTE**: If you run VACUUM on a Delta table, you lose the ability time travel back to a version older than the specified data retention period.
# MAGIC 
# MAGIC ```sql
# MAGIC VACUUM ml_demo_churn.gold_customers RETAIN 0 HOURS
# MAGIC ```
