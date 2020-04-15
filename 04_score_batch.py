from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel


# get Environment bucket location
import os
ENV_BUCKET="s3a://demo-aws-2/datalake/"

try : 
  DL_s3bucket=os.environ["STORAGE"]+"/datalake/"
except KeyError: 
  DL_s3bucket=ENV_BUCKET

save_path = DL_s3bucket+"tmp/wine_batch_predict"

spark = SparkSession.builder \
  .appName('wine-quality-batch-score') \
  .config("spark.executor.memory","2g") \
  .config("spark.executor.cores","2") \
  .config("spark.executor.instances","3") \
  .config("spark.yarn.access.hadoopFileSystems",DL_s3bucket) \
  .getOrCreate()

schema = StructType([StructField("fixedacidity", DoubleType(), True),     
  StructField("volatileacidity", DoubleType(), True),     
  StructField("citricacid", DoubleType(), True),     
  StructField("residualsugar", DoubleType(), True),     
  StructField("chlorides", DoubleType(), True),     
  StructField("freesulfurdioxide", DoubleType(), True),     
  StructField("totalsulfurdioxide", DoubleType(), True),     
  StructField("density", DoubleType(), True),     
  StructField("ph", DoubleType(), True),     
  StructField("sulphates", DoubleType(), True),     
  StructField("alcohol", DoubleType(), True)
])

'''
## From file ###
data_path = DL_s3bucket+"tmp/wine_pred/WineNewGBTDataSet.csv"
wine_raw_df = spark.read.csv(data_path, schema=schema,sep=';')
wine_df = wine_raw_df.drop('quality')
'''

## from Hive ###
wine_df = spark.sql('''SELECT * FROM default.wineds_ext_nolabel''')
print("Show first 5 lines")

wine_df.show(5)

## load Model 
model = PipelineModel.load(DL_s3bucket+"tmp/models/spark")

## Predict 
result=model.transform(wine_df)
result.show(5)

## save file 
result.select("fixedacidity", 
          "volatileacidity", 
          "citricacid",
          "residualsugar",
          "chlorides",
          "freesulfurdioxide",
          "totalsulfurdioxide",
          "density",
          "ph",
          "sulphates",
          "alcohol",
          "probability",
          "prediction").write.mode('overwrite').save(save_path, format="parquet")


spark.stop()