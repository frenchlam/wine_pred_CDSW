from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
  .appName("wine-quality-model") \
  .master("local[*]") \
  .config("spark.driver.memory","2g")\
  .config("spark.hadoop.yarn.resourcemanager.principal","mlamairesse") \
  .getOrCreate()
    
model = PipelineModel.load("file:///home/cdsw/models/spark")


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
  StructField("alcohol", DoubleType(), True),     
])


def predict(args):
  split=args["feature"].split(";")
  features=[list(map(float,split[:11]))]
  features_df = spark.createDataFrame(features, schema)
  result=model.transform(features_df).collect()[0].prediction
  if result == 1.0:
    return {"result": "Bad"}
  else:
    return {"result" : "Good"}
  
# pre-heat the model
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good