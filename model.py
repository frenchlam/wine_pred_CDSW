from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
      .appName("wine-quality-model") \
      .master("local[*]") \
      .getOrCreate()
    
model = PipelineModel.load("file:///home/cdsw/models/spark")

features = ['fixedAcidity', 
             "volatileAcidity", 
             "citricAcid", 
             "residualSugar", 
             "chlorides", 
             "freeSulfurDioxide", 
             "totalSulfurDioxide", 
             "density", 
             "pH", 
             "sulphates", 
             "Alcohol"]


def predict(args):
  wine=args["feature"].split(";")
  feature = spark.createDataFrame([map(float,wine[:11])], features)
  result=model.transform(feature).collect()[0].prediction
  if result == 1.0:
    return {"result": "Bad"}
  else:
    return {"result" : "Good"}
  
# pre-heat the model
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"})