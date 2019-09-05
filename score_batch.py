from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
      .appName("wine-quality-model") \
      .master("yarn") \
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


data_path = "/tmp/wine_pred/WineNewGBTDataSet.csv"
wine_raw_df = spark.read.csv(data_path, schema=schema,sep=';')
wine_df = wine_raw_df.drop('quality')



model = PipelineModel.load("/user/systest/models/spark/")
result=model.transform(wine_df)

result.show(2)

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
          "prediction").write.mode('overwrite').save("wine_predicted.parquet", format="parquet")

spark.stop()

!hdfs dfs -ls 
