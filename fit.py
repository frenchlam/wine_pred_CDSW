import cdsw
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *

conf = SparkConf().setAppName("wine-quality-build-model")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#set path to data
data_path = "/tmp/mlamairesse"
data_file = "WineNewGBTDataSet.csv"

# # Get params
# Declare parameters 
param_numTrees= int(sys.argv[1])
param_maxDepth=int(sys.argv[2])
param_impurity=sys.argv[3]

#param_numTrees= 10
#param_maxDepth= 15
#param_impurity= "gini"

cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)

# # Load the data
# 
# We need to load data from a file in to a Spark DataFrame.
# Each row is an wine, and each column contains
# attributes of that wine.
#
#     Fields:
#     fixedAcidity: numeric
#     volatileAcidity: numeric
#     citricAcid: numeric
#     residualSugar: numeric
#     chlorides: numeric
#     freeSulfurDioxide: numeric
#     totalSulfurDioxide: numeric
#     density: numeric
#     pH: numeric
#     sulphates: numeric
#     Alcohol: numeric
#     Quality: discrete

schema = StructType([StructField("fixedAcidity", DoubleType(), True),     
  StructField("volatileAcidity", DoubleType(), True),     
  StructField("citricAcid", DoubleType(), True),     
  StructField("residualSugar", DoubleType(), True),     
  StructField("chlorides", DoubleType(), True),     
  StructField("freeSulfurDioxide", DoubleType(), True),     
  StructField("totalSulfurDioxide", DoubleType(), True),     
  StructField("density", DoubleType(), True),     
  StructField("pH", DoubleType(), True),     
  StructField("sulphates", DoubleType(), True),     
  StructField("Alcohol", DoubleType(), True),     
  StructField("Quality", StringType(), True)
])

wine_data_raw = sqlContext.read.format('com.databricks.spark.csv').option("delimiter",";").load(data_path+'/'+data_file, schema = schema)

# Remove unvalid data
wine_data = wine_data_raw.filter(wine_data_raw.Quality != "1")

# # Build a classification model using MLLib
# 
# We want to build a predictive model.
# 
# 
# We need to:
# * Gather all features we need into a single column in the DataFrame.
# * Split labeled data into training and testing set
# * Fit the model to the training data.
# 
# ##  Feature Extraction
# We need to define our input features.
# 
# [PySpark Pipeline Docs](https://spark.apache.org/docs/1.5.0/api/python/pyspark.ml.html)
# 

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

labelIndexer = StringIndexer(inputCol = 'Quality', outputCol = 'label')
featureIndexer = VectorAssembler(
    inputCols = ['fixedAcidity', "volatileAcidity", "citricAcid", "residualSugar", "chlorides", "freeSulfurDioxide", "totalSulfurDioxide", "density", "pH", "sulphates", "Alcohol"],
    outputCol = 'features')


# # Fit a RandomForestClassifier
# 
# Fit a random forest classifier to the data. Try experimenting with different values of the `maxDepth`, `numTrees`, and `entropy` parameters to see which gives the best classification performance. Do the settings that give the best classification performance on the training set also give the best classification performance on the test set?
# 
# Have a look at the [documentation](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier]documentation).

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

(trainingData, testData) = wine_data.randomSplit([0.7, 0.3])
 
classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', 
                                    numTrees = param_numTrees, 
                                    maxDepth = param_maxDepth,  
                                    impurity = param_impurity)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, classifier])
model = pipeline.fit(trainingData)

predictions = model.transform(testData)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)

model.write().overwrite().save("models/spark")

!rm -r -f models/spark
!rm -r -f models/spark_rf.tar
!mkdir models
!hdfs dfs -get ./models/spark models/
!tar -cvf models/spark_rf.tar models/spark

cdsw.track_file("models/spark_rf.tar")

sc.stop()

