import cdsw
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# ## Get parameters for experiments 
# ### Note uncomment for experiments
#param_numTrees= int(sys.argv[1])
#param_maxDepth=int(sys.argv[2])
#param_impurity=sys.argv[3]

### Comment out when using experiments - debug
param_numTrees= 10
param_maxDepth= 15
param_impurity= "gini"

# get Environment bucket location
import os
ENV_BUCKET="s3a://demo-aws-2/datalake/"

try : 
  DL_s3bucket=os.environ["STORAGE"]+"/datalake/"
except KeyError: 
  DL_s3bucket=ENV_BUCKET


#track parameters in experiments
cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)


spark = SparkSession\
  .builder\
  .appName('wine-quality-training') \
  .config("spark.executor.memory","2g") \
  .config("spark.executor.cores","2") \
  .config("spark.executor.instances","3") \
  .config("spark.yarn.access.hadoopFileSystems",DL_s3bucket) \
  .getOrCreate()


# # Load the data 
# ### From File

# Define Schema : 
#     fixedacidity: numeric
#     volatileacidity: numeric
#     citricacid: numeric
#     residualsugar: numeric
#     chlorides: numeric
#     freesulfursioxide: numeric
#     totalsulfurdioxide: numeric
#     density: numeric
#     ph: numeric
#     sulphates: numeric
#     alcohol: numeric
#     quality: discrete

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
  StructField("quality", StringType(), True)
])


#set path to data
data_path = "file:///home/cdsw/data/"
data_file = "WineNewGBTDataSet.csv"
wine_data_raw = spark.read.csv(data_path+'/'+data_file, schema=schema,sep=';')

"""
# ### or from Hive 
wine_data_raw = spark.sql('''Select * from default.wineds_ext''')
"""

# Cleanup - Remove invalid data
wine_data = wine_data_raw.filter(wine_data_raw.quality != "1")


# # Build a classification model using MLLib
# # Step 1 Split dataset into train and validation 
# using randomSplit function of datasets
(trainingData, testData) = wine_data.randomSplit([0.7, 0.3])


# # Step 2 : split label and feature and encode for ML Lib
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

# split labels from data frame and encode in numerical format (requiered for Spark)
labelIndexer = StringIndexer(inputCol = 'quality', outputCol = 'label')

# group all features into single column (required for Spark)
featureIndexer = VectorAssembler(
    inputCols = ['fixedacidity', "volatileacidity", 
                 "citricacid","residualsugar", 
                 "chlorides", "freesulfurdioxide", 
                 "totalsulfurdioxide", "density", 
                 "ph", "sulphates", "alcohol"],
    outputCol = 'features')

# # Step 3 : 
# # Prepare Classifier ( Random Forest in this case )
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

#define classifier parameters
classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', 
                                    numTrees = param_numTrees, 
                                    maxDepth = param_maxDepth,  
                                    impurity = param_impurity)
#prepare pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, classifier])
#fit model
model = pipeline.fit(trainingData)

# # Step 4 Evaluate Model 
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Predict on Test Data
predictions = model.transform(testData)

#Evaluate
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
print("The AUROC is {:f} and the AUPR is {:f}".format(auroc, aupr))

#Track metric value in CDSW
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)

# # Save Model for deployement 
model.write().overwrite().save(DL_s3bucket+"tmp/models/spark")

#bring model back into project and tar it
!mv models/ models_OLD/
!mkdir models
!hdfs dfs -get $STORAGE/datalake/tmp/models/
!tar -cvf models/spark_rf.tar models/spark
!mv models/spark_rf.tar spark_rf.tar
!rm -rf models/

cdsw.track_file("spark_rf.tar")
spark.stop()

