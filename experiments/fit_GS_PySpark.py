import cdsw
from pyspark.sql import SparkSession
from pyspark.sql.types import *

"""
# ## Get parameters for experiments 
# ### Note uncomment for experiments
import ast #required to in order to parse arguements a lists
import sys
param_numTrees= ast.literal_eval(sys.argv[1])
param_maxDepth= ast.literal_eval(sys.argv[2])
param_impurity= "gini"
"""

# get Environment bucket location
import os
ENV_BUCKET="s3a://demo-aws-2/datalake/"

try : 
  DL_s3bucket=os.environ["ENV_BUCKET"]
except KeyError: 
  DL_s3bucket=ENV_BUCKET
  os.environ["ENV_BUCKET"] = ENV_BUCKET

#comment out when using experiments
param_numTrees = [10,15,20]
param_maxDepth = [5,10,15]
param_impurity = "gini"

#track parameters in experiments
cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)


# # Load Date
# ### Start PySpark Session
spark = SparkSession\
  .builder\
  .appName('wine-quality-analysis') \
  .config("spark.executor.memory","2g") \
  .config("spark.executor.cores","2") \
  .config("spark.executor.instances","3") \
  .config("spark.yarn.access.hadoopFileSystems",DL_s3bucket) \
  .getOrCreate()

# ### Load the data (From File )
# #### Define Schema
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
  StructField("quality", StringType(), True)])


#set path to data
#set path to data
data_path = "file:///home/cdsw/data/"
data_file = "WineNewGBTDataSet.csv"
wine_data_raw = spark.read.csv(data_path+'/'+data_file, schema=schema,sep=';')

"""
# ### Or Get data from Hive
wine_data_raw = spark.sql('''Select * from default.wineds_ext''')
"""

# ### Cleanup - Remove invalid data
wine_data = wine_data_raw.filter(wine_data_raw.quality != "1")


# # Build a classification model using MLLib
# ## Step 1 Split dataset into train and validation 
(trainingData, testData) = wine_data.randomSplit([0.7, 0.3])


# ## Step 2 : split label and feature and encode for ML Lib
# ### Label encoding and Feature indexor (assembles feature in format appropriate for Spark ML)
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

labelIndexer = StringIndexer(inputCol = 'quality', outputCol = 'label')
featureIndexer = VectorAssembler(
    inputCols = ["fixedacidity","volatileacidity","citricacid","residualsugar",
                 "chlorides","freesulfurdioxide", "totalsulfurdioxide", "density", 
                 "ph", "sulphates", "alcohol"],
    outputCol = 'features')


# ## Step 3 : Prepare Classifier ( Random Forest in this case )
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

# ## Grid Search - Spark ML way
# ### using Grid Search and cross validation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
RFclassifier = RandomForestClassifier(labelCol = 'label', 
                                      featuresCol = 'features', 
                                      impurity = param_impurity)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, RFclassifier])


# ### Define test configutations (to be evaluated in Grid)
paramGrid = ParamGridBuilder()\
   .addGrid(RFclassifier.maxDepth, param_maxDepth )\
   .addGrid(RFclassifier.numTrees, param_numTrees )\
   .build()

# ### Defing metric by wich the model will be evaluated
evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          parallelism=3, #number of models run in ||
                          numFolds=2) 

# ### fit model (note : returns the best model)
cvModel = crossval.fit(trainingData)

# ### show performande of runs 
print(cvModel.avgMetrics)

# # Evaluation of model performance on validation dataset
# #### prepare predictions on test dataset
predictions = cvModel.transform(testData)

evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr =  evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
print("The AUROC is {:f} and the AUPR is {:f}".format(auroc, aupr))


# ### Track metrics in Experiments view
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("numTrees",cvModel.bestModel.stages[2]._java_obj.getNumTrees())
cdsw.track_metric("maxDepth",cvModel.bestModel.stages[2]._java_obj.getMaxDepth())

# ## Save Model for deployement 
# #### Model is 3rd stage of the pipeline
cvModel.bestModel.write().overwrite().save(DL_s3bucket+"tmp/models/spark")

!mv models/ models_OLD/
!mkdir models
!hdfs dfs -get $ENV_BUCKET/tmp/models/
!tar -cvf models/spark_rf_grid.tar models/spark
!mv models/spark_rf_grid.tar spark_rf_grid.tar
!rm -rf models/

cdsw.track_file("spark_rf.tar")


spark.stop()
