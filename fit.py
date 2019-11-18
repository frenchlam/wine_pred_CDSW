import cdsw
from pyspark.sql import SparkSession
from pyspark.sql.types import *

"""#uncomment for experiments
# # Get parameters for experiments
# Declare parameters 
param_numTrees= int(sys.argv[1])
param_maxDepth=int(sys.argv[2])
param_impurity=sys.argv[3]

#track parameters in experiments
cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)
"""

# Comment out when using experiments
param_numTrees= 10
param_maxDepth= 15
param_impurity= "gini"


spark = SparkSession\
  .builder\
  .appName('wine-quality-analysis')\
  .config("spark.executor.memory","2g")\
  .config("spark.executor.cores","2")\
  .config("spark.executor.instances","3")\
  .config("spark.hadoop.fs.s3a.metadatastore.impl","org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")\
  .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
  .config("spark.hadoop.fs.s3a.delegation.token.binding","")\
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
data_path = "s3a://mlamairesse/wine_dataset/data/"
data_file = "WineNewGBTDataSet.csv"
#data_path = "/tmp/wine_pred"
#data_file = "WineNewGBTDataSet.csv"
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
#model.write().overwrite().save("models/spark")

#bring model back into project and tar it
#!rm -rf models/
#!mkdir models
#!hdfs dfs -get ./models/spark models/
#!tar -cvf models/spark_rf.tar models/spark
#!rm -r -f models/spark
#!mv models/spark_rf.tar spark_rf.tar

cdsw.track_file("spark_rf.tar")

spark.stop()

