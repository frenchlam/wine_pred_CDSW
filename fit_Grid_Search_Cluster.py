import cdsw
from pyspark.sql import SparkSession
from pyspark.sql.types import *

""" 
#- uncomment for experiments 
# # Experiments 
# Declare parameters
param_numTrees= list(sys.argv[1])
param_maxDepth= list(sys.argv[2])
param_impurity=sys.argv[3]

#Track Metrics in CDSW
cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)
"""

# comment out for experiments
param_numTrees = [10,15,20]
param_maxDepth = [5,10,15]
param_impurity= "gini"


# # Data Schema 
#
#     Fields:
#     fixedacidity: numeric
#     volatileacidity: numeric
#     citricacid: numeric
#     residualSugar: numeric
#     chlorides: numeric
#     freesulfurDioxide: numeric
#     totalsulfurDioxide: numeric
#     density: numeric
#     ph: numeric
#     sulphates: numeric
#     alcohol: numeric
#     quality: discrete


# # Read data from File
spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('wine-quality-build-model') \
  .getOrCreate()


# # Read from File
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


data_path = "/tmp/mlamairesse"
data_file = "WineNewGBTDataSet.csv"
wine_data_raw = spark.read.csv(data_path+'/'+data_file, schema=schema,sep=';')


"""
# # Read data from Hive
spark = SparkSession \
  .builder \
  .master('yarn') \
  .enableHiveSupport() \
  .appName('wine-quality-build-model') \
  .getOrCreate()


wine_data_raw = spark.sql('''Select * from default.wineds_ext''')
"""

# Cleanup - Remove invalid data
wine_data = wine_data_raw.filter(wine_data_raw.quality != "1")


# # Build a classification model using MLLib
# # Pipeline prep
# Label encoding and Feature indexor (assembles feature in format appropriate for Spark ML)
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

labelIndexer = StringIndexer(inputCol = 'quality', outputCol = 'label')
featureIndexer = VectorAssembler(
    inputCols = ['fixedacidity', 
                 "volatileacidity", 
                 "citricacid", 
                 "residualsugar", 
                 "chlorides", 
                 "freesulfurdioxide", 
                 "totalsulfurdioxide", 
                 "density", 
                 "ph", 
                 "sulphates", 
                 "alcohol"],
    outputCol = 'features')


# # Fit a RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

#Test/Train split
(trainingData, testData) = wine_data.randomSplit([0.7, 0.3])

"""
# # Grid Search - quick and dirty
results=[]
run=1
for tree in param_numTrees:
  for depth in param_maxDepth:
    #pipeline for model Training
    RFclassifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', 
                                      numTrees = tree, 
                                      maxDepth = depth,  
                                      impurity = param_impurity)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, RFclassifier])
    model = pipeline.fit(trainingData)

    #model Evaluation
    predictions = model.transform(testData)
    evaluator = BinaryClassificationEvaluator()
    auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    #write results
    dict_result = {}
    dict_result['numTrees'] = tree
    dict_result['maxDepth'] = depth
    dict_result['auroc'] = auroc
    results.append(dict_result)
    
    #output run perf
    print("run {:d} - auroc:{:f}".format(run,auroc))
    run += 1

# Get best run
best_run = results[0]
for result in results:
  if best_run['auroc'] < result['auroc']:
    best_run=result

# # Evaluation of FINAL model performance
predictions = model.transform(testData)
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
print("The AUROC is {:f} and the AUPR is {:f}".format(auroc, aupr))
"""  

# # Grid Search - Spark ML way
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
RFclassifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features',  
                                      impurity = param_impurity)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, RFclassifier])

#Define test configutation
paramGrid = ParamGridBuilder()\
   .addGrid(RFclassifier.maxDepth, param_maxDepth )\
   .addGrid(RFclassifier.numTrees, param_numTrees )\
   .build()

#metric by wich the model will be evaluated
evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          parallelism=2, #number of models run in ||
                          numFolds=2) 

cvModel = crossval.fit(trainingData)
# note : returns the best model.


# # Evaluation of model performance on validation dataset
evaluator = BinaryClassificationEvaluator()
predictions = cvModel.transform(testData)

auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
print("The AUROC is {:f} and the AUPR is {:f}".format(auroc, aupr))


"""#uncomment for Experiments
# # Save model to project
model.write().overwrite().save("models/spark")

!rm -r -f models/spark
!rm -r -f models/spark_rf.tar
!mkdir models
!hdfs dfs -get ./models/spark models/
!tar -cvf models/spark_rf.tar models/spark


# Track metrics in Experiments view
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_file("models/spark_rf.tar")
"""

spark.stop()

