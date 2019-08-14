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


# # Set up Spark Session and read data
spark = SparkSession \
  .builder \
  .master('yarn') \
  .enableHiveSupport() \
  .appName('wine-quality-build-model') \
  .getOrCreate()


# # Data Schema 
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

#Read from File
#data_path = "/tmp/mlamairesse"
#data_file = "WineNewGBTDataSet.csv"
#wine_data_raw = spark.read.csv(data_path+'/'+data_file, schema=schema,sep=';')

#Read from Hive
spark.sql('''Select * from default.wineds_ext''').show()


# Cleanup - Remove invalid data
wine_data = wine_data_raw.filter(wine_data_raw.Quality != "1")



# # Build a classification model using MLLib
# # Pipeline prep
# Label encoding and Feature indexor (assembles feature in format appropriate for Spark ML)
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

labelIndexer = StringIndexer(inputCol = 'Quality', outputCol = 'label')
featureIndexer = VectorAssembler(
    inputCols = ['fixedAcidity', 
                 "volatileAcidity", 
                 "citricAcid", 
                 "residualSugar", 
                 "chlorides", 
                 "freeSulfurDioxide", 
                 "totalSulfurDioxide", 
                 "density", 
                 "pH", 
                 "sulphates", 
                 "Alcohol"],
    outputCol = 'features')


# # Fit a RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

#Test/Train split
(trainingData, testData) = wine_data.randomSplit([0.7, 0.3])


# # Grid Search
results=[]
run=1
for tree in param_numTrees:
  for depth in param_maxDepth:
    #pipeline for model Training
    classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', 
                                      numTrees = tree, 
                                      maxDepth = depth,  
                                      impurity = param_impurity)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, classifier])
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
  

#
# Create final model
classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', 
                                    numTrees = best_run['numTrees'], 
                                    maxDepth = best_run['maxDepth'],  
                                    impurity = param_impurity)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, classifier])
model = pipeline.fit(trainingData)

# # Evaluation of model performance
predictions = model.transform(testData)
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
print("The AUROC is {:f} and the AUPR is {:f}".format(auroc, aupr))


# # Save model to project
model.write().overwrite().save("models/spark")

!rm -r -f models/spark
!rm -r -f models/spark_rf.tar
!mkdir models
!hdfs dfs -get ./models/spark models/
!tar -cvf models/spark_rf.tar models/spark

"""#uncomment for Experiments
# Track metrics in Experiments view
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_file("models/spark_rf.tar")
"""

spark.stop()

