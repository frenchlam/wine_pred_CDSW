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
#     
#     

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *

conf = SparkConf().setAppName("wine-quality-prediction")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

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

wine_data_raw = sqlContext.read.format('com.databricks.spark.csv').option("delimiter",";").load('/tmp/WineNewGBTDataSet.csv', schema = schema)
wine_data_raw.head()

# # Basic DataFrame operations
# 
# Dataframes essentially allow you to express sql-like statements. We can filter, count, and so on. [DataFrame Operations documentation.](http://spark.apache.org/docs/latest/sql-programming-guide.html#dataframe-operations)

count = wine_data_raw.count()

wine_data_raw.createOrReplaceTempView("wine")


qualities = sqlContext.sql("select distinct(Quality), count(*) from wine GROUP BY Quality")
qualities.show()


# Remove unvalid data
wine_data = wine_data_raw.filter(wine_data_raw.Quality != "1")

good_wines = wine_data.filter(wine_data.Quality == 'Excellent').count()
poor_wines = wine_data.filter(wine_data.Quality == 'Poor').count()

"total: %d, Good ones: %d, Poor ones: %d " % (count, good_wines, poor_wines)



# # Seaborn
# 
# Seaborn is a library for statistical visualization that is built on matplotlib.
# 
# Great support for:
# * plotting distributions
# * regression analyses
# * plotting with categorical splitting
# 
# 
# ## Feature Visualization
# 
# The data vizualization workflow for large data sets is usually:
# 
# * Sample data so it fits in memory on a single machine.
# * Examine single variable distributions.
# * Examine joint distributions and correlations.
# * Look for other types of relationships.
# 
# [DataFrame#sample() documentation](http://people.apache.org/~pwendell/spark-releases/spark-1.5.0-rc1-docs/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample)

sample_data = wine_data.sample(False, 0.5, 83).toPandas()
sample_data.transpose().head(21)
# 
# ## Feature Distributions
# 
# We want to examine the distribution of our features, so start with them one at a time.
# 
# Seaborn has a standard function called [dist()](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.distplot.html#seaborn.distplot) that allows us to easily examine the distribution of a column of a pandas dataframe or a numpy array.

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

sb.distplot(sample_data['Alcohol'], kde=False)

# We can examine feature differences in the distribution of our features when we condition (split) our data.
# 
# [BoxPlot docs](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.boxplot.html)


sb.boxplot(x="Quality", y="Alcohol", data=sample_data)

# ## Joint Distributions
# 
# Looking at joint distributions of data can also tell us a lot, particularly about redundant features. [Seaborn's PairPlot](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot) let's us look at joint distributions for many variables at once.


example_numeric_data = sample_data[["fixedAcidity", "volatileAcidity",
                                       "citricAcid", "residualSugar", "Quality"]]
sb.pairplot(example_numeric_data, hue="Quality")

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

# ## Model Training
# 
# We can now define our classifier and pipeline. With this done, we can split our labeled data in train and test sets and fit a model.
# 
# To train the Gradient-boosted tree, give it the feature vector column and the label column. 
# 
# Pipeline is defined by stages. Index plan column, label column, create vec

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=20)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

(trainingData, testData) = wine_data.randomSplit([0.7, 0.3])

model = pipeline.fit(trainingData)

# ## Model Evaluation

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(15)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
predictions
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)


# # Fit a RandomForestClassifier
# 
# Fit a random forest classifier to the data. Try experimenting with different values of the `maxDepth`, `numTrees`, and `entropy` parameters to see which gives the best classification performance. Do the settings that give the best classification performance on the training set also give the best classification performance on the test set?
# 
# Have a look at the [documentation](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier]documentation).


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
 
classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', numTrees= 10)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, classifier])
model = pipeline.fit(trainingData)

predictions = model.transform(testData)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
"The AUROC is %s" % (auroc)


# If you need to inspect the predictions...

#predictions.select('label','prediction','probability').filter('prediction = 1').show(50)
predictions.select('label','prediction','probability').filter('prediction = 1').toPandas().head(50)

