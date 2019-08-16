# # Load the data
# 
# We need to load data from a file in to a Spark DataFrame.
# Each row is a wine, and each column contains attributes of that wine.
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

# # Data Wrangling 

# ## 1. Read data ( From Spark )
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('wine-quality-analysis') \
  .getOrCreate()

### Data does not have schema, so we declare it manually 
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

data_path = "/tmp/wine_pred"
data_file = "WineNewGBTDataSet.csv"
wine_data_raw = spark.read.csv(data_path+'/'+data_file, schema=schema,sep=';')
wine_data_raw.show(3) 

# ### Basic DataFrame operations
# Dataframes essentially allow you to express sql-like statements. 
# We can filter, count, and so on. 
# Documentation - (http://spark.apache.org/docs/latest/sql-programming-guide.html#dataframe-operations)

"number of lines in dataset : {:d}".format(wine_data_raw.count())

# ### Spark SQL - manipulate data as if it was a table 
wine_data_raw.createOrReplaceTempView("wine")
spark.sql("select distinct(Quality), count(*) from wine GROUP BY Quality").show


# ### Remove invalid data
wine_data = wine_data_raw.filter(wine_data_raw.Quality != "1")
total_wines = wine_data.count()
good_wines = wine_data.filter(wine_data.Quality == 'Excellent').count()
good_wines = wine_data.filter(wine_data.Quality == 'Poor').count()

"Wines total: {}, Good : {}, Poor : {}".format(total_wines,good_wines,good_wines )


# # 2. Data visualisation ( using mathplotlib and Seaborn)
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

# ### Note: toPandas() => brings data localy !!!
sample_data = wine_data.sample(False, 0.5, 83).toPandas()
sample_data.transpose().head(21)

spark.stop()


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
