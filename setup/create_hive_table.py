import os 
#DL_s3bucket = os.environ["DL_S3_BUCKET"]
DL_s3bucket="s3a://mlam-cdp-bucket/mlam-dl/"
path_hive_labeled = DL_s3bucket+'tmp/wine_pred/'
path_hive_predict = DL_s3bucket+'tmp/wine_pred_hive/'


from pyspark.sql import SparkSession
from pyspark.sql.types import *

print("Start Spark session :")
spark = SparkSession \
  .builder \
  .appName('wine-quality-create-table') \
  .config("spark.yarn.access.hadoopFileSystems",DL_s3bucket) \
  .getOrCreate()
  
print("started")
print("read data")

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

data_path = "file:///home/cdsw/data"
data_file = "WineNewGBTDataSet.csv"
wine_data_raw = spark.read.csv(data_path+'/'+data_file, schema=schema,sep=';')
wine_data_raw.show(3) 

wine_data_raw.write.mode('overwrite').parquet(path_hive_labeled)



#Read Data and save to S3
#Create table of labeled data for training 
spark.sql('''DROP TABLE IF EXISTS `default`.`{}`'''.format('wineds_ext'))
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`wineds_ext` (  
`fixedacidity` double ,  
`volatileacidity` double ,  
`citricacid` double ,  
`residualsugar` double ,  
`chlorides` double ,  
`freesulfurdioxide` double ,  
`totalsulfurdioxide` double ,  
`density` double ,  
`ph` double ,  
`sulphates` double ,  
`alcohol` double ,  
`quality` string )  
STORED AS parquet 
LOCATION '{}' 
'''.format(path_hive_labeled)
spark.sql(statement) 

print("First 5 rows of labeled data - hive")
spark.sql('''SELECT * FROM wineDS_ext LIMIT 5''').show()

#Create table of unlabeled data 
#Note using same data as training... 

wine_df = spark.sql(''' SELECT
`fixedacidity`, `volatileacidity`,  
`citricacid`, `residualsugar` ,  
`chlorides` , `freesulfurdioxide`,  
`totalsulfurdioxide` ,`density` ,  
`ph`, `sulphates`,`alcohol`
FROM wineDS_ext''')

# Write in Parquet
wine_df.write.mode('overwrite').parquet(path_hive_predict)

#create external table
spark.sql('''DROP TABLE IF EXISTS `default`.`{}`'''.format('wineds_ext_nolabel'))

statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`{}` (  
`fixedacidity` double ,  
`volatileacidity` double ,  
`citricacid` double ,  
`residualsugar` double ,  
`chlorides` double ,  
`freesulfurdioxide` double ,  
`totalsulfurdioxide` double ,  
`density` double ,  
`ph` double ,  
`sulphates` double ,  
`alcohol` double )  
STORED AS PARQUET 
LOCATION '{}' 
'''.format( 'wineds_ext_nolabel', path_hive_predict, )
spark.sql(statement)

print("First 5 rows of labeled data - hive")
spark.sql('''SELECT * FROM wineds_ext_nolabel LIMIT 5''').show()
  

spark.stop()