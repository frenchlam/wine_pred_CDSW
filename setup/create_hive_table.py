from pyspark.sql import SparkSession
from pyspark.sql.types import *

path_hive_labeled = '/tmp/wine_pred/'
path_hive_predict = '/tmp/wine_pred_hive/'

print("Start Spark session :")
spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('wine-quality-create-table') \
  .getOrCreate()
  
print("started")

#Check file presence in HDFS
import subprocess
proc = subprocess.Popen(['hadoop', 'fs', '-test', '-e', path_hive_labeled])
proc.communicate()

if proc.returncode != 0:
  print("{} does not exist".format(path))
else : 
  print("file found, creating table")
  
  # Create table of labeled data for training 
  spark.sql('''DROP TABLE IF EXISTS `default`.`{}`'''.format('wineds_ext'))
  statement = '''
  CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`{}` (  
  `fixedacidity` double ,  
  `volatileacidity` double ,  
  `citricacid` double ,  
  `residualsugar` double ,  
  `chlorides` double ,  
  `freesulfurdioxide` bigint ,  
  `totalsulfurdioxide` bigint ,  
  `density` double ,  
  `ph` double ,  
  `sulphates` double ,  
  `alcohol` double ,  
  `quality` string )  
  ROW FORMAT DELIMITED FIELDS TERMINATED BY ';' 
  STORED AS TextFile LOCATION '{}' 
  '''.format('wineds_ext', path_hive_labeled,  )
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
  `freesulfurdioxide` bigint ,  
  `totalsulfurdioxide` bigint ,  
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