from pyspark.sql import SparkSession
from pyspark.sql.types import *

path = '/tmp/wine_pred_hive'

print("Start Spark session :")
spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('wine-quality-create-table') \
  .getOrCreate()
print("started")

#Check file presence in HDFS
import subprocess
proc = subprocess.Popen(['hadoop', 'fs', '-test', '-e', path])
proc.communicate()

if proc.returncode != 0:
  print("{} does not exist".format(path))
else : 
  print("file found, creating table")
  statement = '''
  CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`wineDS_ext` (  
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
  '''.format(path)

  spark.sql(statement) 
  print("Show first 10 rows")
  spark.sql('''Select * from `default`.`wineDS_ext` limit 10 ''').show()

spark.stop()