hadoop fs -rm -R -f /tmp/wine_pred_hive
echo "deleted /tmp/wine_pred_hive"
hadoop fs -rm -R -f /tmp/wine_pred
echo "deleted /tmp/wine_pred"

hadoop fs -mkdir /tmp/wine_pred 
echo "created /tmp/wine_pred"
hadoop fs -mkdir /tmp/wine_pred_hive 
echo "created /tmp/wine_pred_hive"

hadoop fs -put /home/cdsw/data/WineNewGBTDataSet.csv /tmp/wine_pred
echo "copied /home/cdsw/data/WineNewGBTDataSet.csv -> hdfs://tmp/wine_pred"

echo "install seaborn and SKlearn"
pip3 install seaborn sklearn

echo "Create Hive Table"
spark-submit /home/cdsw/setup/create_hive_table.py