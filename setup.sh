hadoop fs -rm -R -f /tmp/wine_pred
echo "deleted /tmp/wine_pred"

hadoop fs -mkdir /tmp/wine_pred 
echo "created /tmp/wine_pred"

hadoop fs -put data/WineNewGBTDataSet.csv /tmp/wine_pred
echo "copied data/WineNewGBTDataSet.csv -> hdfs://tmp/wine_pred"

pip3 install seaborn sklearn