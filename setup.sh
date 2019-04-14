hadoop fs -rm -R -f /tmp/mlamairesse
echo "deleted /tmp/mlamairesse"

hadoop fs -mkdir /tmp/mlamairesse 
echo "created /tmp/mlamairesse"

hadoop fs -put data/WineNewGBTDataSet.csv /tmp/mlamairesse
echo "copied data/WineNewGBTDataSet.csv -> hdfs://tmp/mlamairesse"

pip install seaborn