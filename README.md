# Wine Quality Prediction

This notebook demonstrates how to create a Gradient-boosted tree classifier 
to predict wine quality based on characteristics.

Used: Python, Spark, seaborn

Steps:
1. Open a python3 session
2. Open terminal and run setup.sh ( copies data to /tmp on HDFS )
Note : 
 - make sure that the following pyspark env variables point to the same python 
  interpreter (and that the interpreter exists)
  - PYSPARK3_PYTHON
  - PYSPARK_PYTHON
  - PYSPARK_DRIVER_PYTHON
  
  ex : 
  - echo $PYSPARK3_PYTHON => /usr/local/bin/python3
  - ls /usr/local/bin/python3

If not - Set these variable with correct interpreter path in projet env
=> Settings => Engine 

3. In python3 session run analyse.py

4. Run an experiment with fit.py
  * params: numTrees, maxDepth, impurity (e.g. 10 10 gini)
  * add to project spark_rf.tar
  
5. Run a model with model.py
  * input: 
     {"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"} => Bad
     {"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"} => Good
  
  * function: predict
  
6. When finished, run cleanup.sh in the terminal

Recommended Session Sizes: 2 CPU, 4 GB RAM
