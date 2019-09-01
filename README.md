# Wine Quality Prediction

This notebook demonstrates how to create a Gradient-boosted tree classifier 
to predict wine quality based on characteristics.

Used: Python, Spark, seaborn

Steps:

## 0. Prepare Env 
  Set Env variables for projet (Settings -> Engine -> Environmental Variables )
  * make sure that the following pyspark env variables point to the same python 
  interpreter (and that the interpreter exists)
   - PYSPARK3_PYTHON
   - PYSPARK_DRIVER_PYTHON
  
  ex : 
  - echo $PYSPARK3_PYTHON => /usr/local/bin/python3
  - ls /usr/local/bin/python3

## 1. Open a python3 session
Open Workbench => Editor : Workbench 
                  Engine Kernel : Python 3
   
## 2. Run setup 
-installs python libraries ( SKLearn )
-copies data to HDFS (/tmp/wine_pred)
-and create hive table ( wineDS_ext )
Note : Assumes that Hive is setup correctly and can talk to the cluster

Start a Python 3 session using Workbench editor
use either the session input window or terminal session to run setup 
* in session : !./setup/setup.sh
* in terminal : ./setup/setup.sh
 
## 3. Run descriptive anylisis of data
Start a python3 session using Workbensh editor
run "analyse.py"

## 4. create a model with pyspark 
Start a Python 3 session using Workbench editor
Run "fit.py: 

Note : using RandomForest from spark ML library 
params: 
  * numTrees ( number of trees to spawn )
  * maxDepth (depth of leafs for trees )

=> Copies model to root of projet as "spark_rf.tar"
  
## 4Bis. create a model with SKLearn and Jupyter Notebook 
Start a Python 3 session using Jupyter Notebook editor
=> user at least 2 CPU and 4GB of RAM 
=> Wait 15-20 secs for notebook to show up

In the "Jupyter Notebook" folder, run "fit_SKLEARN.ipynb"
Note : using RandomForest from Sklearn 
params: 
  * estimator ( number of trees to spawn )
  * maxDepth (depth of leafs for trees )


## 5. Create a model in the Model API
  * Give the model a name and description
  * Confifure runtime params 
    - Script: "model.py"
    - Function : "predict"
    - (optional) give example input :   
{"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}
{"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}
  
