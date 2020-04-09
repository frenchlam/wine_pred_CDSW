# Wine Quality Prediction

This notebook demonstrates how to create a Gradient-boosted tree classifier 
to predict wine quality based on characteristics.

Used: Python, Spark, seaborn

## 0. Prepare Env
#### FOR CML 
  1. Check-out the CML branch of the project 
  
    a. Open a session 
    Top right hand corner "Open Workbench" -> "Launch Session"
    
    b. Check-out the CML branch
    Once the session is launched run : 
    - `!git fetch origin`
    - `!git checkout CML`
    
    
  2. Create an Environment Variable describing the bucket linked to CDP environement of the workspace
  
    a. Find the bucket :
    In the CDP console go to [NAME OF YOUR ENV]-> "Data Lake"-> "Cloud Storage"and copy the bucket name
    
    b. Create an Environement Variable with containing the bucket name. 
    - Go to "Settings" -> "Engine" -> "Environmental Variables"  
    - Create an env variable called "ENV_BUCKET" and give it the path of your environement bucket  


#### FOR CDSW  
  Make sure that: 
  - The pyhton interpreter paths for PYSPARK are set AND 
  - The Driver and Executors use the same version of Python

  Use the following environment variables (Settings -> Engine -> Environmental Variables ): 
  - PYSPARK_DRIVER_PYTHON - For the driver (by default : /usr/local/bin/python3)
  - PYSPARK3_PYTHON - for executors (Usualy "python3 )

## (Optional) - run setup script 
The setup script creates a hive table that can be used to showcase Hive <=> CML integration  
- Open a session "Open Workbench" -> "Launch Session"
- Run setup : in session  `!./setup/setup.sh` - in terminal `./setup/setup.sh`


## Usage 
 
### 1. Run descriptive anylisis of data
Start a python3 session using Workbensh editor
run "analyse.py"

### 2. create a model  
#### With pyspark 
Start a Python 3 session using Workbench editor
Run "fit.py: 

Note : using RandomForest from spark ML library 
params: 
  * numTrees ( number of trees to spawn )
  * maxDepth (depth of leafs for trees )

=> Copies model to root of projet as "spark_rf.tar"
  
#### With Scikit Learn and Jupyter Notebook 
Start a Python 3 session using Jupyter Notebook editor
=> user at least 2 CPU and 4GB of RAM 

In the "Jupyter Notebook" folder, run "fit_SKLEARN.ipynb"
Note : using RandomForest from Sklearn 
params: 
  * estimator ( number of trees to spawn )
  * maxDepth (depth of leafs for trees )


## 3. Create a model in the Model API
  * Give the model a name and description
  * Confifure runtime params 
    - Script: "model.py"
    - Function : "predict"
    - (optional) give example input :   
{"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}
{"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}
  
