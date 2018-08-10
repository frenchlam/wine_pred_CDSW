# Wine Quality Prediction

Created by Antonin Bruneau (abruneau@cloudera.com)

This notebook demonstrates how to create a Gradient-boosted tree classifier 
to predict wine quality based on characteristics.

Used: Python, Spark, seaborn

Steps:
1. Open a terminal and run setup.sh
3. In your python session run annalyse.py
4. Run an experiment with fit.py
  * params: numTrees, maxDepth, impurity (e.g. 10 10 gini)
  * add to project spark_rf.tar
5. Run a model with model.py
  * input: {"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}
  * function: predict
6. When finished, run cleanup.sh in the terminal

Recommended Session Sizes: 2 CPU, 4 GB RAM
