!pip3 install sklearn

import cdsw
import numpy as np
import pandas as pd


#- uncomment for experiments 
# # Experiments 
# ### Declare parameters
import ast #required to in order to parse arguements a lists
import sys
param_numTrees= ast.literal_eval(sys.argv[1])
param_maxDepth= ast.literal_eval(sys.argv[2])

'''
# comment out when using experiments
param_numTrees = [70,80,90]
param_maxDepth = [5,10,15]
'''


# ### Load the data (From File )
input_file = "data/WineNewGBTDataSet.csv"
col_Names=["fixedAcidity",
    "volatileAcidity",
    "citricAcid",
    "residualSugar",
    "chlorides",
    "freeSulfurDioxide",
    "totalSulfurDioxide",
    "density",
    "pH",
    "sulphates",
    "Alcohol",
    "Quality"]


wine_df = pd.read_csv(input_file,sep=";",header=None, names=col_Names)
wine_df.head()


# ### Cleanup - Remove invalid data
wine_df.Quality.replace('1',"Excellent",inplace=True)
wine_df.describe()


# # Build a classification model using MLLib
# ## Step 1 Encode labels and split dataset into train and validation 

# ### encode labels 
wine_df.Quality = pd.Categorical(wine_df.Quality)
wine_df['Label'] = wine_df.Quality.cat.codes
wine_df.head()

# ### Split Test/Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine_df.iloc[:,:11],
                                                    wine_df['Label'], 
                                                    test_size=0.2, 
                                                    random_state=30)


# ## Step 2 : Prepare Classifier ( Random Forest in this case )
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ### parameters for grid search
rfc = RandomForestClassifier(random_state=10, n_jobs=-1)

GS_params = { 
    'n_estimators': param_numTrees,
    'max_depth' : param_maxDepth
}

# ### Prepare Cross Validation Grid Search
CV_rfc = GridSearchCV(estimator=rfc, 
                      param_grid=GS_params, 
                      cv= 3)

# ### Fit Model
CV_rfc.fit(X_train, y_train)

# ### Show Best Parameters 
print(CV_rfc.best_params_)


# ### Prepare final Model
rfc_final= RandomForestClassifier(n_estimators=CV_rfc.best_params_['n_estimators'] , 
                                  max_depth=CV_rfc.best_params_['max_depth'], 
                                  random_state=10, 
                                  n_jobs=-1)
rfc_final.fit(X_train, y_train)
y_true, y_pred = y_test, rfc_final.predict(X_test)



# ## Step 3 : Evaluate Model
# ### Evaluation metrics
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
auroc =  roc_auc_score(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

print("The AUROC is {:f} and the Average Precision is {:f}".format(auroc, average_precision))

# ### Track Metrics in CDSW
cdsw.track_metric("numTrees", CV_rfc.best_params_['n_estimators'])
cdsw.track_metric("maxDepth", CV_rfc.best_params_['max_depth'])
cdsw.track_metric("auroc", auroc)
