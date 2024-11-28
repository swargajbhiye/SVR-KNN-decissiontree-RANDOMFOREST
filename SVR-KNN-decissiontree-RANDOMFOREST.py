import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\emp_sal.csv")
 
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#svm model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=4,gamma='auto')
svr_regressor.fit(x,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

# knn model 
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=4,weights='uniform',p=2)
knn_reg_model.fit(x,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

#decission tree
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                     splitter='random',
                                     max_depth=4,
                                     min_samples_split=5) 
dt_regressor.fit(x,y)

dt_reg_pred = dt_regressor.predict([[6.5]])
print(dt_reg_pred) 

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=30,
                                     criterion='absolute_error',
                                     max_depth=4,
                                     min_samples_split=5,
                                     random_state=0)
rf_regressor.fit(x,y)

rf_reg_pred = rf_regressor.predict([[6.5]])
print(rf_reg_pred)
