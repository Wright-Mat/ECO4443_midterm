# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:20:46 2024

@author: mwrig
"""
import pandas as pd
from itertools import combinations
from statsmodels.api import add_constant 
import statsmodels.api as sm
from pandas import read_csv
import numpy as np 
import statsmodels as sm
from sklearn.preprocessing import PolynomialFeatures 
from numpy.random import seed 
from sklearn.model_selection import cross_validate 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing 
from sklearn.model_selection import cross_validate 



data = read_csv("D:/UCF_Classes/ECO_4433/mid_term_dataset.csv", delimiter =",")
data["percent_heated"] = data["home_size"]/data["parcel_size"]
data["price"] = data["price"]/1000
data_dummies = pd.get_dummies(data, columns=["year"], dtype=float)

# Scaled price by 1000 to reduced reported mse. Changed the year catagorical
# data to dummy variables, and made a new dataframe with the new variables. 
# Additionally we divided the home size variable by the parcel size to find
# the percent of the parcel used for housing. 

#Random Sampling 
np.random.seed(1234)
data = data.sample(len(data))


#Linear Poly


x_combos =[]
for n in range(1,11):
    combos = combinations(["year_2000", "year_2001", "year_2002", "year_2003", "year_2004", "year_2005", "age", "beds", "baths", "home_size", "pool", "dist_cbd", "dist_lakes", "x_coord", "y_coord", "percent_heated"], n)
    x_combos.extend(combos)
    
y = data_dummies["price"]
x = data_dummies[["year_2000", "year_2001", "year_2002", "year_2003", "year_2004", "year_2005","age", "beds", "baths", "home_size", "pool", "dist_cbd", "dist_lakes", "x_coord", "y_coord", "percent_heated"]]


# Scaling was done here to use the same data for OLS, Ridge, and Lasso 
x_scaled = preprocessing.scale(x)
x_scaled = pd.DataFrame(x_scaled, columns=("year_2000", "year_2001", "year_2002", "year_2003", "year_2004", "year_2005", "age", "beds", "baths", "home_size", "pool", "dist_cbd", "dist_lakes", "x_coord", "y_coord", "percent_heated"))

# Created empty dictionaries to hold results 
ols_mse = {}
lasso_mse = {}

# Something that came up while researching generating polyfeatures was dropping
# dummy variables in the combo list as multiplying or raising 1 or 0 doesn't
# add new data to the model. However I didn't consider this until much later 
# into the process which led to much longer processing times. If I were to 
# remove the dummy variables and generate combos could I then append those
# dummy variables into the combos list at set that equal to x like in line 68?

# Generating MSE
for n in range(0,len(x_combos)):
    combo_list = list(x_combos[n])
    x = x_scaled[combo_list] 
    poly = PolynomialFeatures(4)
    poly_x = poly.fit_transform(x)
    model = LinearRegression()
    cv_scores = cross_validate(model, poly_x, y, cv=10, scoring=("neg_mean_squared_error"))
    ols_mse[str(combo_list)] = np.mean(cv_scores["test_score"])
    
print("Outcomes from the Best Linear Regrssion Model:")
min_mse = abs(max(ols_mse.values()))
print("Minimum average test MSE:", min_mse.round(2))
for possibles, i in ols_mse.items():
    if i == -min_mse: 
        print("the combination of variables:", possibles)


# Lasso

lasso = Lasso(alpha = 2)
lasso.fit(x_scaled, y)
lasso_pred = lasso.predict(x_scaled)


print(pd.Series(lasso.coef_))

for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    x = x_scaled[combo_list]
    
    Lasso_cv_scores = cross_validate(Lasso(alpha=2), x_scaled, y, cv=10, scoring = ("neg_mean_squared_error"))

    lasso_mse[str(combo_list)] = np.mean(Lasso_cv_scores["test_score"])

print("Outcomes from Best Lasso Model:")
lasso_min_mse = abs(max(lasso_mse.values()))
print("Minimum Average Lasso Test MSE:", lasso_min_mse.round(3))
for possibles, r in lasso_mse.items():
    if r == lasso_min_mse:
        print("The Lasso Combination of Variables:", possibles)


#Ridge was dropped from testing as Linear Regression had seemed to provide the 
# best MSE however it was the mmost time consuming to calculate. So in order to 
# test higher polyfeatures in linear regression within the time I decided to 
# drop ridge -Matthew 


# validation 

data = read_csv("D:/UCF_Classes/ECO_4433/mid_term_dataset.csv", delimiter =",")
data["percent_heated"] = data["home_size"]/data["parcel_size"]
data["price"] = data["price"]/1000
data_dummies = pd.get_dummies(data, columns=["year"], dtype=float)

x = data_dummies[['year_2000', 'year_2001', 'year_2002', 'year_2005', 'pool', 'home_size', 'dist_cbd', 'dist_lakes', 'percent_heated']]
poly = PolynomialFeatures(4)
poly_x = poly.fit_transform(x)

x = data_dummies(poly.fit_transform(x), columns=poly.get_feature_names_out(x.columns))
y = data['sale_def']

best_model = sm.OLS(y, x)
results = best_model.fit()
print(results.summary())



