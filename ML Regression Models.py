#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roger H Hayden III
#Udemy - The Complete Machine Learning Course with Python
#Regression Modeling techniques
#4/13/22


# In[212]:


#General Imports
import pandas as pd
import numpy as np
import sklearn

#Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns

#Linear Regression
from sklearn.linear_model import LinearRegression

#RANSAC inlier regression
from sklearn.linear_model import RANSACRegressor

#Train/test split
from sklearn.model_selection import train_test_split

#MSE
from sklearn.metrics import mean_squared_error

#R-squared Values
from sklearn.metrics import r2_score

#Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Standardize
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Ridge Regression
from sklearn.linear_model import Ridge

#Lasso Regression
from sklearn.linear_model import Lasso

#Elastic New Regression
from sklearn.linear_model import ElasticNet

#Polynomoial Regression
from sklearn.preprocessing import PolynomialFeatures

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

#AdaBoost Regression
from sklearn.ensemble import AdaBoostRegressor

#Data Preprocessing
from sklearn import preprocessing

#One Hot/One of K Encoding - Binary Responses
from sklearn.preprocessing import OneHotEncoder

#Used within Variance-Bias Trade off
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

#Used within the Validation Curve portion
from sklearn.model_selection import validation_curve

#Used within Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#Used with Cross Validation
from sklearn import svm

#K-folds
from sklearn.model_selection import KFold

#Stratified K-fold
from sklearn.model_selection import StratifiedKFold

#Use PCA within Stratified K-fold portion
from sklearn.decomposition import PCA


# In[4]:


print(pd.__version__)
print(np.__version__)
import sys
print(sys.version)
print(sklearn.__version__)


# Import Data and View it

# In[5]:


#Read in Data
df = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\Boston_Housing.csv', delim_whitespace = True, header = None)
print(df)


# In[6]:


df.head()


# Add Headers

# In[7]:


col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[8]:


df.columns = col_name
df.head()


# ==================================================================================================================

# Exploratory Data Analysis (EDA)

# In[9]:


#Producing general statistical information for each column
df.describe()


# In[10]:


sns.pairplot(df, size = 1.5);
plt.show()


# In[11]:


col_study = ['ZN', 'INDUS', 'NOX', 'RM']


# In[12]:


sns.pairplot(df[col_study], size = 2.5);
plt.show()


# ==================================================================================================================

# Correlation Analysis and Feature Selection

# In[13]:


#How each column relates to eachother and what we are trying to predict
df.corr()


# In[14]:


#We want to use the variables that have the highest correlation to each other. 
#Having the relationship is important for predictions

#Creating a Heat map to show the above table with coloring
plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[15]:


#Creating a Heat map with the main 4 variables and the median value
plt.figure(figsize = (16, 10))
sns.heatmap(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'MEDV']].corr(), annot = True)
plt.show()


# ==================================================================================================================

# Linear Regression Using Scikit-Learn

# In[16]:


df.head()


# In[17]:


#Using RM as our indepentent variable or our predictor
X = df['RM'].values.reshape(-1, 1)


# In[18]:


#Using MEDV as our dependent variable or what we are trying to predict
y = df['MEDV'].values


# In[19]:


#Set the model type
LinearReg_Model = LinearRegression()


# In[20]:


#Pass the underlying data through the machine learning model 
#Pass through the features and target variable - Model Fitting
LinearReg_Model.fit(X, y)


# In[21]:


#Model Coefficient
LinearReg_Model.coef_


# In[22]:


#Model Intercept
LinearReg_Model.intercept_


# In[23]:


#Appears as the number of rooms increases so does the median value of the home
#Obviously not all points fall exactly in line and there are some situations where this is not true

#Plotting X vs. y with the seaborn libarary
#Attempt to visualize the relationship here
plt.figure(figsize = (12, 10));
sns.regplot(X, y);
plt.xlabel('Average Number of Rooms per Home')
plt.ylabel('Median Value of Owned Homes in Thousands')
plt.show()


# In[24]:


#Adding Distributions along with the scatter plot
sns.jointplot(x = 'RM', y = 'MEDV', data = df, kind = 'reg', height = 10);
plt.show();


# In[25]:


#Create a Prediction for when the room value is 5
#Predicting House price to be around 10 when there are 5 rooms
LinearReg_Model.predict(np.array([5]).reshape(-1, 1))


# In[26]:


#Create a Prediction for when the room value is 7
#Predicting House price to be around 29 when there are 7 rooms
LinearReg_Model.predict(np.array([7]).reshape(-1, 1))


# ==================================================================================================================

# Five Steps to the Machine Learning Process from Jacob T. VanderPlas

# 1. Choose a class of model by importing the appropriate estimator class from Scikit-Learn
# 
# 2. Choose model hyperparameters by instantiating this class with desired values
# 
# 3. Arrange data into a features matrix and target vector folllowing the discussion from before
# 
# 4. Fit the model to your data by calling the fit() method
# 
# 5. Apply the model to new data
#     - For supervised learning, often predict labels for unkown data using the predict() method
#     - For unsupervised learning, we often transform or infer properties of the data using the transform() or predict() method

# ==================================================================================================================

# Random Sample Consensus (RANSAC) Algorithm - type of Robust Algorithm

# In[27]:


X = df['RM'].values.reshape(-1, 1)
y = df['MEDV'].values


# In[28]:


RANSAC_Model = RANSACRegressor()


# In[29]:


#Preforms fitting with the inliers
RANSAC_Model.fit(X, y)


# In[30]:


#Inlier and outlier masks are made within the RANSAC model

#The inner mask or inlier is the data within the suspected range
#The outter mask or outlier is just everything that is not an inlier
inlier_mask = RANSAC_Model.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# In[31]:


#Pulling points from 3 to 9 by 1 spot
np.arange(3, 10, 1)


# In[32]:


#Storing an array into line_X then creating a prediction for line_y with the RANSAC Algorithm based off of line_X
#Performs Prediction with inliers
line_X = np.arange(3, 10, 1)
line_y_RANSAC = RANSAC_Model.predict(line_X.reshape(-1, 1))


# In[33]:


#Plotting the data passed through the RANSAC Model with the inliers as blue and the outliers as brown
sns.set(style = 'darkgrid', context = 'notebook')
plt.figure(figsize = (12, 8))
plt.scatter(X[inlier_mask], y[inlier_mask], c = 'blue', marker = 'o', label = 'Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c = 'brown', marker = 's', label = 'Outliers')
plt.plot(line_X, line_y_RANSAC, color = 'red')
plt.xlabel('Average Number of Rooms per Home')
plt.ylabel('Median Value of Owned Homes in Thousands')
plt.legend(loc = 'upper left')
plt.show()


# ==================================================================================================================

# Evaluate Regression Model Performance

# In[34]:


df.head()


# In[35]:


#We are not as interested in the intrinsic prediction capabilities but how the extrinsic variables work
#This time we use all of the data for the features without the MEDV column
X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values
y = df['MEDV'].values


# In[36]:


#Train/Test Split
#The Random state is important to keep the data generating at the same point
#Select test data size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[37]:


LRM = LinearRegression()


# In[38]:


#Produce the model fit with the training data
LRM.fit(X_train, y_train)


# In[39]:


#Create predictions
y_train_pred = LRM.predict(X_train)
y_test_pred = LRM.predict(X_test)


# Method 1: Residual Analysis

# In[40]:


plt.figure(figsize = (12, 8))
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker = 'o', label = 'Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'orange', marker = '*', label = 'Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'k')
plt.xlim([-10, 50])
plt.show()


# Method 2: Mean Squared Error (MSE)

# In[41]:


#For MSE, the lower the number the better the model is working
mean_squared_error(y_train, y_train_pred)


# In[42]:


mean_squared_error(y_test, y_test_pred)


# Method 3: Coefficient of Determination or R^2

# In[43]:


#A higher R^2 the better the model is working
r2_score(y_train, y_train_pred)


# In[44]:


r2_score(y_test, y_test_pred)


# ==================================================================================================================

# Multiple Regression

# In[45]:


X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values
y = df['MEDV'].values


# Statsmodels

# In[46]:


#Add a constant term to allow statsmodel api to calulate the bias/intercepts
X_constant = sm.add_constant(X)


# In[47]:


pd.DataFrame(X_constant)


# In[48]:


StatsReg = sm.OLS(y, X_constant)


# In[49]:


Reg = StatsReg.fit()


# In[50]:


Reg.summary()


# Statsmodels Formula API

# In[51]:


form_lr = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data = df)
mlr = form_lr.fit()


# In[52]:


mlr.summary()


# ====================================================================================================================

# Review Collinearity

# Correlation Matrix

# In[53]:


corr_matrix = df.corr()
corr_matrix


# In[54]:


#Create a mask to make correlation matrix easier to read
corr_matrix[np.abs(corr_matrix) < 0.6] = 0
corr_matrix


# In[55]:


#All of the 0 values are not actually 0, they are between -0.6 and 0.6
plt.figure(figsize = (16, 10))
sns.heatmap(corr_matrix, annot = True, cmap = 'YlGnBu')
plt.show()


# Detecting Collinearity with Eigenvectors

# In[56]:


eigenvalues, eigenvectors = np.linalg.eig(df.corr())


# In[57]:


pd.Series(eigenvalues).sort_values()


# In[58]:


#Highest values are the worst or trouble makers
np.abs(pd.Series(eigenvectors[:, 8])).sort_values(ascending = False)


# In[59]:


#These are the features causing the multi-collinearity problem
print(df.columns[2], df.columns[8], df.columns[9])


# =================================================================================================================

# Revisiting Feature Importance and Extractions

# In[60]:


df.head()


# In[61]:


#Scaling is quite large compared to other variables
plt.hist(df['TAX'])


# In[62]:


#Scaling is quite large compared to other variables
plt.hist(df['NOX'])


# Standardize Variable to Identify Key Features

# In[63]:


StandardizeModel = LinearRegression()


# In[64]:


StandardizeModel.fit(X, y)


# In[65]:


StandardizeResult = pd.DataFrame(list(zip(StandardizeModel.coef_, df.columns)), columns = ['coefficient', 'name']).set_index('name')
np.abs(StandardizeResult).sort_values(by = 'coefficient', ascending = False)


# In[66]:


scaler = StandardScaler()
standard_coefficient_linear_reg = make_pipeline(scaler, StandardizeModel)


# In[67]:


standard_coefficient_linear_reg.fit(X, y)
result = pd.DataFrame(list(zip(standard_coefficient_linear_reg.steps[1][1].coef_, df.columns)), columns = ['coefficient', 'name']).set_index('name')
np.abs(result).sort_values(by = 'coefficient', ascending = False)


# Use R-squared to Identify Key Features

# In[68]:


linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data = df)
benchmark = linear_reg.fit()
r2_score(y, benchmark.predict(df))


# In[69]:


#Without LSTAT
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B', data = df)
without_LSTAT = linear_reg.fit()
r2_score(y, without_LSTAT.predict(df))


# In[70]:


#Without AGE
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', data = df)
without_AGE = linear_reg.fit()
r2_score(y, without_AGE.predict(df))


# =================================================================================================================

# Regularized Regression

# Linear Regression

# In[71]:


#Simple Linear Regression
np.random.seed(42)
n_samples = 100
rng = np.random.randn(n_samples) * 10
y_gen = 0.5 * rng + 2 * np.random.randn(n_samples)

lr = LinearRegression()
lr.fit(rng.reshape(-1, 1), y_gen)
model_pred = lr.predict(rng.reshape(-1, 1))

plt.figure(figsize = (10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, model_pred)
print("Coefficient Estimate: ", lr.coef_)


# In[72]:


#Linear Regression with added outliers
idx = rng.argmax()
y_gen[idx] = 200
idx = rng.argmin()
y_gen[idx] = -200


# In[73]:


plt.figure(figsize = (10, 8))
plt.scatter(rng, y_gen)

o_lr = LinearRegression(normalize = True)
o_lr.fit(rng.reshape(-1, 1), y_gen)
o_model_pred = o_lr.predict(rng.reshape(-1, 1))

plt.scatter(rng, y_gen)
plt.plot(rng, o_model_pred)
print("Coefficient Estimate: ", o_lr.coef_)


# Ridge Regression

# In[74]:


#Should be used what you can't zero out coefficients

#Improves the Coefficient Estimate some from before when we added the outliers
#Ridge Regression restricts how much the line will change based off of potential outliers
ridge_model = Ridge(alpha = 0.5, normalize = True)
ridge_model.fit(rng.reshape(-1, 1), y_gen)
ridge_model_pred = ridge_model.predict(rng.reshape(-1, 1))

plt.figure(figsize = (10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, ridge_model_pred)
print("Coefficient Estimate: ", ridge_model.coef_)


# Lasso Regression

# In[75]:


#Should be used for parameter shrinkage and variable selection

#Lasso is very similar to Ridge Regression
lasso_model = Lasso(alpha = 0.4, normalize = True)
lasso_model.fit(rng.reshape(-1, 1), y_gen)
lasso_model_pred = lasso_model.predict(rng.reshape(-1, 1))

plt.figure(figsize = (10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, lasso_model_pred)
print("Coefficient Estimate: ", lasso_model.coef_)


# Elastic Net Regression

# In[76]:


#Should be used if some covariates are highly correlated

#Elastic Net Regression appears to be the best here for minimizing the weight of the outliers
en_model = ElasticNet(alpha = 0.2, normalize = True)
en_model.fit(rng.reshape(-1, 1), y_gen)
en_model_pred = en_model.predict(rng.reshape(-1, 1))

plt.figure(figsize = (10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, en_model_pred)
print("Coefficient Estimate: ", en_model.coef_)


# ================================================================================================================

# Polynomial Regression

# In[77]:


np.random.seed(42)
nsamples = 100

X = np.linspace(0, 10, 100)
rng = np.random.randn(n_samples) * 100

y = X ** 3 + rng + 100

plt.figure(figsize = (10, 8))
plt.scatter(X, y)


# Attempted Linear Regression

# In[78]:


lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)
model_pred = lr.predict(X.reshape(-1, 1))

plt.figure(figsize = (10, 8))
plt.scatter(X, y)
plt.plot(X, model_pred)
print(r2_score(y, model_pred))


# Polynomial Regression

# In[79]:


poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))


# In[80]:


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))
y_pred = lin_reg_2.predict(X_poly)


# In[81]:


plt.figure(figsize = (10, 8))
plt.scatter(X, y)
plt.plot(X, y_pred)
print(r2_score(y, y_pred))


# ========================================================================================================================

# Boston Housing Dataset

# In[82]:


X_boston = df['DIS'].values
y_boston = df['NOX'].values


# In[83]:


plt.figure(figsize = (12, 8))
plt.scatter(X_boston, y_boston)


# Linear Regression

# In[84]:


lr = LinearRegression()
lr.fit(X_boston.reshape(-1, 1), y_boston)
model_pred = lr.predict(X_boston.reshape(-1, 1))

plt.figure(figsize = (12, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_boston, model_pred)
print("R-squared Value = {:.2f}".format(r2_score(y_boston, model_pred)))


# Quadratic Regression

# In[85]:


poly_reg = PolynomialFeatures(degree = 2)
X_poly_b = poly_reg.fit_transform(X_boston.reshape(-1, 1))
lin_reg_2 = LinearRegression()


# In[86]:


lin_reg_2.fit(X_poly_b, y_boston)


# In[87]:


X_fit = np.arange(X_boston.min(), X_boston.max(), 1)[:, np.newaxis]


# In[88]:


X_fit


# In[89]:


y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_fit.reshape(-1, 1)))


# In[90]:


plt.figure(figsize = (10, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)
print("R-squared Value = {:.2f}".format(r2_score(y_boston, lin_reg_2.predict(X_poly_b))))


# Cubic

# In[91]:


poly_reg = PolynomialFeatures(degree = 3)
X_poly_b = poly_reg.fit_transform(X_boston.reshape(-1, 1))
lin_reg_3 = LinearRegression()


# In[92]:


lin_reg_3.fit(X_poly_b, y_boston)


# In[93]:


X_fit - np.arange(X_boston.min(), X_boston.max(), 1)[:, np.newaxis]


# In[94]:


y_pred = lin_reg_3.predict(poly_reg.fit_transform(X_fit.reshape(-1, 1)))


# In[95]:


plt.figure(figsize = (10, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)
print("R-squared Value = {:.2f}".format(r2_score(y_boston, lin_reg_3.predict(X_poly_b))))


# ================================================================================================================

# Nonlinear Relationships

# Brief Introduction to Decision Trees

# In[96]:


X = df[['LSTAT']].values
y = df[['MEDV']].values


# In[97]:


tree = DecisionTreeRegressor(max_depth = 5)


# In[98]:


tree.fit(X, y)


# In[99]:


sort_idx = X.flatten().argsort()


# In[100]:


plt.figure(figsize = (10, 8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color = 'k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')


# In[101]:


#Using a mad depth of 5 overfit the data so we will try 2
tree = DecisionTreeRegressor(max_depth = 2)


# In[102]:


tree.fit(X, y)


# In[103]:


sort_idx = X.flatten().argsort()


# In[104]:


plt.figure(figsize = (10, 8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color = 'k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')


# Brief Introduction to Random Forest

# In[105]:


X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values
y = df['MEDV'].values


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[107]:


RF = RandomForestRegressor(n_estimators = 500, criterion = 'mse', random_state = 42, n_jobs = -1)


# In[108]:


RF.fit(X_train, y_train)


# In[109]:


y_train_pred = RF.predict(X_train)


# In[110]:


y_test_pred = RF.predict(X_test)


# In[111]:


print("MSE train: {0:.4f}, test: {0:.4f}".     format(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))


# In[112]:


print("R-squared train: {0:.4f}, test: {0:.4f}".     format(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# Brief Introduction to AdaBoost

# In[113]:


ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 500, random_state = 42)


# In[114]:


ada.fit(X_train, y_train)


# In[115]:


y_train_pred = ada.predict(X_train)


# In[116]:


y_test_pred = ada.predict(X_test)


# In[117]:


print("MSE train: {0:.4f}, test: {1:.4f}".     format(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))


# In[118]:


print("R-squared train: {0:.4f}, test: {1:.4f}".     format(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# ===================================================================================================================

# Revisit Feature Importance

# Using AdaBoost

# In[119]:


ada.feature_importances_


# In[120]:


df.columns


# In[121]:


df2 = df.iloc[:, 0:13]
df2.columns


# In[122]:


result = pd.DataFrame(ada.feature_importances_, df2.columns)
result.columns = ['feature']


# In[123]:


result.sort_values(by = 'feature', ascending = False)


# In[124]:


result.sort_values(by = 'feature', ascending = False).plot(kind = 'bar')


# Using Random Forest

# In[125]:


RF.feature_importances_


# In[126]:


result = pd.DataFrame(RF.feature_importances_, df2.columns)
result.columns = ['feature']
result.sort_values(by = 'feature', ascending = False).plot(kind = 'bar')


# ===================================================================================================================

# Data Preprocessing

# In[127]:


X = df[['LSTAT']].values
y = df[['MEDV']].values


# Without Preprocessing

# In[128]:


alpha = 0.0001
w_ = np.zeros(1 + X.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X, w_[1:] + w_[0])
    errors = (y - y_pred)
    
    w_[1:] += alpha * X.T.dot(errors) #Producing Error Here
    w_[0] += alpha * errors.sum()
    
    cost = (errors ** 2).sum() / 2.0
    cost_.append(cost)
    
plt.figure(figsize = (8, 6))
plt.plot(range(1, n_ + 1), cost_)
plt.xlabel('Epoch')
plt.ylabel('SSE')


# With Preprocessing

# In[129]:


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

alpha = 0.0001
w_ = np.zeros(1 + X_std.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X_std, w_[1:] + w_[0])
    errors = (y_std - y_pred)
    
    w_[1:] += alpha * X_std.T.dot(errors) #Producing Error Here
    w_[0] += alpha * errors.sum()
    
    cost = (errors ** 2).sum() / 2.0
    cost_.append(cost)
    
plt.figure(figsize = (8, 6))
plt.plot(range(1, n_ + 1), cost_)
plt.xlabel('Epoch')
plt.ylabel('SSE')


# Data Preprocessing

# In[130]:


#Used for:
    #Standardization/Mean Removal
    #Min-Max or Scaling Features to a Range
    #Normalization
    #Binarization


# In[131]:


X_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])


# In[132]:


X_train.mean(axis = 0)


# Standardization/Mean Removal/Variance Scaling

# In[133]:


#Remove mean and center data on 0 to remove bias
#Preprocess only the training data
#Only transform the test data


# In[134]:


X_scaled = preprocessing.scale(X_train)
X_scaled


# In[135]:


#When you preprocess mean values are all 0
X_scaled.mean(axis = 0)


# In[136]:


#When you preprocess standard deviation values are all 1
X_scaled.std(axis = 0)


# In[137]:


scaler = preprocessing.StandardScaler().fit(X_train)
scaler


# In[138]:


scaler.mean_


# In[139]:


scaler.scale_


# In[140]:


scaler.transform(X_train)


# In[141]:


plt.figure(figsize = (8, 6))
plt.hist(X_train)


# In[142]:


X_test = [[-1, 1, 0]]


# In[143]:


scaler.transform(X_test)


# Min-Max or Scaling Features to a Range

# In[144]:


#Scaling features to lie between the minimum and maximum value


# In[145]:


#Scale the data to the range 0 - 1
X_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])


# In[146]:


min_max_scaler = preprocessing.MinMaxScaler()


# In[147]:


X_train_minmax = min_max_scaler.fit_transform(X_train)


# In[148]:


X_train_minmax


# In[149]:


X_test = np.array([[-3, -1, 0], [2, 1.5, 4]])


# In[150]:


X_test_minmax = min_max_scaler.transform(X_test)


# In[151]:


X_test_minmax


# MaxAbsScaler

# In[152]:


#Works similar to Min-Max but scales usually from -1 - 1 
#Meant for data that is already centered at 0 or sparce data


# In[153]:


X_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])


# In[154]:


max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs


# Sometimes you need to use PCA to eliminate linear correlation between features

# Normalization

# In[155]:


#Process of scaling individual samples to have a unit norm
    #L1 Normalization - Least Absolute Deviations ensure the sum of absolute values are 1 in each row
    #L2 Normalization - Least Squares ensure that the sum of squares is 1


# In[156]:


X = [[1, -1, 2], [2, 0, 0], [0, 1, -1]]
X_normalized = preprocessing.normalize(X, norm = 'l2')
X_normalized


# Binarization

# In[157]:


X = [[1, -1, 2], [2, 0, 0], [0, 1, -1]]
binarizer = preprocessing.Binarizer().fit(X) # fit does nothing
binarizer


# In[158]:


binarizer.transform(X)


# In[159]:


binarizer = preprocessing.Binarizer(threshold = -0.5)


# In[160]:


binarizer.transform(X)


# Encoding Categorical Features

# In[161]:


source = ['australia', 'singapore', 'new zealand', 'hong kong']


# In[162]:


label_enc = preprocessing.LabelEncoder()
src = label_enc.fit_transform(source)


# In[163]:


print("Country to code mapping: \n")
for k, v in enumerate(label_enc.classes_):
    print(v, '\t', k)


# In[164]:


test_data = ['hong kong', 'singapore', 'australia', 'new zealand']


# In[165]:


result = label_enc.transform(test_data)


# In[166]:


print(result)


# One Hot/One of K Encoding

# In[167]:


source


# In[168]:


src


# In[169]:


#The process of turning a series of catergorical data into a set of binary results


# In[170]:


one_hot_enc = OneHotEncoder(sparse = False, categories = 'auto')
src = src.reshape(len(src), 1)
one_hot = one_hot_enc.fit_transform(src)
print(one_hot)


# In[171]:


invert_res = label_enc.inverse_transform([np.argmax(one_hot[0, :])])
print(invert_res)


# In[172]:


invert_res = label_enc.inverse_transform([np.argmax(one_hot[3, :])])
print(invert_res)


# =================================================================================================================

# Variance-Bias Trade Off

# In[180]:


#The Bias of an estimator is its average error for different training sets
#The Variance determines how sensitive it is to varying training sets

#We want to reduce the Bias and Variance as much as possible


# In[181]:


#Here we are looking at how the degrees of freedom affects how well the model performs
def function(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = function(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize = (14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks = (), yticks = ())
               
    polynomial_features = PolynomialFeatures(degree = degrees[i], include_bias = False)
               
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features), ('linear_regression', linear_regression)])
    
    pipeline.fit(X[:, np.newaxis], y)
    
    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring = 'neg_mean_squared_error', cv = 10)
               
    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label = 'Model')
    plt.plot(X_test, function(X_test), label = 'True Function')
    plt.scatter(X, y, edgecolor = 'b', s = 20, label = 'Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc = 'best')
    plt.title('Degree {}\nMSE = {:.2e}(+/1 {:.2e})'.format(degrees[i], -scores.mean(), scores.std()))
plt.show


# Learning Curve

# In[199]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, 
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[200]:


X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values
y = df['MEDV'].values

title = 'Learning Curves (Ridge Regression)'

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = Ridge()
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
plt.show()


# =====================================================================

# Cross Validation

# In[202]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=0)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

regression = svm.SVR(kernel='linear', C=1).fit(X_train, y_train)
regression.score(X_test, y_test)   


# Compute Cross Validation Metrics

# In[203]:


regression = svm.SVR(kernel='linear', C=1)
scores = cross_val_score(regression, X, y, cv=5)
scores   


# In[204]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# In[205]:


scores = cross_val_score(regression, X, y, cv=5, 
                         scoring='neg_mean_squared_error')
scores 


# K-fold

# In[207]:


X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))


# Stratified K-fold

# In[209]:


X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))


# In[213]:


pipe_svm = make_pipeline(StandardScaler(),
                         PCA(n_components=2),
                         svm.SVR(kernel='linear', C=1))
pipe_svm.fit(X_train, y_train)
y_pred = pipe_svm.predict(X_test)
print('Test Accuracy: %.3f' % pipe_svm.score(X_test, y_test))


# In[214]:


scores = cross_val_score(estimator=pipe_svm,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)


# In[215]:


print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

