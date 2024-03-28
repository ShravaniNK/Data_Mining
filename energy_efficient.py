import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv(r'C:\Data_BVS\Datasets\EnergyEfficiency\energy_efficiency_data.csv')
#print(df)

df.rename(columns = {'X1': 'Relative_Compactness', 'X2':'Surface_Area', 'X3':'Wall_Area', 'X4':'Roof_Area', 'X5':'Overall_Height', 'X6':'Orientation', 'X7':'Glazing_Area', 'X8':'Glazing_Area_Distribution', 'Y1':'Heating_Load' , 'Y2':'Cooling_Load'}, inplace = True)
print(df.head())

# Dimensions
print(data.shape)

# Checking for missing data
print(data.isna().sum())

# Information about the Dataset
print(data.info())

# Descriptive Statistics gives summary of each attribute in the dataset
print(data.describe())

# Unique values in each column
data1 = data.iloc[:, 0:8]
for c in data1.columns:
    print(c, data1[c].unique())
    print("-------------------------------------------------------------")
    
# 1.Distribution of input features in data
data1.hist(bins=20, figsize=(20,10))
plt.show()

# 2.Distribution of input features in data
data1.hist(edgecolor="red",figsize=(20,10))
plt.show()

# 3. Distribution of input features in data
data.hist(edgecolor="black",bins=20,figsize=(20,10))
plt.show()

#valuecounts
for c in data1.columns:
    print(c)
    print(data1[c].value_counts(normalize=True))
    
 # Heating load distribution for each input feature
for c in data1.columns:
    df = data.groupby(c)['Heating_Load'].mean()    
    sns.barplot(x = df.index , y = df.values)
    plt.xticks(rotation = 45)
    plt.show()
    
# Cooling load distribution for each input feature
for c in data1.columns:    
    df = data.groupby(c)['Cooling_Load'].mean()
    sns.barplot(x = df.index , y = df.values)
    plt.xticks(rotation = 45)
    plt.show()   



# Heating and Cooling load average of each input feature
for c in data1.columns:
    data.groupby(c)['Heating_Load'].mean().plot(label = "Heat")   
    data.groupby(c)['Cooling_Load'].mean().plot(label = "Cool")
    plt.ylabel('Average')
    plt.legend(loc="upper left")
    plt.show()

# Heating and Cooling load average of each input feature
    
label =['Heating_Load' , 'Cooling_Load']
for c in data1.columns:
    df = data.groupby(c)[label].mean()
    df1= pd.DataFrame({ c: list(df.index),
                     "Heat" :list(df.values[:,0]),
                       "Cool":list(df.values[:,1])
                      })
    df1.plot(c , y = ["Heat","Cool"], kind= 'bar')
    plt.xticks(rotation = 45)
    plt.show()     
#Visualize
# Visualizing the distribution of all independent features 
fig, axs = plt.subplots(2, 4, sharey=True,figsize=(10,8))

axs[0,0].hist(data['Relative_Compactness'], bins =14)
axs[0,0].set_xlabel("Relative_Compactness")

axs[0,1].hist(data['Surface_Area'])
axs[0,1].set_xlabel("Surface_Area")

axs[0,2].hist(data['Wall_Area'])
axs[0,2].set_xlabel("Wall_Area")

axs[0,3].hist(data['Roof_Area'])
axs[0,3].set_xlabel("Roof_Area")

axs[1,0].hist(data['Overall_Height'])
axs[1,0].set_xlabel("Overall_Height")

axs[1,1].hist(data['Orientation'])
axs[1,1].set_xlabel("Orientation")

axs[1,2].hist(data['Glazing_Area'])
axs[1,2].set_xlabel("Glazing_Area")

axs[1,3].hist(data['Glazing_Area_Distribution'])
axs[1,3].set_xlabel("Glazing_Area_Distribution")   
# Checking for outliers in the dataset which may cause negative effects on the performance of the analysis
fig =plt.figure(figsize =(20,20))

# Boxplots
for c in range(len(data.columns)):
    fig.add_subplot(6,2, c+1)
    sns.boxplot(x = data.iloc[:,c])
plt.show()

# Distribution of values
fig =plt.figure(figsize =(20,20))

for c in range(len(data1.columns)):
    fig.add_subplot(6,2, c+1)
    sns.distplot(x = data1.iloc[:,c])

plt.show()

  
# Correlation between attributes
round(data.corr() , 2)

# Visualizing the correlation matrix using heatmap
plt.figure(figsize=(10,8))
corr = data.corr().round(2)
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title('Correlation Plot')
plt.show()


# PairPlot
plt.figure()
sns.pairplot(data,hue='Overall_Height')
plt.show()


df = data

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=20):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations:\n")
print(round(get_top_abs_correlations(df, 11),2))

#The diagonal axes are univariate distribution of the data for the variable in that column.
plt.figure()
sns.pairplot(data)
plt.show()

print("============================================================================================")

# Data Splitting and Scaling

# Split the data into train and test set
from sklearn.model_selection import train_test_split
X = data.iloc[:,:-2]
y = data.iloc[:,-2:]
X_train , X_test,y_train, y_test = train_test_split(X,y, test_size = 0.20 , random_state = 1234)

#forward selection
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector
lr =  LinearRegression()
lr.fit(X,y)
ffs = SequentialFeatureSelector(lr, k_features ='best' , forward= True, n_jobs = -1 )
ffs.fit(X,y)
feat = list(ffs.k_feature_names_)
print(feat)
#feat = list(map(int, feat))
lr.fit(X_train[feat], y_train)
y_p = lr.predict(X_train[feat])

r2 = r2_score(y_train, y_p)
mse = mean_squared_error(y_train, y_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_p)


# backward
ffs = SequentialFeatureSelector(lr, k_features ='best' , forward= False , n_jobs = -1 )
ffs.fit(X,y)
feat = list(ffs.k_feature_names_)
print(feat)
#feat = list(map(int, feat))
lr.fit(X_train[feat], y_train)
y_p = lr.predict(X_train[feat])

r2 = r2_score(y_train, y_p)
mse = mean_squared_error(y_train, y_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_p)

print(r2)
print(mse)
print(rmse)
print(mae)

# Recursive feature selection
from sklearn.feature_selection import RFE
rfe = RFE(lr , n_features_to_select = 7)
rfe.fit(X_train , y_train)
y_p = rfe.predict(X_train)

r2 = r2_score(y_train, y_p)
mse = mean_squared_error(y_train, y_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_p)

print(r2)
print(mse)
print(rmse)
print(mae)

print("=============================================================================================")

## Min_Max scaling on data after split(correct method) 
from sklearn.preprocessing import MinMaxScaler

MinMax = MinMaxScaler(feature_range= (0,1))
X_tr = MinMax.fit_transform(X_train)
X_te = MinMax.fit_transform(X_test)
df_X_tr = pd.DataFrame(data=X_tr, columns= X_train.columns)
df_X_te= pd.DataFrame(data=X_te, columns= X_test.columns)


# Normalize 
norm = Normalizer()
X_tr_norm = norm.fit_transform(X_train)
X_te_norm = norm.transform(X_test)
df_X_tr_norm = pd.DataFrame(data=X_tr_norm, columns= X_train.columns)
df_X_te_norm = pd.DataFrame(data=X_te_norm, columns= X_test.columns) 

# Standardize

std_scaler = StandardScaler()
# transform train data
X_tr_std = std_scaler.fit_transform(X_train)
# transform test data
X_te_std = std_scaler.transform(X_test)

df_X_tr_std = pd.DataFrame(data=X_tr_std, columns= X_train.columns)
df_X_te_std = pd.DataFrame(data=X_te_std, columns= X_test.columns)


print("=============================================================================================")

from sklearn.decomposition import PCA
pca = PCA(n_components = 7 ,random_state=42)
X_train1 = pca.fit_transform(df_X_tr_mm)
X_test1 = pca.transform(df_X_te_mm)

train_and_test_models(X_train1 ,X_test1, y_train,y_test)
#Multivariate Analysis
from sklearn.linear_model import LinearRegression  # PCA

  
classifier = LinearRegression()
fit = classifier.fit(X_train1, y_train)
y_pred = classifier.predict(X_test1)

print("Training set score:{:.3f}".format(fit.score(X_train1, y_train)))
print("Test set score:{:.3f}".format(r2_score(y_test,y_pred)))
print('Mean Squared Error:',mean_squared_error(y_test,y_pred))
print('Root Of Mean Squared Error:',np.sqrt(mean_squared_error(y_test, y_pred)))


print("=============================================================================================")

# DATA MODELING


# 1. Linear Regression Model

LR_model = LinearRegression()
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)

# Calculating an error score to summarize the predictive skill of a model

# R2 Score
r2_s = r2_score(y_test,y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae =  mean_absolute_error(y_test, y_pred)

print("Linear Regression Model:")
print("R2 score:" ,round(r2_s,3))
print("Mean Squared Error:" , round(mse,3))
print("Root Mean Squared Error:" , round(rmse,3))
print("Mean Absolute Error:" , round(mae,3))
print("Predicted:\n",y_pred[0:5],"\n Actual:\n",y_test.head() )

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")

print("-------------------------------------------------------------------")

# 2. DecisionTree Regressor Model

DT_model = DecisionTreeRegressor()
DT_model.fit(X_train, y_train)

y_pred = DT_model.predict(X_test)

# Calculating an error score to summarize the predictive skill of a model

# R2 Score
r2_s = r2_score(y_test,y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae =  mean_absolute_error(y_test, y_pred)

print("Decision Tree Regressor Model:")
print("R2 score:" ,round(r2_s,3))
print("Mean Squared Error:" , round(mse,3))
print("Root Mean Squared Error:" , round(rmse,3))
print("Mean Absolute Error:" , round(mae,3))
print("Predicted:\n",y_pred[0:5],"\n Actual:\n",y_test.head() )

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")

print("-------------------------------------------------------------------")

#3.Random Forest Regressor Model

RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred = RF_model.predict(X_test)

# Calculating an error score to summarize the predictive skill of a model

# R2 Score
r2_s = r2_score(y_test,y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae =  mean_absolute_error(y_test, y_pred)

print("Random Forest Regressor Model:")
print("R2 score:" ,round(r2_s,3))
print("Mean Squared Error:" , round(mse,3))
print("Root Mean Squared Error:" , round(rmse,3))
print("Mean Absolute Error:" , round(mae,3))
print("Predicted:\n",y_pred[0:5],"\n Actual:\n",y_test.head())
plt.scatter(y_test, y_pred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(RF_model, X_train,y_train, cv=5)
print(scores)

print("-------------------------------------------------------------------")

# 4. K-Neighbors Regressor Model

KNN_model = KNeighborsRegressor()
KNN_model.fit(X_train, y_train)
y_pred= KNN_model.predict(X_test)

# Calculating an error score to summarize the predictive skill of a model

# R2 Score
r2_s = r2_score(y_test,y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae =  mean_absolute_error(y_test, y_pred)

print("K-Neighbors Regressor Model:")
print("R2 score:" ,round(r2_s,3))
print("Mean Squared Error:" , round(mse,3))
print("Root Mean Squared Error:" , round(rmse,3))
print("Mean Absolute Error:" , round(mae,3))
print("Predicted:\n",y_pred[0:5],"\n Actual:\n",y_test.head() )
plt.scatter(y_test, y_pred)

print("-------------------------------------------------------------------")

# 5. Multi Output Regressor Model

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

MOR_model = MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
MOR_model.fit(X_train, y_train)
y_pred = MOR_model.predict(X_test)

# Calculating an error score to summarize the predictive skill of a model

# R2 Score
r2_s = r2_score(y_test,y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae =  mean_absolute_error(y_test, y_pred)

print("Multi Output Regressor Model:")
print("R2 score:" ,round(r2_s,3))
print("Mean Squared Error:" , round(mse,3))
print("Root Mean Squared Error:" , round(rmse,3))
print("Mean Absolute Error:" , round(mae,3))
print("Predicted:\n",y_pred[0:5],"\n Actual:\n",y_test.head() )
plt.scatter(y_test, y_pred)

print("-------------------------------------------------------------------")

# 6. MOR_LVSR

MOR_model = MultiOutputRegressor(LinearSVR())
MOR_model.fit(X_train, y_train)
y_pred = MOR_model.predict(X_test)

# Calculating an error score to summarize the predictive skill of a model

# R2 Score
r2_s = r2_score(y_test,y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae =  mean_absolute_error(y_test, y_pred)

print("Multi Output Regressor Model:")
print("R2 score:" ,round(r2_s,3))
print("Mean Squared Error:" , round(mse,3))
print("Root Mean Squared Error:" , round(rmse,3))
print("Mean Absolute Error:" , round(mae,3))
plt.scatter(y_test, y_pred)



print("***********************************************************************************************************")
#train and evaluate different regression models 
def train_and_test_models(X_train, X_test, y_train, y_test):
    # Initialize different regression models
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("MOR_SVR", MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))),  
        ("MOR_LSVR", MultiOutputRegressor(LinearSVR()))
        
        ]
    
    #Create lists to store the results
    model_names = []
    r2_tr_values = []
    r2_values = []
    mse_values = []    
    rmse_values=[]
    mae_values = []
           
    # Fit and evaluate each model
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_tr_pred = model.predict(X_train)
        y_pred = model.predict(X_test)
        
        r2_tr_score = r2_score(y_train, y_tr_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
               
        model_names.append(model_name)
        r2_tr_values.append(r2_tr_score)
        r2_values.append(r2)
        mse_values.append(mse)
        mae_values.append(mae)
        rmse_values.append(rmse)
        
        
        
    metrics_df=pd.DataFrame({
            'Model': model_names,
            'R2_train' : np.round(r2_tr_values,3),
            'R2_test': np.round(r2_values,3),            
            'MSE': np.round(mse_values,3),
            'RMSE': np.round(rmse_values,3),
            'MAE': np.round(mae_values,3)
            
        })
    return metrics_df


# raw data results
train_and_test_models(X_train , X_test,y_train, y_test) 
# Minmax scaler results
train_and_test_models(df_X_tr_mm, df_X_te_mm, y_train, y_test)
#train_and_test_models(X_tr_mm, X_te_mm, y_train, y_test)

# Normalizer results
train_and_test_models(df_X_tr_norm, df_X_te_norm, y_train, y_test)
#train_and_test_models(X_tr_norm , X_te_norm ,y_train, y_test) 

#Standardized data results
train_and_test_models(df_X_tr_std , df_X_te_std ,y_train, y_test)

​print("***********************************************************************************************************")

# HYPERPARAMETER TUNING
# Decision Tree
reg_decision_model = DecisionTreeRegressor()
# fit independent varaibles to the dependent variables
reg_decision_model.fit(X_train,y_train)
print("Parameters:", reg_decision_model.get_params)
print('train score:',reg_decision_model.score(X_train,y_train))
print('test score:',np.round(reg_decision_model.score(X_test,y_test),3))

reg_decision_model.get_params(deep=True)

# define parameters for tuning 
parameters = {"criterion": ['squared_error', 'absolute_error'],
              "splitter":["best","random"],
              "max_depth" : [3,5,6,7,None],
              "max_leaf_nodes" : [1,3,5,None],
              "min_samples_leaf":[1,2,3,4,5,6,7],
              "min_samples_split": [1,2,3,4,5],
              "min_weight_fraction_leaf":[0.0, 0.1,0.2,0.3],
              "max_features":[1,2,"log2","sqrt",None],
              "max_leaf_nodes":[20,30] }

# peform grid search
tuning_model = GridSearchCV(reg_decision_model,param_grid=parameters,scoring='r2',cv=5)

# function for calculating how much time take for hyperparameter tuning

def timer(start_time=None):
    if not start_time:
        start_time=datetime.now()
        return start_time
    elif start_time:
        thour,temp_sec=divmod((datetime.now()-start_time).total_seconds(),3600)
        tmin,tsec=divmod(temp_sec,60)
        print(thour,":",tmin,':',round(tsec,2))
        
# %%capture
from datetime import datetime
start_time=timer(None)
tuning_model.fit(X_train, y_train)
timer(start_time)

  # best hyperparameters & best model score
print("Best Hyperparameters:", tuning_model.best_params_)
print("Best Accuracy:", tuning_model.best_score_)

 # Train and evaluate the model with the best hyperparameter
best_dtm = DecisionTreeRegressor(criterion ='squared_error', max_depth=7,
                                 max_features=None, max_leaf_nodes=30,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 min_samples_split = 2, splitter='best')

# model fit
best_dtm.fit(X_train,y_train)

# predict on test data
tuned_pred =best_dtm.predict(X_test)

plt.scatter(y_test,tuned_pred)   

# METRICS with hyperparameter tuned 
print('R2 train:',r2_score(y_train, best_dtm.predict(X_train)) )
print('R2:' , r2_score(y_test, tuned_pred))
print('MSE:', mean_squared_error(y_test, tuned_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, tuned_pred)))
print('MAE:', mean_absolute_error(y_test,tuned_pred)) 



