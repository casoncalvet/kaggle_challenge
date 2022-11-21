# Predicting the Price of Diamonds using Machine Learning 
<div>

## Step 1: 
- ## Make training and testing dataframes model readable: 
  - #### Normalize data with outliers and convert cualitative data to numerical values
    - ##### Translate Cut, Color , Clarity columns
    - ##### Remove ID column since it is the same as the index


![image info](../kaggle_challenge/ogcsv.png)


##### Create Function for fetching unique values
``` python 
def get_unique_dict(df_col): 
    new_dict= {}
    for i in list(df_col.unique()):
        new_dict[i] = 0
    return new_dict
``` 

##### Update Dictionary with desired values 
###### The most valuable keys were given the highest integer 
``` python 
clarity_dict= {'IF': 8, 'VVS1': 7, 'VVS2': 6, 'VS1': 5, 'VS2':4, 'SI1':3, 'SI2':2, 'I1':1}
```

##### Apply to both dataframes 
``` python 
df_train["clarity"] = df_train.clarity.map(clarity_dict)
df_test["clarity"]= df_test.clarity.map(clarity_dict)
```

#### Repeat for each Cualitative Column 

### Normalizing data: 

![hist](../kaggle_challenge/hist.png) 

#### Carat has outliers and no assumed normality, so must normalize the data 

``` python 
df_train['carat'] = min_max.fit_transform(df_train['carat'].values.reshape(-1, 1))
``` 

### Correlation Matrix to infer possible relationships to Price
![corr](../kaggle_challenge/corr.png) 


# Step 2: 
### Make dictionary of possible prediction models for Price

``` python 
models = {
    "lr": LinReg(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "sgd": SGDRegressor(),
    "knn": KNeighborsRegressor(),
    "grad": GradientBoostingRegressor(),
    "svr": SVR()
}
```

#### Fit all models in for loop 
``` python 
for name, model in models.items():
    print("Training 🏋️‍:", name)
    model.fit(X_train, y_train)
``` 

### Follow the same process to make predictions from each model and get metrics

``` python 
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"------{name}------")
    print('MAE - ', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE - ', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE - ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 - ', metrics.r2_score(y_test, y_pred))
``` 
#### Remove the weakest models (those with low R2 and/or very high error)
``` python 
models = {
    "lr": LinReg(),
    "ridge": Ridge(),
    "knn": KNeighborsRegressor(),
    "grad": GradientBoostingRegressor(),
    "svr": SVR()
}
```
### GradientBoostingRegressor is the best (R2= 0.9887492233649249) at predicting data w a 0.2 split 

# Step 3: 
### Now try with different splits

``` python 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
``` 

### It seems (a little bit) better to split the data by 0.15 (R2= 0.9889437851359357) for the GradientBoostingRegressor() model 

- ### Use cvs to get better read of models, taking the mean of R2's after running 5 times

``` python 
for name, model in models.items():
    scores = cvs(model, X, y, scoring = "r2", cv=5)
    print("Model: ", name, "Score: ", np.mean(scores))
``` 

# Step 4: 
### Now that I have chosen the model (GradientBoostingRegressor), lets test with the removal lowest ranking categories 

``` python 

estimator = GradientBoostingRegressor()
selector = RFE(estimator)
selector.fit(X_test, y_test)
selector.ranking_
``` 

### Evaluate best model only including x,y,z and carat variables, according to the rankings

``` python 
train_filt= df_train[['carat', 'x','y','z', 'price']]
test_filt= df_test[['carat', 'x','y','z']]

X_filt = train_filt.iloc[:,:-1]

y_filt = train_filt['price']

X_train_filt, X_test_filt, y_train_filt, y_test_filt = train_test_split(X_filt, y_filt, test_size=0.15)

``` 
### Include all models in new evaluation just in case 

## R2 decreases for all values, but its more possible that model is not being over-fitted
#### - Run CVS 5 times and take the mean of the outputs: R2= 0.9432860495987118

#### Will go with the original model rather than using only the highest ranking variables 

# Step 5: 
## Test models against test dataframe with a 0.15 split

![hist](../kaggle_challenge/kag.png) 
