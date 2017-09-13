![GitHub Logo](https://s3.ap-south-1.amazonaws.com/greyatom-social/logo.png)

# Linear Regression Assignment

## 1: Write `load_data()` function:

* The function should load `load_diabetes` dataset from sklearn's inbuilt datasets:
* Accepts no parameters
* Returns
    - feature matrix of `load_diabetes`
    - target array of `load_diabetes`

## 2. Training linear regression model in sklearn

Now, we will train a linear regression model using X and y from the previous example and claculate `mean squares error` by predicting the value of y using X.

Write a function called `myLinearModel()` which accepts
- X, y (Numpy arrays for training; any format acceptable by sklearn will work)

Returns
- mse
- Trained linear model on X and y

## 3. Training Ridge regression model in sklearn

Now, we will train a Ridge model using X and y from the previous example and claculate `mean squares error` by predicting the value of y using X.

Write a function called `myRidge()` which accepts
- X, y (Numpy arrays for training; any format acceptable by sklearn will work)
- alpha (float) (alpha for regularization)

Returns
- mse
- Trained linear model on X and y

## 4. Training Lasso regression model in sklearn

Now, we will train a Lasso model using X and y from the previous example and claculate `mean squares error` by predicting the value of y using X.

Write a function called `myLasso()` which accepts
- X, y (Numpy arrays for training; any format acceptable by sklearn will work)
- alpha (float) (alpha for regularization)

Returns
- mse
- Trained linear model on X and y

## 5. Training Linear Regression model with Polynomial Features in sklearn

Now, we will train a linear regression model with polynomial features using X and y from the previous example and claculate `mean squares error` by predicting the value of y using X.

Write a function called `polynomial_lr_pipeline()` which accepts
- X, y (Numpy arrays for training; any format acceptable by sklearn will work)
- power (int) (power for polynomial feature creation)

Returns
- mse
- Trained linear model on X and y

## 6. Training Ridge Regression model with Polynomial Features in sklearn

Now, we will train a ridge regression model with polynomial features using X and y from the previous example and claculate `mean squares error` by predicting the value of y using X.

Write a function called `polynomial_ridge_pipeline()` which accepts
- X, y (Numpy arrays for training; any format acceptable by sklearn will work)
- power (int) (power for polynomial feature creation)
- alpha (float) (alpha for regularization)

Returns
- mse
- Trained linear model on X and y

## 7. Training Ridge Regression model with Polynomial Features in sklearn, along with k-fold cross-validation

Now, we will train a ridge model with polynomial features using X and y from the previous example and claculate `mean squares error` by predicting the value of y using X.

Write a function called `cross_val()` which accepts
- X, y (Numpy arrays for training; any format acceptable by sklearn will work)
- power (int) (power for polynomial feature creation)
- alpha (float) (alpha for regularization)
- k (int) (for k-fold)

Returns
- mean mse