from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def load_data():
    pass

def myLinearModel(X, y):
    pass

def myRidge(X, y, alpha):
    pass

def myLasso(X, y, alpha):
    pass

def polynomial_lr_pipeline(X, y, power):
    pass

def polynomial_redge_pipeline(X, y, power, alpha):
    pass

def cross_val(X, y, power, alpha, k):
    pass