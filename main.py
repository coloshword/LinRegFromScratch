### main.py: defines an example usage of LinearRegression module 
import LinearRegression as linreg
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():
    housing = fetch_openml(name="house_prices", as_frame=True)
    X = housing.data
    y = housing.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = linreg.LinearRegression()
    model.fit(X_train, X_test)    

main()