### main.py: defines an example usage of LinearRegression module 
import LinearRegression as linreg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

'''
- taking a look at the data, there are 12 features 
- first column is the labels, so let's separate them 
- 545 training examples 
- split into training and 
'''
def normalize_df(df):
    def normalize_yes_no(str):
        # normalizes yes and no, yes = 1, no = 0
        if str == 'yes':
            return 1
        else:
            return 0

    def normalize_furnished(str):
        if str == 'unfurnished':
            return 0
        elif str == 'semi-furnished':
            return 1
        elif str == 'furnished':
            return 2
        else:
            raise Exception(f"Did not handle furnished case " + str) 

    df['mainroad'] = df['mainroad'].apply(normalize_yes_no)
    df['guestroom'] = df['guestroom'].apply(normalize_yes_no)
    df['basement'] = df['basement'].apply(normalize_yes_no)
    df['hotwaterheating'] = df['hotwaterheating'].apply(normalize_yes_no)
    df['airconditioning'] = df['airconditioning'].apply(normalize_yes_no)
    df['prefarea'] = df['prefarea'].apply(normalize_yes_no)
    df['furnishingstatus'] = df['furnishingstatus'].apply(normalize_furnished)
    return df 
      

def split_and_clean_dataset(input_data, labels):
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=40)
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(y_train, dtype=np.float32), np.array(y_test, dtype=np.float32)

def main():
    dataset = "./Housing.csv"
    df = pd.read_csv(dataset)
    # separate labels from features 
    # labels are the first column (so we get all rows, but only column 0)
    labels = df.iloc[:, 0]
    input_data = df.iloc[:, 1:]
    # normalize dataset, before splitting 
    input_data = normalize_df(input_data)
    X_train, X_test, y_train, y_test = split_and_clean_dataset(input_data, labels)
    # now we can run fit, more epochs the better
    model = linreg.LinearRegression(num_epochs=1000, learning_rate=0.005)
    model.fit(X_train, y_train)
    # pull a random example from test
    print("Testing random example:")
    print(f"Inputs {X_test[0]}")
    test_example = X_test[0]
    test_example_correct_label = y_test[0]
    print(f"Model prediction: {model.predict(test_example)}")
    print(f"Correct label: {test_example_correct_label}")
    print(f"Calculating the mean squared error")
    y_pred = np.array([model.predict(x) for x in X_test])
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean squared error: {mse}" )
    print(f"Mean absolute error: {mae}")
    

def comparison():
    print("Running comparison Linear Regression")
    dataset = "./Housing.csv"
    df = pd.read_csv(dataset)
    print(df.head())

    # Separate labels from features
    labels = df.iloc[:, 0]
    input_data = df.iloc[:, 1:]

    # Normalize categorical and numerical data
    input_data = normalize_df(input_data)
    
    # Scale the features
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=40)
    
    # Fit the scikit-learn Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")



if __name__=="__main__":
    main()
    comparison()
