#!/bin/python

import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    data_set = pd.read_csv(r'Salary_Data.csv')
    data_set.head()

    # Training/testing data
    x = data_set.iloc[:, :-1].values  # Independent variable
    y = data_set.iloc[:,1].values     # Dependent variable

    # 0.2 - use 20% of data for each test
    x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.2, random_state=0)

    # Regressor model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predictions
    y_pred = regressor.predict(x_test)

    print(repr(y_pred), '\n')     # The predicted salaries
    print(repr(y_test))           # The real salaries

    # Training Plot
    plt.scatter(x_train, y_train) # Scatter data

    # Plot regression line
    plt.plot(x_train, regressor.predict(x_train), color='red')

    plt.title("Salary vs Experience (Training set)")
    plt.xlabel("Years of experience")
    plt.ylabel("Salaries")
    plt.show()

    # Test Plot
    plt.scatter(x_train, y_train) # Scatter data

    # Plot regression line
    plt.plot(x_train, regressor.predict(x_train), color='red')

    plt.title('Salary vs Experience (Testing set)')
    plt.xlabel("Years of experience")
    plt.ylabel("Salaries")
    plt.show()

if __name__ == '__main__':
    main()

