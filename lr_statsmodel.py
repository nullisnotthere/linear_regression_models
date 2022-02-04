#!/bin/python

# from https://www.statology.org/simple-linear-regression-in-python/

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def main():
    data = {
        'hours':[1, 2, 3, 3, 4, 5, 5, 5, 6, 8, 10, 13, 13],
        'score':[62, 65, 70, 75, 80, 82, 76, 88, 80, 90, 91, 93, 92]
    }

    data_frame = pd.DataFrame(data)

    print(data_frame[0:6])

    plt.scatter(data_frame.hours, data_frame.score)
    plt.title('Hours studied vs. Exam Score')
    plt.xlabel('Hours')
    plt.ylabel('Score')

    data_frame.boxplot(column=['score'])

    # response variable
    y = data_frame['score']

    # explanatory variable
    x = data_frame['hours']

    # add constant to predictor variable
    x = sm.add_constant(x)

    # fit linear regression model
    # actually interpret the data
    model = sm.OLS(y, x).fit()

    # view model summary
    print(model.summary())

    figure = plt.figure(figsize=(12, 8))
    figure = sm.graphics.plot_regress_exog(model, 'hours', fig=figure)

    # show all plots
    plt.show()

if __name__ == '__main__':
    main()
