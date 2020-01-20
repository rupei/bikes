import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading csv file into a pandas dataframe
data = pd.read_csv("data/day.csv")
data = data.drop(['instant', 'dteday', 'yr'], 1)
predicts = ['cnt']

# finding strength of correlation between IV and DVs
for predict in predicts:
    for col in data.columns:
        x = np.array(data[col])
        y = np.array(data[predict])

        plt.scatter(x, y)
        plt.title(col + " vs " + predict + " with r: " + str(round(np.corrcoef(x, y)[1][0], 3)))
        plt.show()

# testing for hypothesized multicollinearity in atemp vs temp
x = np.array(data['atemp'])
y = np.array(data['temp'])
plt.scatter(x, y)
plt.title('atemp vs temp with r: ' + str(round(np.corrcoef(x, y)[1][0], 3)))
plt.show()