import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# reading csv file into a pandas dataframe
data = pd.read_csv("data/day.csv")

# finding delta in usage rate between year 0 and year 1
yr0 = data[data['yr'] == 0]
yr1 = data[data['yr'] == 1]
print(sum(yr1['cnt'].values.tolist()) - sum(yr0['cnt'].values.tolist()))

# trimming
data = data.drop(['instant', 'dteday', 'yr', 'casual', 'registered'], 1)
predicts = ['cnt']
categorical = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
quant = ['temp', 'atemp', 'hum', 'windspeed']


# finding strength of correlation between quantitative IV and DVs
for predict in predicts:
    for col in quant:
        x = np.array(data[col])
        y = np.array(data[predict])

        plt.scatter(x, y)
        plt.title(col + " vs " + predict + " with r: " + str(round(np.corrcoef(x, y)[1][0], 3)))
        plt.show()

# plotting categorical variables
for predict in predicts:
    for col in categorical:
        x = np.array(data[col])
        y = np.array(data[predict])

        sns.distplot(x, kde=False)
        plt.title("histogram of " + col)
        plt.show()

        plt.scatter(x, y)
        plt.title(col + " vs " + predict)
        plt.show()

# creating correlation heatmap
df = data.drop(categorical, 1)
sns.heatmap(df.corr(), vmin=0.0, vmax=1.0, cmap='coolwarm', annot=True)
plt.show()

# testing for hypothesized multicollinearity in atemp vs temp
x = np.array(data['atemp'])
y = np.array(data['temp'])
plt.scatter(x, y)
plt.title('atemp vs temp with r: ' + str(round(np.corrcoef(x, y)[1][0], 3)))
plt.show()
