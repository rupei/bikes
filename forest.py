import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# reading csv file into a pandas dataframe
data = pd.read_csv("data/day.csv")

# data trimming
data = data[['temp', 'season', 'windspeed', 'hum', 'cnt']]

x = np.array(data.drop(['cnt'], 1))
y = np.array(data['cnt'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regressor = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=100)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

accuracy = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("accuracy: " + str(accuracy) + "\n")
print("mse: " + str(mse) + "\n")
