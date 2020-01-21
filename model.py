import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# reading csv file into a pandas dataframe
data = pd.read_csv("data/day.csv")

# data trimming
data = data[['temp', 'season', 'windspeed', 'hum', 'cnt']]

# plot histogram of data
sns.distplot(np.array(data['cnt']))
plt.xlabel('cnt')
plt.title('Bike Count Histogram')
plt.show()

# eliminating outliers
print("Before: " + str(len(data)))
q3, q1 = np.percentile(np.array(data['cnt']), [75, 25])
iqr = q3 - q1
lo = q1 - (1.5 * iqr)
hi = q3 + (1.5 * iqr)
data = data.loc[(data['cnt'] >= lo) & (data['cnt'] <= hi)]
print("After: " + str(len(data)))

# one hot encoding
data['spring'] = np.multiply(data['season'] == 1, 1)
data['summer'] = np.multiply(data['season'] == 2, 1)
data['fall'] = np.multiply(data['season'] == 3, 1)
data['winter'] = np.multiply(data['season'] == 4, 1)
data = data.drop(['season'], 1)

# splitting to training/testing
x = np.array(data.drop(['cnt'], 1))
y = np.array(data['cnt'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# instantiating/fitting model
poly = PolynomialFeatures(degree=2)
model = LinearRegression()

x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.fit_transform(x_test)

model.fit(x_poly_train, y_train)
y_pred = model.predict(x_poly_test)

# evaluating
plt.scatter(y_pred, y_test)
plt.title("pred vs actual")
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

x_temp = [lst[0] for lst in x_test]
plt.scatter(x_temp, y_pred)
plt.title("temperature vs predicted")
plt.show()

accuracy = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("accuracy (r squared): " + str(round(accuracy, 2)) + "\n")
print("mse: " + str(round(mse, 2)) + "\n")

y_res = y_pred - y_test
plt.scatter(x_temp, y_res)
plt.title("temp vs cnt residuals")
plt.xlabel('temp')
plt.ylabel('cnt residuals')
plt.show()


# prediction
def bikePredict(temp, windspeed, hum, spring, summer, fall, winter):
    x_poly = poly.fit_transform([[float(temp), float(windspeed), float(hum), float(spring), float(summer), float(fall), float(winter)]])
    return model.predict(x_poly).tolist()[0]
