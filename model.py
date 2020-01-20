import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime


class currInfo():

    def __init__(self):
        self.curr = datetime.today()

    def currSeason(self):
        doy = self.curr.timetuple().tm_yday
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        if doy in spring:
            return 2
        elif doy in summer:
            return 3
        elif doy in fall:
            return 4
        else:
            return 1

    def currWorkingDay(self):
        return bool(len(pd.bdate_range(self.curr, self.curr)))

    def selectRows(self, df):
        df = df.loc[df['season'] == self.currSeason()]
        # df = df.loc[df['workingday'] == self.currWorkingDay()]
        return df

# reading csv file into a pandas dataframe
data = pd.read_csv("data/day.csv")

# data trimming
data = data[['temp', 'season', 'windspeed', 'hum', 'cnt']]

data['spring'] = np.multiply(data['season'] == 1, 1)
data['summer'] = np.multiply(data['season'] == 2, 1)
data['fall'] = np.multiply(data['season'] == 3, 1)
data['winter'] = np.multiply(data['season'] == 4, 1)
data = data.drop(['season'], 1)

curr = currInfo()
# data = curr.selectRows(data)

x = np.array(data.drop(['cnt'], 1))
y = np.array(data['cnt'])

accuracy_lst = []

for _ in range(1000):
    # training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    poly = PolynomialFeatures(degree=2)
    model = LinearRegression()

    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.fit_transform(x_test)

    model.fit(x_poly_train, y_train)
    y_pred = model.predict(x_poly_test)

    accuracy = model.score(x_poly_test, y_test)
    accuracy_lst.append(accuracy)

accuracy_lst.sort()
print("accuracy: " + str(accuracy_lst[500]) + "\n")
