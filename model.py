import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar


class currInfo():

    def __init__(self):
        self.curr = datetime.today()
        self.weekday = -1
        self.holiday = -1

    def currMonth(self):
        return self.curr.month

    def currHoliday(self):
        holidays = USFederalHolidayCalendar().holidays().to_pydatetime()
        if self.curr in holidays:
            self.holiday = 1
            return 1
        else:
            self.holiday = 0
            return 0

    def currWeekday(self):
        day_of_week = self.curr.isoweekday()
        if day_of_week == 6 or day_of_week == 7:
            self.weekday = 0
            return 0
        else:
            self.weekday = 1
            return 1

    def currWorkingDay(self):
        return 1 if not self.weekday and not self.holiday else 0

    def selectRows(self, df):
        df = df.loc[df['mnth'] == self.currMonth()]
        df = df.loc[df['holiday'] == self.currHoliday()]
        # df = df.loc[df['weekday'] == self.currWeekday()]
        # df = df.loc[df['workingday'] == self.currWorkingDay()]
        return df


# reading csv file into a pandas dataframe
data = pd.read_csv("data/day.csv")

# data trimming
predict = 'cnt'
data = data.drop(['instant', 'dteday', 'yr', 'weathersit', 'season', 'casual', 'registered', 'weekday', 'workingday'], 1)
curr = currInfo()
data = curr.selectRows(data)

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
model = linear_model.LinearRegression()

# fitting
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("accuracy: " + str(accuracy) + "\n")

# next day prediction
