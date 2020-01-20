from datetime import datetime
from datetime import timedelta
from model import bikePredict

temp = input("Enter tomorrow's normalized temperature: ")
hum = input("Enter tomorrow's humidity: ")
wind = input("Enter tomorrow's windspeed: ")
GROWTH = 8.25

doy = (datetime.today() + timedelta(days=1)).timetuple().tm_yday
summer_dates = range(80, 172)
fall_dates = range(172, 264)
winter_dates = range(264, 355)
spring = summer = fall = winter = 0

if doy in summer_dates:
    summer = 1
elif doy in fall_dates:
    fall = 1
elif doy in winter_dates:
    winter = 1
else:
    spring = 1

print("\n" + "Tomorrow's prediction is: " + str(round(GROWTH * bikePredict(temp, wind, hum, spring, summer, fall, winter))))

