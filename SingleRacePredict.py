import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
import requests
import json

load_dotenv(dotenv_path=r"F1 Predicts\.env")

api_key = os.getenv("WEATHER_API_KEY")
city = os.getenv("CITY_NAME")
UNITS = os.getenv("UNITS")


cache_dir = "/content/f1_cache"



if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

api_call = api_call = f"https://api.openweathermap.org/data/2.5/weather?lat=44.34&lon=10.99&appid={api_key}&q={city}&units={UNITS}"

response = requests.get(api_call)

if response.status_code == 200:
    data = response.json()
    print(" Meteo attuale:")
    print(" Temperatura:", data["main"]["temp"], "°C")
    print(" Condizioni:", data["weather"][0]["description"])
else:
    print(" Errore nella richiesta:", response.status_code)
temp = data["main"]["temp"]
weather_description = data["weather"][0]["description"]
humidity = data["main"]["humidity"]
wind_speed = data["wind"]["speed"]
clouds = data["clouds"]["all"]


fastf1.Cache.enable_cache(cache_dir)

print("Cache enabled at:", cache_dir)


session_2024 = fastf1.get_session(2024, 3, 'R')
session_2024.load()


laps_2024 = session_2024.laps[["Driver","LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()
laps_2024.head()


qualifying_2025 = pd.DataFrame({
    "Driver":[
        "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russel",
        "Yuki Tsunado", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
        "Pierre Gasly", "Carlos Sainz", "Lance Stroll", "Fernando Alonso"],
    "QualifyingTime (s)":[
        75.096, 75.103 ,75.481, 75.546, 75.670,
        75.737, 75.753 ,75.973, 75.980, 76.662,76.4, 76.51
    ]
})
qualifying_2025.head(12)



drive_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri":"PIA", "Max Verstappen":"VER", "George Russel":"RUS",
    "Yuki Tsunado":"TSU", "Alexander Albon":"ALB", "Charles Leclerc":"LEC", "Lewis Hamilton":"HAM",
    "Pierre Gasly":"GAS", "Carlos Sainz":"SAI", "Lance Stroll":"STR", "Fernando Alonso":"ALO"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(drive_mapping)



merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver", how="left")
merged_data.dropna(inplace=True)
merged_data["LapTime"] = merged_data["LapTime"].dt.total_seconds()
merged_data["Temperature"] = temp
merged_data["Humidity"] = humidity
merged_data["WindSpeed"] = wind_speed
merged_data["CloudCoverage"] = clouds
merged_data.head()
features = ["QualifyingTime (s)", "Temperature", "Humidity", "WindSpeed", "CloudCoverage"]
x = merged_data[features]


#x = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime"]

if x.shape[0] == 0:
  raise ValueError("No data available for training.")

print(y.head())  

print(merged_data.dtypes)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(x_train, y_train)


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


predicted_lap_times = model.predict(qualifying_2025[features])
qualifying_2025["Temperature"] = temp
qualifying_2025["Humidity"] = humidity
qualifying_2025["WindSpeed"] = wind_speed
qualifying_2025["CloudCoverage"] = clouds
qualifying_2025["PredictedRaceTime"] = predicted_lap_times
qualifying_2025.head()


qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime")
qualifying_2025["Position"] = range(1, len(qualifying_2025) + 1)
qualifying_2025.head()


print("\n Predicted 2025 Australian GP Winner \n")
print(qualifying_2025[["Driver","PredictedRaceTime"]])


y_pred = model.predict(x_test)
print(f"\n Model Error(MAE):{mean_absolute_error(y_test, y_pred):.2f} seconds")
