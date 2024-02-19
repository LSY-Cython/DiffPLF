import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle as pkl
import os
from meteostat import Point, Hourly, Daily
import shutil

def get_weather_data():  # CA time zone = UTC standard - 8h
    # Set time period
    start = datetime.datetime(2016, 1, 1)
    end = datetime.datetime(2019, 12, 31)
    # Create Point for Palo Alto, CA
    location = Point(37.468319, -122.143936)
    # # Get hourly data
    # data = Hourly(location, start, end)
    # data = data.fetch()
    # data.to_csv("dataset/weather_hourly.csv")
    # Get daily data
    data = Daily(location, start, end)
    data = data.fetch()
    data.to_csv("dataset/weather_daily.csv")

def get_daily_weather():
    weather_data = pd.read_csv("dataset/weather_hourly.csv").iloc[8:]
    data_num = len(weather_data)//24
    daily_data = dict()
    for i in range(0, data_num, 1):
        temp_data = weather_data.iloc[i*24:(i+1)*24]
        temp_date = weather_data.iloc[i*24]["time"].split(" ")[0]
        year = int(temp_date.split("-")[0])
        month = int(temp_date.split("-")[1])
        day = int(temp_date.split("-")[2])
        temp_date = f"{year}-{month}-{day}"
        temp_t = np.array(temp_data["temp"])  # temperature
        temp_h = np.array(temp_data["rhum"])  # relative humidity (%)
        daily_data[temp_date] = {}
        daily_data[temp_date]["temp"] = temp_t
        daily_data[temp_date]["rhum"] = temp_h
        print(f"weather {temp_date} done!")
    with open("dataset/weather_data.pkl", "wb") as f:
        pkl.dump(daily_data, f)

def get_period_data(start_date, end_date):
    data_file = "D:\\Datasets\\Palo_Alto_EV_Charging_Station_Usage_Open_Data.csv"
    raw_data = pd.read_csv(data_file)
    raw_data["Start Date"] = pd.to_datetime(raw_data["Start Date"])
    period_data = raw_data[(raw_data['Start Date'] >= start_date) & (raw_data['Start Date'] <= end_date)]
    return period_data

def time_to_step(time):  # the number of intervals
    hour = int(time.split(":")[0])
    min = int(time.split(":")[1])
    if min%interval >= interval*0.5:
        step_num = (hour*60+min)//interval + 1
    else:
        step_num = (hour*60+min)//interval
    return step_num

def date_to_int(date):
    date_struct = date.split("-")
    year = date_struct[0]
    month = date_struct[1]
    if len(month) == 1:
        month = f"0{month}"
    day = date_struct[2]
    if len(day) == 1:
        day = f"0{day}"
    date_int = int(f"{year}{month}{day}")
    week_day = datetime.date(int(year), int(month), int(day)).weekday() + 1
    return date_int, week_day

def get_power_series(start_step, end_step, energy, init_series=None):
    series_num = 24*60//interval
    if init_series is None:
        daily_series = np.zeros(series_num)
    else:
        daily_series = init_series.copy()
    step_num = end_step - start_step + 1
    tail_step = start_step + int((step_num-1)*2/3)
    if step_num == 1:
        const_power = energy
    else:
        const_power = energy/((step_num-1)*15/60)  # kW
    temp_series = np.zeros(step_num)
    temp_series[0:tail_step-start_step+1] = const_power
    tail_series = np.linspace(const_power, 0, end_step-tail_step+1)
    temp_series[tail_step-start_step:] = tail_series
    if end_step >= series_num:
        next_series = np.zeros(series_num)  # for the charging duration spanning two days
        next_series[0:end_step+1-series_num] = temp_series[-(end_step+1-series_num):]
        daily_series[start_step:] += temp_series[0:series_num-start_step]
        return daily_series, next_series
    else:
        daily_series[start_step:end_step+1] += temp_series
        return daily_series, None

def get_daily_load(start_date, end_date):
    period_data = get_period_data(start_date, end_date)
    start_time = period_data["Start Date"]
    charging_duration = period_data["Charging Time (hh:mm:ss)"]
    energy = period_data["Energy (kWh)"]
    daily_data = dict()
    init_series = None
    for i in range(len(start_time)):
        temp_st = start_time.iloc[i]
        temp_cd = time_to_step(charging_duration.iloc[i])
        temp_e = energy.iloc[i]
        temp_date = f"{temp_st.year}-{temp_st.month}-{temp_st.day}"
        temp_time = f"{temp_st.hour}:{temp_st.minute}"
        start_step = time_to_step(temp_time)
        end_step = start_step + temp_cd
        power_series, init_series = get_power_series(start_step, end_step, temp_e, init_series)
        if temp_date not in daily_data.keys():
            daily_data[temp_date] = {"profile": [], "energy": []}
        daily_data[temp_date]["profile"].append(power_series)
        daily_data[temp_date]["energy"].append(temp_e)
        print(f"session {temp_date} {temp_time} done!")
    with open("dataset/charging_data.pkl", "wb") as f:
        pkl.dump(daily_data, f)

def output_daily_data(start_date, end_date):
    start_int, _ = date_to_int(start_date)
    end_int, _ = date_to_int(end_date)
    with open("dataset/charging_data.pkl", "rb") as f:
        load_data = pkl.load(f)
    with open("dataset/weather_data.pkl", "rb") as f:
        weather_data = pkl.load(f)
    # output daily data file
    output_dir = "dataset/daily"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)
    energy, evs, history = [], [], []
    start_int_new = start_int + lag
    for date, data in load_data.items():
        date_int, date_day = date_to_int(date)
        if date_int < start_int or date_int > end_int:
            continue
        # get daily charging data
        session_profiles = load_data[date]["profile"]
        session_energy = load_data[date]["energy"]
        daily_profile = np.sum(np.array(session_profiles), axis=0)
        if start_int <= date_int < start_int_new:
            history.append(daily_profile.tolist())
            continue
        daily_energy = np.around(np.sum(session_energy), 4)
        energy.append(daily_energy)
        daily_evs = len(session_profiles)
        evs.append(daily_evs)
        # get daily weather data
        if date not in weather_data.keys():
            break
        daily_weather = weather_data[date]
        daily_temp = np.repeat(daily_weather["temp"], 60//interval)
        daily_rhum = np.repeat(daily_weather["rhum"], 60//interval)
        output_file = f"{output_dir}/{date}"
        output_data = {"profile": daily_profile, "history": np.array(history), "energy": daily_energy, "evs": daily_evs,
                       "day": date_day, "temp": daily_temp, "rhum": daily_rhum}
        plt.plot(np.arange(lag*96), np.array(history).reshape(-1), color="blue", label="History")
        plt.plot(np.arange(lag*96, (lag+1)*96, 1), daily_profile, color="red", label="Ground truth")
        plt.xlim((0, 96*(lag+1)))
        plt.xticks([])
        plt.title(f"Weekday: {date_day}\nRequest energy: {daily_energy}kWh, EVs: {daily_evs}")
        plt.legend()
        plt.tight_layout(True)
        plt.savefig(f"{output_file}.png")
        plt.clf()
        with open(f"{output_file}.pkl", "wb") as f:
            pkl.dump(output_data, f)
        print(f"{date} output done!")
        history.pop(0)
        history.append(daily_profile)
    # # output total statistics
    # plt.plot(energy)
    # plt.savefig("dataset/request_energy.png")
    # plt.clf()
    # plt.plot(evs)
    # plt.savefig("dataset/EV number.png")
    # plt.clf()

if __name__ == "__main__":
    interval = 15  # min
    lag = 5  # day
    # get_weather_data()
    # get_daily_weather()
    # get_daily_load("1/1/2011 00:00", "31/12/2020 23:59")
    # output_daily_data("2016-1-1", "2019-12-31")