import datetime

import numpy as np

from src.cleanup_plot import confirmed_df, deaths_df


def func_1_get_confirmed_and_deaths():
    confirmed_cols = confirmed_df.keys()
    deaths_cols = deaths_df.keys()

    # Get all the dates for the ongoing coronavirus pandemic
    confirmed = confirmed_df.loc[:, confirmed_cols[4]:]
    deaths = deaths_df.loc[:, deaths_cols[4]:]
    return confirmed, deaths


def func_2(confirmed, deaths):
    num_dates = len(confirmed.keys())
    ck = confirmed.keys()
    dk = deaths.keys()

    world_cases = []
    total_deaths = []
    mortality_rate = []

    for i in range(num_dates):
        confirmed_sum = confirmed[ck[i]].sum()
        death_sum = deaths[dk[i]].sum()

        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)

        # calculate rates
        mortality_rate.append(death_sum / confirmed_sum)
    return world_cases, total_deaths, mortality_rate


# Getting daily increases and moving averages
def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i] - data[i - 1])
    return d


def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i + window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average


# window size
window = 7


def func_3_get_daily_increase_and_avg(world_cases):
    # confirmed cases
    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg = moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)
    return world_daily_increase, world_confirmed_avg, world_daily_increase_avg


def func_4_get_daily_increase_and_avg_2(total_deaths):
    # deaths
    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)
    return world_daily_death, world_death_avg, world_daily_death_avg


def get_days_since_1_22(ck):
    days_since_1_22 = np.array([i for i in range(len(ck))]).reshape(-1, 1)
    return days_since_1_22


def get_reshaped_world_cases_and_total_deaths(world_cases, total_deaths):
    # CAREFULL WITH UNMUTABLE!
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    return world_cases, total_deaths


def func_5_get_future_forecast_and_adjusted_dates(ck):
    days_in_future = 10
    future_forcast = np.array([i for i in range(len(ck) + days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-10]
    return future_forcast, adjusted_dates


def func_6_get_future_forecast_dates(future_forecast, start: str = '1/22/2020'):
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forecast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    return future_forcast_dates
