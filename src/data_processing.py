import numpy as np
from pandas import DataFrame


class DataProcessing:

    @staticmethod
    def remove_unused_cols_from_confirmed_df(confirmed_df: DataFrame):
        confirmed_cols = confirmed_df.keys()
        confirmed = confirmed_df.loc[:, confirmed_cols[4]:]
        return confirmed

    @staticmethod
    def get_days_since_1_22(confirmed_keys):
        days_since_1_22 = np.array([day for day in range(len(confirmed_keys))]).reshape(-1, 1)
        return days_since_1_22

    @staticmethod
    def get_world_cases_per_day(confirmed_df: DataFrame):
        confirmed_keys = confirmed_df.keys()
        num_days = len(confirmed_keys)
        world_cases = []
        for day in range(num_days):
            confirmed_sum = confirmed_df[confirmed_keys[day]].sum()
            world_cases.append(confirmed_sum)
        return world_cases
