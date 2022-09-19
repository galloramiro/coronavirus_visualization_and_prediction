import pandas as pd


class Dataframes:
    base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
    confirmed_url = f'{base_url}/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

    def __init__(self):
        self.confirmed_df = None

    def get_confirmed_df(self):
        if not self.confirmed_df:
            self.confirmed_df = pd.read_csv(self.confirmed_url)
        return self.confirmed_df
