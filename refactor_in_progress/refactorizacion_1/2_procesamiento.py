
confirmed_cols = confirmed_df.keys()
deaths_cols = deaths_df.keys()

#
# Get all the dates for the ongoing coronavirus pandemic

# In[7]:


confirmed = confirmed_df.loc[:, confirmed_cols[4]:]
deaths = deaths_df.loc[:, deaths_cols[4]:]

# In[8]:


confirmed.keys()

# In[9]:


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





# Getting daily increases and moving averages

# In[10]:


def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d

def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i+window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average

# window size
window = 7

# confirmed cases
world_daily_increase = daily_increase(world_cases)
world_confirmed_avg = moving_average(world_cases, window)
world_daily_increase_avg = moving_average(world_daily_increase, window)

# deaths
world_daily_death = daily_increase(total_deaths)
world_death_avg = moving_average(total_deaths, window)
world_daily_death_avg = moving_average(world_daily_death, window)


# In[11]:


days_since_1_22 = np.array([i for i in range(len(ck))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)


# Future forcasting

# In[12]:


days_in_future = 10
future_forcast = np.array([i for i in range(len(ck)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


# Convert integer into datetime for better visualization

# In[13]:


start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# We are using data from 5/1/2022 to present for the prediction model


################
unique_countries = list(latest_data['Country_Region'].unique())

country_confirmed_cases = []
country_death_cases = []
country_active_cases = []
country_incidence_rate = []
country_mortality_rate = []

no_cases = []
for i in unique_countries:
    cases = latest_data[latest_data['Country_Region'] == i]['Confirmed'].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_countries.remove(i)

# sort countries by the number of confirmed cases
unique_countries = [k for k, v in
                    sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_data[latest_data['Country_Region'] == unique_countries[i]]['Confirmed'].sum()
    country_death_cases.append(latest_data[latest_data['Country_Region'] == unique_countries[i]]['Deaths'].sum())
    country_incidence_rate.append(
        latest_data[latest_data['Country_Region'] == unique_countries[i]]['Incident_Rate'].sum())
    country_mortality_rate.append(country_death_cases[i] / country_confirmed_cases[i])


country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': [format(int(i), ',d') for i in country_confirmed_cases],
                          'Number of Deaths': [format(int(i), ',d') for i in country_death_cases],
                          'Incidence Rate' : country_incidence_rate,
                          'Mortality Rate': country_mortality_rate})
# number of cases per country/region

country_df.style.background_gradient(cmap='Oranges')

#####################################


unique_provinces = list(latest_data['Province_State'].unique())

# Getting the latest information about **provinces/states** that have confirmed coronavirus cases

# In[46]:


province_confirmed_cases = []
province_country = []
province_death_cases = []
province_incidence_rate = []
province_mortality_rate = []

no_cases = []
for i in unique_provinces:
    cases = latest_data[latest_data['Province_State'] == i]['Confirmed'].sum()
    if cases > 0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

# remove areas with no confirmed cases
for i in no_cases:
    unique_provinces.remove(i)

unique_provinces = [k for k, v in
                    sorted(zip(unique_provinces, province_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_provinces)):
    province_confirmed_cases[i] = latest_data[latest_data['Province_State'] == unique_provinces[i]]['Confirmed'].sum()
    province_country.append(
        latest_data[latest_data['Province_State'] == unique_provinces[i]]['Country_Region'].unique()[0])
    province_death_cases.append(latest_data[latest_data['Province_State'] == unique_provinces[i]]['Deaths'].sum())
    #     province_recovery_cases.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Recovered'].sum())
    #     province_active.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Active'].sum())
    province_incidence_rate.append(
        latest_data[latest_data['Province_State'] == unique_provinces[i]]['Incident_Rate'].sum())
    province_mortality_rate.append(province_death_cases[i] / province_confirmed_cases[i])

# In[47]:


# nan_indices = []

# # handle nan if there is any, it is usually a float: float('nan')

# for i in range(len(unique_provinces)):
#     if type(unique_provinces[i]) == float:
#         nan_indices.append(i)

# unique_provinces = list(unique_provinces)
# province_confirmed_cases = list(province_confirmed_cases)

# for i in nan_indices:
#     unique_provinces.pop(i)
#     province_confirmed_cases.pop(i)


# In[48]:


# number of cases per province/state/city top 100
province_limit = 100
province_df = pd.DataFrame(
    {'Province/State Name': unique_provinces[:province_limit], 'Country': province_country[:province_limit],
     'Number of Confirmed Cases': [format(int(i), ',d') for i in province_confirmed_cases[:province_limit]],
     'Number of Deaths': [format(int(i), ',d') for i in province_death_cases[:province_limit]],
     'Incidence Rate': province_incidence_rate[:province_limit],
     'Mortality Rate': province_mortality_rate[:province_limit]})
# number of cases per country/region

province_df.style.background_gradient(cmap='Oranges')


########################


# return the data table with province/state info for a given country
def country_table(country_name):
    states = list(latest_data[latest_data['Country_Region'] == country_name]['Province_State'].unique())
    state_confirmed_cases = []
    state_death_cases = []
    # state_recovery_cases = []
    #     state_active = []
    state_incidence_rate = []
    state_mortality_rate = []

    no_cases = []
    for i in states:
        cases = latest_data[latest_data['Province_State'] == i]['Confirmed'].sum()
        if cases > 0:
            state_confirmed_cases.append(cases)
        else:
            no_cases.append(i)

    # remove areas with no confirmed cases
    for i in no_cases:
        states.remove(i)

    states = [k for k, v in sorted(zip(states, state_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
    for i in range(len(states)):
        state_confirmed_cases[i] = latest_data[latest_data['Province_State'] == states[i]]['Confirmed'].sum()
        state_death_cases.append(latest_data[latest_data['Province_State'] == states[i]]['Deaths'].sum())
        #     state_recovery_cases.append(latest_data[latest_data['Province_State']==states[i]]['Recovered'].sum())
        #         state_active.append(latest_data[latest_data['Province_State']==states[i]]['Active'].sum())
        state_incidence_rate.append(latest_data[latest_data['Province_State'] == states[i]]['Incident_Rate'].sum())
        state_mortality_rate.append(state_death_cases[i] / state_confirmed_cases[i])

    state_df = pd.DataFrame(
        {'State Name': states, 'Number of Confirmed Cases': [format(int(i), ',d') for i in state_confirmed_cases],
         'Number of Deaths': [format(int(i), ',d') for i in state_death_cases],
         'Incidence Rate': state_incidence_rate, 'Mortality Rate': state_mortality_rate})
    # number of cases per country/region
    return state_df

####################



# Data table for the **United States**

# In[50]:


us_table = country_table('US')
us_table.style.background_gradient(cmap='Oranges')


# Data table for **India**

# In[51]:


india_table = country_table('India')
india_table.style.background_gradient(cmap='Oranges')


# Data table for **Brazil**

# In[52]:


brazil_table = country_table('Brazil')
brazil_table.style.background_gradient(cmap='Oranges')


# Data table for **Russia**

# In[53]:


russia_table = country_table('Russia')
russia_table.style.background_gradient(cmap='Oranges')


# Data table for **United Kingdom**

# In[54]:


uk_table = country_table('United Kingdom')
uk_table.style.background_gradient(cmap='Oranges')


# Data table for **France**

# In[55]:


france_table = country_table('France')
france_table.style.background_gradient(cmap='Oranges')


# Data table for **Italy**

# In[56]:


italy_table = country_table('Italy')
italy_table.style.background_gradient(cmap='Oranges')


# Data table for **Germany**

# In[57]:


germany_table = country_table('Germany')
germany_table.style.background_gradient(cmap='Oranges')


# Data table for **Spain**

# In[58]:


spain_table = country_table('Spain')
spain_table.style.background_gradient(cmap='Oranges')


# Data table for **China**

# In[59]:


china_table = country_table('China')
china_table.style.background_gradient(cmap='Oranges')


# Data table for **Mexico**

# In[60]:


mexico_table = country_table('Mexico')
mexico_table.style.background_gradient(cmap='Oranges')




###########################


# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category
visual_unique_countries = []
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])

visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)




########################


# Only show 10 provinces with the most confirmed cases, the rest are grouped into the other category
visual_unique_provinces = []
visual_confirmed_cases2 = []
others = np.sum(province_confirmed_cases[10:])
for i in range(len(province_confirmed_cases[:10])):
    visual_unique_provinces.append(unique_provinces[i])
    visual_confirmed_cases2.append(province_confirmed_cases[i])

visual_unique_provinces.append('Others')
visual_confirmed_cases2.append(others)


#####################
