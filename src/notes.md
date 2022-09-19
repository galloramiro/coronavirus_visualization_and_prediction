### Broad idea

- single responsability principle
- reuse the functions that you are declarating in this step:
```python
def get_confirmed():
    confirmed_cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, confirmed_cols[4]:]
    return confirmed

def get_confirmed_cases():
    confirmed = get_confirmed()
    return confirmed.keys()
```
- One single big class that relates each other through `self`
- Helper functions from notebook can be staticmethods or single functions outside the class
- focus on the returns. Can they be more granular and depend on each other?

```python
    def get_world_cases(self):
    confirmed = self.get_confirmed()
    ck = confirmed.keys()
    num_dates = len(ck)
    world_cases = []
    for i in range(num_dates):
        confirmed_sum = confirmed[ck[i]].sum()
        world_cases.append(confirmed_sum)
    return world_cases


def get_total_deaths(self):
    confirmed = self.get_confirmed()
    deaths = self.get_deaths()
    ck = confirmed.keys()
    dk = deaths.keys()
    num_dates = len(ck)
    total_deaths = []
    for i in range(num_dates):
        death_sum = deaths[dk[i]].sum()
        total_deaths.append(death_sum)
    return total_deaths


def get_mortality_rate(self):
    world_cases = self.get_world_cases_per_day()
    total_deaths = self.get_total_deaths()
    mortality_rate = [wc / td for wc, td in zip(world_cases, total_deaths)]
    return mortality_rate
```
- For now focus on returning a single thing, or at least very closely related stuff
- Don't worry about writting a lot of lines. It's preferable this:

```python
        world_cases = self.get_world_cases_per_day()
world_confirmed_avg = self.moving_average(world_cases, self.window)
return world_confirmed_avg
```
  to this:

```python
        world_confirmed_avg = self.moving_average(self.get_world_cases_per_day(), self.window)
return world_confirmed_avg
```
and on python execution it is exaclty the same (REFERENCE?)
- Carefull with unmutable objects when operations are made!
- When calling sub-routines, pass on arguments
```python
def func_5_get_future_forecast_and_adjusted_dates(ck):
    days_in_future = 10
    future_forcast = np.array([i for i in range(len(ck) + days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-10]
    return future_forcast, adjusted_dates
```
to
```python
def get_future_forecast(self, days_in_future=10):
    ck = self.get_confirmed().keys()
    future_forecast = np.array([i for i in range(len(ck) + days_in_future)]).reshape(-1, 1)
    return future_forecast

def get_adjusted_dates(self, days_in_future=10):
    future_forecast = self.get_future_forecast(days_in_future=days_in_future)
    adjusted_dates = future_forecast[:-10]
    return adjusted_dates
```
- every once in a while review the functions made. Do they make sense?
- Consejo: tratá de no ser distraido al hacer esta tarea. Hacerlo por intervalos de tiempo. Pomodoro?
- si llamo a la misma función obtengo siempre lo mismo? Ej: train test split.
  Tal vez tratarlo aparte. En su propia clase.
- classes with constants? go for it!
  


Next steps:
- cerrar la refactorizacion_3
- Agregar servicio principal que hace la magia desde una sola llamada a una función
- separar branchs:
  - master: proyecto como está ahora. Con entorno de desarrollo
  - 


cosas en el repo:
- archivo original limpio
- version corta del notebook (short 1)
- 1 archivo por cada clase
- servicio ppal
- makefile o algo
- short - paso a paso con readme


procedimiento:
- darles clases bases con una función de ejemplo y la función a completar. Orientandose del notebook
- Ellos llenan cada una de las clases necesarias: Servicio ppal, y A, B, C, D, Cuando aplique.


- 

