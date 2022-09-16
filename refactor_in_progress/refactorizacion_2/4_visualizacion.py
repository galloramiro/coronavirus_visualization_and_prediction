from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_n_cases_deaths_over_time(
        adjusted_dates, world_cases, world_confirmed_avg, total_deaths, world_death_avg, window: int = 7):
    adjusted_dates = adjusted_dates.reshape(1, -1)[0]
    plt.figure(figsize=(16, 8))
    plt.plot(adjusted_dates, world_cases)
    plt.plot(adjusted_dates, world_confirmed_avg, linestyle='dashed', color='orange')
    plt.title('# of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Worldwide Coronavirus Cases', 'Moving Average {} Days'.format(window)], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(adjusted_dates, total_deaths)
    plt.plot(adjusted_dates, world_death_avg, linestyle='dashed', color='orange')
    plt.title('# of Coronavirus Deaths Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Worldwide Coronavirus Deaths', 'Moving Average {} Days'.format(window)], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


def plot_increase_in_cases_and_deaths(
        adjusted_dates, world_daily_increase, world_daily_increase_avg, world_daily_death, world_daily_death_avg,
        window: int = 7):
    plt.figure(figsize=(16, 10))
    plt.bar(adjusted_dates, world_daily_increase)
    plt.plot(adjusted_dates, world_daily_increase_avg, color='orange', linestyle='dashed')
    plt.title('World Daily Increases in Confirmed Cases', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Moving Average {} Days'.format(window), 'World Daily Increase in COVID-19 Cases'], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.bar(adjusted_dates, world_daily_death)
    plt.plot(adjusted_dates, world_daily_death_avg, color='orange', linestyle='dashed')
    plt.title('World Daily Increases in Confirmed Deaths', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Moving Average {} Days'.format(window), 'World Daily Increase in COVID-19 Deaths'], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


def plot_log_cases_and_deaths(adjusted_dates, world_cases, total_deaths):
    plt.figure(figsize=(16, 10))
    plt.plot(adjusted_dates, np.log10(world_cases))
    plt.title('Log of # of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.plot(adjusted_dates, np.log10(total_deaths))
    plt.title('Log of # of Coronavirus Deaths Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


def plot_predictions(future_forecast, x, y, pred, algo_name, color):
    plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.plot(future_forecast, pred, linestyle='dashed', color=color)
    plt.title('Worldwide Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


def svm_check_against_test_data(svm_confirmed, X_test_confirmed, y_test_confirmed):
    # check against testing data
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    plt.plot(y_test_confirmed)
    plt.plot(svm_test_pred)
    plt.legend(['Test Data', 'SVM Predictions'])
    print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
    print('MSE:', mean_squared_error(svm_test_pred, y_test_confirmed))


def plot_poly_regression(y_test_confirmed, test_linear_pred):
    plt.plot(y_test_confirmed)
    plt.plot(test_linear_pred)
    plt.legend(['Test Data', 'Polynomial Regression Predictions'])


def plot_bayesian_ridge(y_test_confirmed, test_bayesian_pred):
    plt.plot(y_test_confirmed)
    plt.plot(test_bayesian_pred)
    plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])


def plot_pred_svm(adjusted_dates, world_cases, svm_pred):
    plot_predictions(adjusted_dates, world_cases, svm_pred, 'SVM Predictions', 'purple')


def plot_pred_linear(adjusted_dates, world_cases, linear_pred):
    plot_predictions(adjusted_dates, world_cases, linear_pred, 'Polynomial Regression Predictions', 'orange')


def plot_pred_bayesian(adjusted_dates, world_cases, bayesian_pred):
    plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Bayesian Ridge Regression Predictions', 'green')


def plot_future_preds_svm(future_forecast_dates, svm_pred):
    # Future predictions using SVM
    svm_df = pd.DataFrame(
        {'Date': future_forecast_dates[-10:], 'SVM Predicted # of Confirmed Cases Worldwide': np.round(svm_pred[-10:])})
    svm_df.style.background_gradient(cmap='Reds')


def plot_future_preds_linear(future_forecast_dates, linear_pred):
    linear_pred = linear_pred.reshape(1, -1)[0]
    linear_df = pd.DataFrame({'Date': future_forecast_dates[-10:],
                              'Polynomial Predicted # of Confirmed Cases Worldwide': np.round(linear_pred[-10:])})
    linear_df.style.background_gradient(cmap='Reds')


def plot_future_preds_bayesian(future_forecast_dates, bayesian_pred):
    bayesian_df = pd.DataFrame({'Date': future_forecast_dates[-10:],
                                'Bayesian Ridge Predicted # of Confirmed Cases Worldwide': np.round(
                                    bayesian_pred[-10:])})
    bayesian_df.style.background_gradient(cmap='Reds')


def plot_mortality_rate(mortality_rate, adjusted_dates):
    mean_mortality_rate = np.mean(mortality_rate)
    plt.figure(figsize=(16, 10))
    plt.plot(adjusted_dates, mortality_rate, color='orange')
    plt.axhline(y=mean_mortality_rate, linestyle='--', color='black')
    plt.title('Worldwide Mortality Rate of Coronavirus Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Case Mortality Rate', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
