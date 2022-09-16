from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


def get_trai_test_split(days_since_1_22, world_cases, days_to_skip: int = 830):
    # slightly modify the data to fit the model better (regression models cannot pick the pattern), we are using data from 4/1/22 and onwards for the prediction modeling
    days_to_skip = 830
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(
        days_since_1_22[days_to_skip:], world_cases[days_to_skip:], test_size=0.10, shuffle=False)
    return X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed


def func_1_find_svm_best_params(X_train_confirmed, y_train_confirmed):
    c = [0.01, 0.1, 1]
    gamma = [0.01, 0.1, 1]
    epsilon = [0.01, 0.1, 1]
    shrinking = [True, False]

    svm_grid = {'C': c, 'gamma': gamma, 'epsilon': epsilon, 'shrinking': shrinking}

    svm = SVR(kernel='poly', degree=3)
    svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                    n_jobs=-1, n_iter=30, verbose=1)
    svm_search.fit(X_train_confirmed, y_train_confirmed)
    return svm_search.best_params_


def func_2_get_svm_predictions(X_train_confirmed, y_train_confirmed, future_forecast):
    svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forecast)
    return svm_pred


def func_3_polynomical_regression(X_train_confirmed, X_test_confirmed, y_train_confirmed, future_forcast):
    # transform our data for polynomial regression
    poly = PolynomialFeatures(degree=3)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    bayesian_poly = PolynomialFeatures(degree=3)
    bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
    bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)

    # polynomial regression
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)
    # print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
    # print('MSE:', mean_squared_error(test_linear_pred, y_test_confirmed))
    return bayesian_poly_X_train_confirmed, bayesian_poly_X_test_confirmed, \
           bayesian_poly_future_forcast, linear_pred, test_linear_pred


def func_4_bayesian_ridge(
        bayesian_poly_X_train_confirmed, y_train_confirmed, bayesian_poly_X_test_confirmed,
        bayesian_poly_future_forcast):
    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                     'normalize': normalize}

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3,
                                         return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)

    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    # print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
    # print('MSE:', mean_squared_error(test_bayesian_pred, y_test_confirmed))
    return bayesian_search, test_bayesian_pred, bayesian_pred
