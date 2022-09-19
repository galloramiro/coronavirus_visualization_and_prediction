from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Graphics:

    @staticmethod
    def svm_check_against_test_data(svm_confirmed, X_test_confirmed, y_test_confirmed):
        # check against testing data
        svm_test_pred = svm_confirmed.predict(X_test_confirmed)
        plt.plot(y_test_confirmed)
        plt.plot(svm_test_pred)
        plt.legend(['Test Data', 'SVM Predictions'])
        print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
        print('MSE:', mean_squared_error(svm_test_pred, y_test_confirmed))
