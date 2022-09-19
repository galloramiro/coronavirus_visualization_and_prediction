from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


class MLModels:
    X_train_confirmed = None
    X_test_confirmed = None
    y_train_confirmed = None
    y_test_confirmed = None

    def get_train_test_split(self, days_since_1_22, world_cases, days_to_skip: int = 830):
        # slightly modify the data to fit the model better (regression models cannot pick the pattern),
        # we are using data from 4/1/22 and onwards for the prediction modeling
        if not self.X_train_confirmed:
            splitted = train_test_split(
                days_since_1_22[days_to_skip:], world_cases[days_to_skip:], test_size=0.10, shuffle=False)
            self.X_train_confirmed, self.X_test_confirmed, self.y_train_confirmed, self.y_test_confirmed = splitted
        return self.X_train_confirmed, self.X_test_confirmed, self.y_train_confirmed, self.y_test_confirmed

    def get_trained_svm(self):
        svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=3, C=0.1)
        svm_confirmed.fit(self.X_train_confirmed, self.y_train_confirmed)
        return svm_confirmed
