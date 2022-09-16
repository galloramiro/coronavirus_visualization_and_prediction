from data_processing import DataProcessing
from dataframes import Dataframes
from graphics import Graphics
from ml_models import MLModels


class CoronavirusService:

    def __init__(
            self, dataframes: Dataframes, data_processing: DataProcessing, ml_models: MLModels, graphics: Graphics
    ):
        self.dataframes = dataframes
        self.data_processing = data_processing
        self.ml_models = ml_models
        self.graphics = graphics

    def do_svm_check_against_test_data(self):
        # obtencion de datos
        confirmed_df = self.dataframes.get_confirmed_df()
        deaths_df = self.dataframes.get_deaths_df()

        # procesamiento de datos
        confirmed_df = self.data_processing.remove_unused_cols_from_confirmed_df(confirmed_df=confirmed_df)
        confirmed_keys = confirmed_df.keys()  # former ck
        days_since_1_22 = self.data_processing.get_days_since_1_22(confirmed_keys=confirmed_keys)
        world_cases = self.data_processing.get_world_cases_per_day(confirmed_df=confirmed_df)

        # obtencion del modelo predictivo
        _, X_test_confirmed, _, y_test_confirmed = self.ml_models.get_train_test_split(
            days_since_1_22=days_since_1_22,
            world_cases=world_cases,
            days_to_skip=830
        )
        svm_confirmed = self.ml_models.get_trained_svm()

        # ploteo de resultados del modelo predictivo
        self.graphics.svm_check_against_test_data(
            svm_confirmed=svm_confirmed,
            X_test_confirmed=X_test_confirmed,
            y_test_confirmed=y_test_confirmed
        )

    @staticmethod
    def build():
        dataframes = Dataframes()
        data_processing = DataProcessing()
        ml_models = MLModels()
        graphics = Graphics()
        return CoronavirusService(
            dataframes=dataframes,
            data_processing=data_processing,
            ml_models=ml_models,
            graphics=graphics
        )
