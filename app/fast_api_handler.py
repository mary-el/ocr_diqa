"""Класс FastApiHandler, который обрабатывает запросы API."""

from catboost import CatBoostClassifier


class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запрос и возвращает предсказание."""

    def __init__(self):
        """Инициализация переменных класса."""

        # типы параметров запроса для проверки
        self.param_types = {
            "user_id": str,
            "model_params": dict
        }

        self.model_path = "models/catboost_churn_model.bin"
        self.load_churn_model(model_path=self.model_path)

        # необходимые параметры для предсказаний модели оттока
        self.required_model_params = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Type', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'days', 'services'
        ]

        def load_churn_model(self, model_path: str):
            """Загружаем обученную модель оттока.
            Args:
                model_path (str): Путь до модели.
            """

        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")

    def churn_predict(self, model_params: dict) -> float:
        """Предсказываем вероятность оттока.

        Args:
            model_params (dict): Параметры для модели.

        Returns:
            float - вероятность оттока от 0 до 1
        """
        # ваш код здесь

    def check_required_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора параметров.

        Args:
            query_params (dict): Параметры запроса.

        Returns:
                bool: True - если есть нужные параметры, False - иначе
        """
        # ваш код здесь

    def check_required_model_params(self, model_params: dict) -> bool:
        """Проверяем параметры пользователя на наличие обязательного набора.

        Args:
            model_params (dict): Параметры пользователя для предсказания.

        Returns:
            bool: True - если есть нужные параметры, False - иначе
        """
        # ваш код здесь


def validate_params(self, params: dict) -> bool:
    """Разбираем запрос и проверяем его корректность.

    Args:
        params (dict): Словарь параметров запроса.

    Returns:
        - **dict**: Cловарь со всеми параметрами запроса.
    """
    if self.check_required_query_params(params):
        print("All query params exist")
    else:
        print("Not all query params exist")
        return False

    if self.check_required_model_params(params["model_params"]):
        print("All model params exist")
    else:
        print("Not all model params exist")
        return False
    return True


def handle(self, params):
    """Функция для обработки запросов API параметров входящего запроса.

    Args:
        params (dict): Словарь параметров запроса.

    Returns:
        dict: Словарь, содержащий результат выполнения запроса.
    """
    try:
    # Ваш код здесь
    # Формат response такой:
    # response = {
    #    "user_id": ...,
    #    "probability": ..., # float от 0 до 1
    #    "is_churn": ... # 0 или 1
    # }

    except Exception as e:
        print(f"Error while handling request: {e}")
    else:
        return response