# from abc import ABC, abstractmethod
# from ml_engine.validation_strategy import choose_strategy, run_validation


# class ModelTemplate(ABC):

#     def __init__(self, config: dict):
#         """
#         config = {
#             "init": model init params,
#             "task": classification / regression / clustering
#         }
#         """
#         self.config = config
#         self.model = None

#     @abstractmethod
#     def build(self):
#         pass

#     @abstractmethod
#     def metric(self, y_true, y_pred):
#         pass

#     def run(self, X, y=None):

#         self.build()

#         strategy = choose_strategy(
#             X=X,
#             y=y,
#             task=self.config["task"]
#         )

#         metrics = run_validation(
#             model=self.model,
#             X=X,
#             y=y,
#             strategy=strategy,
#             metric_fn=self.metric,
#             task=self.config["task"]
#         )

#         return {
#             "model": self.model,
#             "metrics": metrics,
#             "validation_strategy": strategy
#         }

# from abc import ABC, abstractmethod
# from ml_engine.validation_strategy import choose_strategy, run_validation


# class ModelTemplate(ABC):

#     def __init__(self, config: dict):
#         self.config = config
#         self.model = None

#     @abstractmethod
#     def build(self):
#         pass

#     def metric(self, y_true, y_pred):
#         return None

#     def run(self, X, y=None):

#         self.build()

#         strategy = choose_strategy(
#             X=X,
#             y=y,
#             task=self.config["task"]
#         )

#         validation_result = run_validation(
#             model=self.model,
#             X=X,
#             y=y,
#             strategy=strategy,
#             metric_fn=self.metric,
#             task=self.config["task"]
#         )

#         return {
#             "model": self.model,
#             "result": validation_result,
#             "validation_strategy": strategy
#         }

from abc import ABC, abstractmethod
from ml_engine.validation_strategy import choose_strategy, run_validation


class ModelTemplate(ABC):

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    @abstractmethod
    def build(self):
        pass

    def metric(self, y_true, y_pred):
        return None

    def run(self, X, y=None):

        self.build()

        strategy = choose_strategy(
            X=X,
            y=y,
            task=self.config["task"]
        )

        validation_result = run_validation(
            model=self.model,
            X=X,
            y=y,
            strategy=strategy,
            metric_fn=self.metric,
            task=self.config["task"]
        )

        return {
            "model": self.model,
            "result": validation_result,
            "validation_strategy": strategy
        }