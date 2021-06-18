from abc import ABC
from typing import Iterable


class AbstractBaseModel(ABC):
    """Defining the API for all isotherm models"""
    def get_latex(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def fitted(self) -> bool:
        ...

    def plot(self, p: Iterable):
        ...
    
    def get_temperature_dependent_parameters(self):
        ...

    def get_temperature_independent_parameters(self):
        ...