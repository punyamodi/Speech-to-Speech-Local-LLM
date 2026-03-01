from abc import ABC, abstractmethod


class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: str, **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
