from abc import ABC, abstractmethod
from typing import Any
from monitoring.logger import get_logger


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's primary task."""

    def log_info(self, msg: str, **extra):
        self.logger.info(msg, extra=extra)

    def log_error(self, msg: str, **extra):
        self.logger.error(msg, extra=extra)

    def log_warning(self, msg: str, **extra):
        self.logger.warning(msg, extra=extra)
