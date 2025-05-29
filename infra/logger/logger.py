import sys
import logging
import contextvars
from typing import Dict, Any
from .json_formatter import JsonFormatter


# Create a context variable to store request-specific information
context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "context_dict"
)


class Logger:
    def __init__(self, context=None, use_context_var=False):
        self.logger = logging.getLogger("json_logger")
        self.context = context or {}
        self.base_context = context or {}
        self.use_context_var = use_context_var
        self._setup()

    def _setup(self):
        self.logger.setLevel(logging.DEBUG)
        self.context = (
            context_var.set(self.base_context) if self.use_context_var else self.context
        )
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(console_handler)

    def debug(self, data, context={}):
        self.log(logging.DEBUG, data, context)

    def info(self, data, context={}):
        self.log(logging.INFO, data, context)

    def warning(self, data, context={}):
        self.log(logging.WARNING, data, context)

    def error(self, data, error=None, context={}):
        self.log(logging.ERROR, data, context, error)

    def log(self, level, data, context={}, error=None):
        self.update_context(context=context)
        self.logger.log(
            level,
            msg=data,
            extra={"context": self._get_context()},
            exc_info=error,
        )

    def _get_context(self):
        return context_var.get() if self.use_context_var else self.context

    def reset_context(self):
        if self.use_context_var:
            context_var.set(self.base_context)
        else:
            self.context = self.base_context or {}

    def update_context(self, context):
        if self.use_context_var:
            context_var.set({**context_var.get(), **context})
        else:
            self.context.update(context)
