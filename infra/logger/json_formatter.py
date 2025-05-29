import json
import logging
from time import strftime, gmtime


class JsonFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[33;32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def __init__(self):
        super().__init__()

    @staticmethod
    def serialize_to_json(data):
        try:
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Failed to serialize data to JSON: {str(data)}\nError: {str(e)}"

    def format(self, record):
        error_json = (
            {"error": self.formatException(record.exc_info)}
            if record.levelno == logging.ERROR and record.exc_info
            else {}
        )
        context = record.__dict__["context"]
        json_record = {
            "message": record.getMessage(),
            "level": record.levelname,
            "logged_at": strftime("%Y-%m-%d %H:%M:%S", gmtime(record.created)),
            **context,
            **error_json,
        }
        try:
            json_log = f"{self.FORMATS.get(record.levelno)}{json.dumps(json_record, indent=2)}{self.reset}"
            colorful_json = json_log.encode("utf-8").decode("unicode_escape")
            return colorful_json
        except Exception as e:
            return (
                f"Failed to serialize data to JSON: {str(json_record)}\nError: {str(e)}"
            )
