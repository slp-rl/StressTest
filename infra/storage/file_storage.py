import json
import os
from pathlib import Path


class FileStorage:
    def __init__(self, storage_path):
        assert os.path.exists(storage_path), f"Path {storage_path} does not exist"
        self.storage_path = storage_path

    def path_exists(self, path):
        return os.path.exists(f"{self.storage_path}/{path}")

    def create_path_if_not_exists(self, path):
        if not self.path_exists(path):
            Path(f"{self.storage_path}/{path}").mkdir(parents=True, exist_ok=True)

    def save_json(self, file_name, data):
        with open(f"{self.storage_path}/{file_name}", "w") as file:
            json.dump(data, file)

    def load_json(self, file_name):
        file_path = f"{self.storage_path}/{file_name}"
        if not self.path_exists(file_name):
            raise FileNotFoundError(f"File {file_path} not found")
        with open(f"{file_path}", "r") as file:
            return json.load(file)

    def save_wav(self, file_name, data):
        with open(f"{self.storage_path}/{file_name}", "wb") as file:
            file.write(data)

    def load_wav(self, file_name):
        with open(f"{self.storage_path}/{file_name}", "rb") as file:
            return file.read()
