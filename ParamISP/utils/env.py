from dotenv import load_dotenv as load
from os import getenv as get


def get_or_throw(key: str) -> str:
    """ Get an environment variable, throw error if it doesn't exist. """
    if value := get(key):
        return value
    raise KeyError(f"Environment variable '{key}' not found.")
