from openai import OpenAI
import os

class CreateClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance = OpenAI(api_key='YOUR OPENAI KEY')
        return cls._instance
