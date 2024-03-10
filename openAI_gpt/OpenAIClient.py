from openai import OpenAI
import os

class CreateClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance = OpenAI(api_key='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')
        return cls._instance
