from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='openAI_gpt',
    version='1.0',
    description='Mettalex GPT',
    install_requires=requirements,
)
