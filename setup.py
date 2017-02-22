"""SyllableParser setup.py script"""

from setuptools import setup, find_packages

requirements = ["numpy", "pandas", "beautifulsoup4"]

try:
    __import__('tensorflow')
except ImportError:
    requirements.append("tensorflow>=1.0.0")

setup(
    name="syllable_parser",
    description="LSTM-based Tensorflow model for breaking words into syllables.",
    version="0.1",
    install_requires=requirements,
    packages=find_packages(),
)
