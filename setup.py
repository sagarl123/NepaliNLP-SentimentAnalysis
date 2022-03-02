from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        "numpy==1.22.1",
        "pandas==1.4.0",
        "torch==1.10.2",
        "scikit_learn",
        "transformers==4.16.1",
        "scipy==1.7.3",
    ],
)