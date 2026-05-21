from setuptools import setup, find_packages

setup(
    name="fibfl",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
)
