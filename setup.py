from setuptools import setup, find_packages

setup(
    name="leachates_classification_model",
    version="0.0.1",
    author="@mikdevio",
    author_email="tuemail@ejemplo.com",
    description="Breve descripción del proyecto",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)