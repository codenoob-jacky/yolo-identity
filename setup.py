"""
Setup script for Construction Site Safety Detection project
"""

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="construction-site-safety-detection",
    version="1.0.0",
    author="Safety Detection Team",
    author_email="safety@example.com",
    description="YOLO-based system for detecting safety violations on construction sites",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/construction-site-safety-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "safety-detect=src.detection:main",
            "train-safety-model=src.train_safety_model:main",
            "prepare-safety-data=utils.data_preparation:main",
        ],
    },
)