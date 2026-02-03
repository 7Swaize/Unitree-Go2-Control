from setuptools import setup, find_packages
from pathlib import Path


requirements_path = Path("requirements/runtime.txt")
install_requires = []
if requirements_path.exists():
    with open(requirements_path, encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(install_requires=install_requires)