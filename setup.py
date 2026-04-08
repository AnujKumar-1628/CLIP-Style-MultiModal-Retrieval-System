from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("-"):
                requirements.append(line)
    return requirements


setup(
    name="clip-style-multimodal-retrieval-system",
    version="0.1.0",
    description="CLIP-style multimodal retrieval system",
    author="anuj kumar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)
