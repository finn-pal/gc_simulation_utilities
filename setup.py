import os

from setuptools import setup


def get_requirements():
    """
    Read the requirements from a file
    """
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as req:
            for line in req:
                # skip commented lines
                if not line.startswith("#"):
                    requirements.append(line.strip())
    return requirements


setup(
    name="gc_simulation_utilities",  # the name of the module
    packages=["src"],  # the location of the module
    version=0.1,
    install_requires=get_requirements(),
    python_requires=">=3.11",
)
