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
    name="gc_simulation_utilities",
    version="0.1",
    packages=["gc_utils"],
    package_dir={"gc_utils": "src"},
    install_requires=get_requirements(),
    python_requires=">=3.11",
    url="https://github.com/finn-pal/gc_simulation_utilities",
    author="Finn Pal",
    author_email="f.pal@unsw.edu.au",
    description="Functions to help analyse globular cluster simulation models.",
)
