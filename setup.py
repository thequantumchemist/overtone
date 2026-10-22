import re
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")
version = re.search(
    r'^__version__ = "([^"]+)"',
    (this_directory / "src" / "overtone" / "__init__.py").read_text(encoding="utf-8"),
    re.M,
).group(1)

setup(
    name="overtone",
    version=version,
    description=(
        "Anharmonic vibrational spectra (fundamentals, overtones, combination "
        "bands, IR/NIR intensities) with VPT2 for any ASE calculator"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thequantumchemist/overtone",
    author="Basila Bhanu Pattani Ameerjan, Mads-Peter Verner Christiansen and Wilke Dononelli",
    author_email="wido@uni-bremen.de",
    license="GPL-3.0-only",  # SPDX identifier; use "GPL-3.0-or-later" to allow later versions
    license_files=["LICENSE"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "ase>=3.23",
        "pydantic>=2.0",
        "rich",
        "matplotlib",          # spectrum plotting is part of the core package
    ],
    extras_require={
        "plot": [],                        # kept for compatibility; matplotlib is now core
        "cluster": ["shephex"],            # multi-node / queueing system runs
        "falcon": ["falcon-md"],           # on-the-fly active learning
        "tutorial": ["tblite"],
        "test": ["pytest", "tblite"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
