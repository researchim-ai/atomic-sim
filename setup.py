"""
Setup script for atomic-sim package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="atomic-sim",
    version="1.1.0",
    description="High-fidelity nuclear reactor simulator for ML/RL dataset generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="researchim-ai",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.0.290",
        ],
    },
    entry_points={
        "console_scripts": [
            "atomic-sim=generate_dataset:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

