"""Setup configuration for waldo_finder package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="waldo_finder",
    version="2.0.0",
    author="Noah",
    description="Modern JAX-based implementation for finding Waldo using Vision Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noah-ing/theres-waldo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.7.5",
        "optax>=0.1.7",
        "tensorflow>=2.15.0",
        "tensorflow-datasets>=4.9.3",
        "opencv-python>=4.8.1",
        "Pillow>=10.1.0",
        "matplotlib>=3.8.2",
        "plotly>=5.18.0",
        "wandb>=0.16.2",
        "hydra-core>=1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.12.1",
            "isort>=5.13.2",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "find-waldo=waldo_finder.inference:main",
        ],
    },
)
