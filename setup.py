from setuptools import setup, find_packages

setup(
    name="varsmooth",
    version="0.1.0",
    description="Variational Iterated Gaussian Smoothing",
    author="Hany Abdulsamad",
    author_email="hany@robot-learning.de",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "matplotlib",
    ],
    packages=find_packages(exclude=["tests", "examples", "experiments", "build*"]),
    zip_safe=False,
)
