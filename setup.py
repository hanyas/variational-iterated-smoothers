from setuptools import setup

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
        "jaxopt",
        "typing_extensions",
        "matplotlib",
    ],
    packages=["varsmooth"],
    zip_safe=False,
)
