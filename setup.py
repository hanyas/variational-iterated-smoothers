from setuptools import setup

setup(
    name="variational_smoothers",
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
    packages=["variational_smoothers"],
    zip_safe=False,
)
