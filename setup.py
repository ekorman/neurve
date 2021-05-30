from setuptools import find_packages, setup

setup(
    name="neurve",
    version="0.1.0",
    license="MIT",
    packages=find_packages(include=["neurve", "neurve.*"]),
    install_requires=[
        "numpy>=1.17.4",
        "torch>=1.3.1",
        "torchvision>=0.4.2",
        "scipy>=1.5.3",
        "seaborn",
        "matplotlib",
        "tqdm",
        "tensorboardX",
    ],
    extras_require={"wandb": ["wandb"]},
    test_suite="pytest",
    tests_require=["pytest"],
)
