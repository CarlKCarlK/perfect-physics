from setuptools import setup

# Version number
version = "0.1.3"


def readme():
    with open("README.md") as f:
        return f.read()


install_requires = [
    "sympy",
    "numpy",
    "matplotlib",
]


setup(
    name="perfect-physics",
    version=version,
    description="Perfect Physics",
    long_description=readme(),
    long_description_content_type="text/markdown",
    license="MIT or Apache-2.0",
    # project_urls={
    #     "Bug Tracker": "https://github.com/fastlmm/PySnpTools/issues",
    #     "Documentation": "http://fastlmm.github.io/PySnpTools",
    #     "Source Code": "https://github.com/fastlmm/PySnpTools",
    # },
    url="https://towardsdatascience.com/perfect-infinite-precision-game-physics-in-python-part-1-698211c08d95",
    author="Carl Kadie",
    packages=["perfect_physics"],  # basically, everything with a __init__.py
    install_requires=install_requires,
)
