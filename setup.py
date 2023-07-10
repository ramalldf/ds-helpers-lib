# setup.py placed at root directory
import pkg_resources
from setuptools import setup

setup(
    name='ds_helpers',
    version='0.2.0',
    author='',
    description="This is a helper's library for data exploration and ML workflows",
    url='',
    keywords='',
    python_requires='>=3.7, <4',
    install_requires=["matplotlib==3.7.0",
                      "pandas==1.3.5",
                      "pytest==7.2.0",
                      "scikit-learn",
                      "seaborn",
                      "scikit-image"],
    include_package_data=True,
    package_data={'': ['data/*.csv']}
)