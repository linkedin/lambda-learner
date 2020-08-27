from setuptools import find_namespace_packages, setup

from distgradle import GradleDistribution


setup(
    distclass=GradleDistribution,
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    include_package_data=True,
)
