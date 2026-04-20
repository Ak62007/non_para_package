from setuptools import setup, find_packages

setup(
    name='nonparam_safe',
    version='0.1.0',
    description='A safe and robust wrapper for non-parametric statistical tests.',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.5.0',
        'pandas>=1.0.0'
    ],
)
