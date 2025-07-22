#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy', 'scipy', 'matplotlib', 'plotly', 'sympy'
]
# with open('requirements.txt', encoding='utf8') as req:
#     requirements = req.readlines()

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    author="Jesus del Hoyo Munoz / Luis Miguel Sanchez Brea",
    author_email='optbrea@ucm.es',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Jones and Mueller polarization - Optics",
    entry_points={
        'console_scripts': [
            'py_pol=py_pol.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=[
        'py_pol', 'optics', 'polarization', 'Jones', 'Stokes', 'Mueller'
    ],
    name='py_pol',
    packages=find_packages(include=['py_pol']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JesusDelHoyo/py_pol',
    version='1.3.0',
    zip_safe=False,
)
