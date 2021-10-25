# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools

INSTALL_REQUIRES = []

TEST_REQUIRES = [
    'bandit',
    'flake8',
    'mypy',
    'pylint',
    'pytest-cov',
    'pytest-flake8',
    'pytest-mypy',
    'pytest-pylint',
    'pytest-xdist'
]

setuptools.setup(
    name='zen_nas',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES
    },
    python_requires='>= 3.6'
)
