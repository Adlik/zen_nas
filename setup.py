# Copyright 2021 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup"""

import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'torch >= 1.8.1',
    'torchvision >= 0.9.1',
    'ptflops',
    'tensorboard >= 1.15',
    'horovod >= 0.22.1',
    'apex'
]

TEST_REQUIRES = [
    'bandit',
    'flake8',
    'mypy==0.812',
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
    python_requires='>= 3.7'
)
