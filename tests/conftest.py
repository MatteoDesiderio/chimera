#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:32:09 2024

@author: matteo
"""

from pathlib import Path
from pytest import FixtureRequest, fixture

@fixture(scope="session")
def repo_dir():
    return Path(__file__).parent.parent.resolve().as_posix()

@fixture(scope="session")
def example_dir(request, repo_dir):
    return f"{repo_dir}/examples"


