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
    return f"{repo_dir}/examples/"

@fixture(scope="session")
def input_data_dir(request, example_dir):
    return f"{example_dir}/inputData/"

@fixture(scope="session")
def ground_truth_dir(request, example_dir):
    return f"{example_dir}/groundTruthOutput/"

@fixture(scope="session")
def ground_truth_vel_anomaly_dir(request, ground_truth_dir):
    return f"{example_dir}/GroundTruthProject/stagyyModel/2/seism_vel-fields/"

