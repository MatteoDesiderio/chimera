#!/usr/bin/env python3
"""
Created on Thu Aug 15 18:32:09 2024.

@author: matteo
"""
from pathlib import Path

from pytest import fixture


# Paths to be shared among all tests in test session
@fixture(scope="session")
def repo_dir():
    return Path(__file__).parent.parent.resolve().as_posix()

@fixture(scope="session")
def example_dir(request, repo_dir):
    return f"{repo_dir}/examples/"

# These are not strictly necessary but convenient
@fixture(scope="session")
def input_data_dir(request, example_dir):
    return f"{example_dir}/inputData/"

@fixture(scope="session")
def ground_truth_dir(request, example_dir):
    return f"{example_dir}/groundTruthOutput/"

@fixture(scope="session")
def ground_truth_vel_anomaly_dir(request, ground_truth_dir):
    return (f"{ground_truth_dir}/GroundTruthProject" +
            "/stagyyModel/2/seism_vel-fields/")

@fixture(scope="session")
def project_path(tmp_path_factory):
    return tmp_path_factory.mktemp("temporary_project_path")

@fixture(scope="session")
def thermo_data_description():
    return "ExamplePerplexTables"

def pytest_collection_modifyitems(items):
    """
    Modifies test items in place to ensure
    test modules run in a given order.
    """
    MODULE_ORDER = ["tests.test_thermo_data",
                    "tests.test_analysis_project",
                    "tests.test_compute_analysis"]

    module_mapping = {item: item.module.__name__ for item in items}

    sorted_itms = items.copy()
    # Iteratively move tests of each module to the end of the test queue
    for module in MODULE_ORDER:
        sorted_itms= ([i for i in sorted_itms if module_mapping[i] != module] +
                      [i for i in sorted_itms if module_mapping[i] == module])
    items[:] = sorted_itms
