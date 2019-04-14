#!/usr/bin/env python3

from setuptools import setup, find_packages

print(find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))

setup(
    name = "mltools",
    version = "dev",
    packages = find_packages(exclude=["*.tests"])
    )

