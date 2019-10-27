# Makefile

build: cleanbuild
	~/.virtualenvs/python3/bin/python3 setup.py bdist_egg
	mv dist/mltools-dev-py3.6.egg dist/mltools.egg

cleanbuild:
	@rm -rf dist/ build/ mltools.egg-info

test:
	PYTHONPATH=. pytest

