# Makefile

build: cleanbuild
	~/.virtualenvs/python3/bin/python3 setup.py bdist_egg

cleanbuild:
	@rm -rf dist/ build/ mltools.egg-info

