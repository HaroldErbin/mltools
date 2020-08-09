# Makefile

.PHONY: build cleanbuild upload build-upload test

build: cleanbuild
	~/.virtualenvs/python3/bin/python3 setup.py bdist_egg
	mv dist/mltools-*.egg dist/mltools.egg

cleanbuild:
	@rm -rf dist/ build/ mltools.egg-info

upload:
	rclone copy dist/mltools.egg gdrive:ML/
	lftp melsophia -e "put -O wp/ dist/mltools.egg; quit"

build-upload: build upload

test:
	PYTHONPATH=. pytest
