#
# Makefile for CatchCore
#
# Copyright (c), 2018-2020
#

all: clean demo

clean:
	rm -rf output
	mkdir output
	find ./ -name *.pyc | xargs rm -rf
	@echo

demo:
	@echo "run demo."
	./demo.sh
	@echo

tar:
	rm -rf output/*
	@echo [Packaging]
	./package.sh
	@echo [Packaging] "done."
	@echo
