#!/bin/bash

paths=(include/*.hpp src/**/*.cpp src/**/*cu)
for i in "${paths[@]}"
do
	clang-format -i --verbose $i
done
