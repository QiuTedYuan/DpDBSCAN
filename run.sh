#!/bin/bash

mkdir -p output
python main.py -d circles >> output/circles.out 2>&1
python main.py -d moons >> output/moons.out 2>&1
python main.py -d blobs >> output/blobs.out 2>&1
python main.py -d cluto_t4 >> output/cluto_t4.out 2>&1
python main.py -d cluto_t5 >> output/cluto_t5.out 2>&1
python main.py -d cluto_t7 >> output/cluto_t7.out 2>&1
python main.py -d crash >> output/crash.out 2>&1
python main.py -d cabspot >> output/cabspot.out 2>&1   # takes minutes