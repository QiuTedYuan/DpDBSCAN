#!/bin/bash

mkdir -p output
python main.py -d circles --info --linear >> output/circles.out 2>&1
python main.py -d moons --info --linear >> output/moons.out 2>&1
python main.py -d blobs --info --linear >> output/blobs.out 2>&1
python main.py -d cluto_t4 --info --linear >> output/cluto_t4.out 2>&1
python main.py -d cluto_t5 --info --linear >> output/cluto_t5.out 2>&1
python main.py -d cluto_t7 --info --linear >> output/cluto_t7.out 2>&1
python main.py -d crash --info --linear >> output/crash.out 2>&1
python main.py -d cabspot --info --linear >> output/cabspot.out 2>&1