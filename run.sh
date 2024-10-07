#!/bin/bash

mkdir -p output
python main.py -d circles --info >> output/circles.out 2>&1
python main.py -d moons --info >> output/moons.out 2>&1
python main.py -d blobs --info >> output/blobs.out 2>&1
python main.py -d cluto_t4 --info >> output/cluto_t4.out 2>&1
python main.py -d cluto_t5 --info >> output/cluto_t5.out 2>&1
python main.py -d cluto_t7 --info >> output/cluto_t7.out 2>&1
python main.py -d crash --info >> output/crash.out 2>&1
python main.py -d cabspot_ends --info >> output/cabspot.out 2>&1   # takes minutes