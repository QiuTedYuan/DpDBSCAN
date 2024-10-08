#!/bin/bash

mkdir -p output/out
echo "Running Circles..." && python main.py -d circles --info --linear >> output/out/circles.out 2>&1 && echo "Done!"
echo "Running Moons..." && python main.py -d moons --info --linear >> output/out/moons.out 2>&1 && echo "Done!"
echo "Running Blobs..." && python main.py -d blobs --info --linear >> output/out/blobs.out 2>&1 && echo "Done!"
echo "Running Cluto-t4..." && python main.py -d cluto_t4 --info --linear >> output/out/cluto_t4.out 2>&1 && echo "Done!"
echo "Running Cluto-t5..." && python main.py -d cluto_t5 --info --linear >> output/out/cluto_t5.out 2>&1 && echo "Done!"
echo "Running Cluto-t7..." && python main.py -d cluto_t7 --info --linear >> output/out/cluto_t7.out 2>&1 && echo "Done!"
echo "Running Crash..." && python main.py -d crash --info --linear >> output/out/crash.out 2>&1 && echo "Done!"
echo "Running Cabs-tiny..." && python main.py -d cabs_tiny --info --linear >> output/out/cabs_tiny.out 2>&1 && echo "Done!"
echo "Running Cabs..." && python main.py -d cabs --info --linear >> output/out/cabs.out 2>&1 && echo "Done!"