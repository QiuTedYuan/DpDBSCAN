#!/bin/bash

mkdir -p output/out
echo "Running Crash..." && python main.py -d crash --info --linear >> output/out/crash.out 2>&1 && echo "Done!"
echo "Running Cabs-tiny..." && python main.py -d cabs_tiny --info --linear >> output/out/cabs_tiny.out 2>&1 && echo "Done!"
echo "Running Cabs..." && python main.py -d cabs --info --linear >> output/out/cabs.out 2>&1 && echo "Done!"