# DP-DBSCAN

This repo contains code for Approximate DBSCAN under Differential Privacy. The experiments were implemented in Python v3.12.4.

## Installation

Download the repo (datasets included) and clone submodules.

```zsh
cd db-dpscan
git submodule add https://github.com/deric/clustering-benchmark.git
pip install -r requirements.txt

cd datasets
unzip crashes_240928.zip
cd ..
```

## Datasets

The Moons/Blobs/Circles datasets are generated using `sklearn.datasets` as in [here](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py).

The Cluto-t4/t5/t7 datasets and their true labels are from the [clustering-benchmark](https://github.com/deric/clustering-benchmark) repo. 

The CabSpot dataset is obtained from [IEEE DataPort](https://ieee-dataport.org/open-access/crawdad-epflmobility), this repo contains a copy for reproducibility.

The Crash dataset is obtained from [NYC OpenData](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data), this repo contains a compressed copy for reproducibility.

The HAR70+ dataset is obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/780/har70), this repo contains a copy for reproducibility.

## Run Experiments
```zsh
python main.py -h
```

### Parameters

| Arg                | Default    | Usage                      | Candidates                                                                       |
|--------------------|------------|----------------------------|----------------------------------------------------------------------------------|
| -d                 | moons      | dataset                    | moons, blobs, circles, cluto_t4, cluto_t5, cluto_t7, cabs, cabs_tiny, crash, har |
| -s                 | 0          | random seed                | int                                                                              |
| --noise            | Laplace    | type of the noise          | Laplace, Geometric, Gaussian                                                     |
| --epsilon, --delta | 1, 0       | DP parameters              | float                                                                            |
| --beta             | 0.5        | fail prob. for bound       | 0~1                                                                              |
| --alpha, --minpts  | by dataset | override DBSCAN parameters | float, int                                                                       |
| --grid_scale       | 1          | override grid_scale        | 0~1 (eta' in paper)                                                              |

### Algorithm Choices

| Flag              | Default        | Usage                              |
|-------------------|----------------|------------------------------------|
| --linear, --naive | linear if n<2X | force using linear/naive histogram |
| --skip_dbscan     | False          | skip running original DBSCAN       |
| --skip_dp_dbscan  | False          | skip running DP-DBSCAN             |
| --skip_kmeans     | False          | skip running kmeans                |
| --skip_dp_kmeans  | False          | skip running DP-kmeans             |
| --run_trivial     | False          | run the trivial algorithm          |

### Debugging and Plotting
| Arg             | Default       | Usage          | Candidates    |
|-----------------|---------------|----------------|---------------|
| --debug, --info | logging.ERROR | log level      |               |
| -p, --plot      | False         | plot figures   |               |
| --dpi           | 100           | dpi of figures | int           |
| --ext           | png           | figure format  | png, pdf, svg |


##  Results

| Dataset                       | n        | low                             | high                         | alpha              | minpts |                                                        |                   
|-------------------------------|----------|---------------------------------|------------------------------|--------------------|--------|--------------------------------------------------------|
| [moons](output/moons)         | 2000     | (-1.84, 1.80)                   | (1.83, 1.74)                 | 0.2                | 7      | `python main.py -d moons --info`                       |
| [circles](output/circles)     | 2000     | (-2.04, -1.95)                  | (1.98, 2.01)                 | 0.2                | 10     | `python main.py -d circles --info`                     |
| [blobs](output/blobs)         | 2000     | (-1.48, -2.82)                  | (2.89, 2.46)                 | 0.2                | 10     | `python main.py -d blobs --info`                       |
| [cluto-t4](output/cluto_t4)   | 8000     | (14.64, 21.38)                  | (634.96, 320.87)             | 9.0                | 11     | `python main.py -d cluto_t4 --info`                    |
| [cluto-t5](output/cluto_t5)   | 8000     | (14.76, 11.05)                  | (803.33, 155.79)             | 9.0                | 20     | `python main.py -d cluto_t5 --info`                    |
| [cluto-t7](output/cluto_t7)   | 8000     | (0.80, 23.06)                   | (696.32, 473.70)             | 12.0               | 20     | `python main.py -d cluto_t7 --info`                    |
| [crash](output/crash)         | 1860785  | (-56.84, 40.51)                 | (-56.42, 40.93)              | (0.1 km / 110 km)  | 300    | `python main.py -d crash --info`                       |
| [cabs_tiny](output/cabs_tiny) | 845685   | (-96.77, 37.72)                 | (-96.70, 37.81)              | (0.05 km / 110 km) | 1000   | `python main.py -d cabs_tiny --info　-p`                |
| [cabs](output/cabs)           | 10995626 | (-96.87, 37.55)                 | (-96.63, 37.85)              | (0.02 km / 110 km) | 500    | `python main.py -d cabs --info --linear --skip_dbscan` |
| har                           | 103860   | (-2.895264 -1.157471 -2.204834) | (0.239746 0.893555 0.979004) | 0.01               | 5      | `python main.py -d har --info`                         |
