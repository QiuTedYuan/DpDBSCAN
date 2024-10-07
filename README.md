# DP-DBSCAN

This repo contains code for DBSCAN under Pure Differential Privacy. The experiments were implemented in Python v3.12.4.

## Installation

```zsh
git clone --depth 1 --recurse-submodules <url>
cd dbscan
pip install -r requirements.txt
```

## Datasets

The Moons/Blobs/Circles datasets are generated `sklearn.datasets` as in [here](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py).

The Cluto-t4/t5/t7 datasets and their true labels are from the [clustering-benchmark](https://github.com/deric/clustering-benchmark) repo. 

The CabSpot dataset is obtained from [IEEE DataPort](https://ieee-dataport.org/open-access/crawdad-epflmobility), this repo contains a copy for reproducibility.

The Crash dataset is obtained from [NYC OpenData](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data), this repo contains a compressed copy for reproducibility.
Unzip the data before using.
```zsh
cd datasets
unzip crashes_240928.zip
cd ..
```

## Run Experiments

```
$ python main.py -h
usage: main.py [-h]
               [-d {moons,blobs,circles,cluto_t4,cluto_t5,cluto_t7,cabspot_ends,cabspot_raw,crash}]
               [-s SEED] [--noise {Laplace,Geometric,Gaussian}]
               [--epsilon EPSILON] [--delta DELTA] [--beta BETA]
               [--alpha ALPHA] [--minpts MINPTS] [--grid_scale GRID_SCALE]
               [-p] [--debug] [--info] [--linear] [--skip_dbscan]
               [--skip_kmeans] [--skip_dp_dbscan] [--skip_dp_kmeans]

DPDBSCAN Experiments.

options:
  -h, --help            show this help message and exit
  -d {moons,blobs,circles,cluto_t4,cluto_t5,cluto_t7,cabspot_ends,cabspot_raw,crash}, --dataset {moons,blobs,circles,cluto_t4,cluto_t5,cluto_t7,cabspot_ends,cabspot_raw,crash}
                        Dataset, default moons
  -s SEED, --seed SEED  Random Seed, default 0
  --noise {Laplace,Geometric,Gaussian}
                        Noise to generate DP histogram, default Laplace
  --epsilon EPSILON     Epsilon for DP mechanisms, default 1
  --delta DELTA         Delta for DP mechanisms, default 0
  --beta BETA           Beta for Calculating Error Bounds, default 0.5
  --alpha ALPHA         override alpha for DBSCAN
  --minpts MINPTS       override minpts for DBSCAN
  --grid_scale GRID_SCALE
                        override grid_scale for DP-DBSCAN
  -p, --plot            plot results
  --debug               log level debug
  --info                log level info
  --linear              use linear time histogram
  --skip_dbscan
  --skip_kmeans
  --skip_dp_dbscan
  --skip_dp_kmeans
```

### Parameters

| Arg                | Default          | Usage                      | Candidates                                                                            |
|--------------------|------------------|----------------------------|---------------------------------------------------------------------------------------|
| -d                 | moons            | dataset                    | moons, blobs, circles, cluto_t4, cluto_t5, cluto_t7, cabspot_raw, cabspot_ends, crash |
| -s                 | 0                | random seed                | integer                                                                               |
| --noise            | Laplace          | type of the noise          | Laplace, Geometric, Gaussian                                                          |
| --epsilon, --delta | 1, 0             | DP parameters              | positive                                                                              |
| --beta             | 0.5              | fail prob. for bound       | 0~1                                                                                   |
| --alpha, --minpts  | by dataset       | override DBSCAN parameters | float, int                                                                            |
| --grid_scale       | 1, or by dataset | override grid_scale        | 0~1 (eta' in paper)                                                                   |

### Flags

| Flag                                                             | Default       | Usage                                                 |
|------------------------------------------------------------------|---------------|-------------------------------------------------------|
| -p, --plot                                                       | False         | plot figures                                          |
| --debug, --info                                                  | logging.ERROR | log level                                             |
| --linear                                                         | False         | use linear-time histogram (otherwise naive histogram) |
| --skip_dbscan, --skip_kmeans, --skip_dp_dbscan, --skip_dp_kmeans | False         | skip running the algorithm                            |

##  Example
```zsh
mkdir -p output
python main.py -d moons -p
```
ARI/AMI scores will be printed to the console and `./output/<CurrentTime>` will contain the figures.
For running time, run with the `--info` flag.
```zsh
python main.py -d moons --info
```
