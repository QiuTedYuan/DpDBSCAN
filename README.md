# DP-DBSCAN

## Clone and get data

```
$ git clone --depth 1 --recurse-submodules <url>
$ cd dbscan
$ cd datasets
$ unzip crashes_240928.zip
$ cd ..
$ python main.py -h
usage: main.py [-h]
               [-d {moons,blobs,circles,cluto_t4,cluto_t5,cluto_t7,cabspot,crash}]
               [-s SEED] [--noise {Laplace,Geometric,Gaussian}]
               [--epsilon EPSILON] [--delta DELTA] [--beta BETA] [-p]
               [--skip_dbscan] [--skip_kmeans] [--skip_dp_dbscan]
               [--skip_dp_kmeans]

DPDBSCAN Experiments.

options:
  -h, --help            show this help message and exit
  -d {moons,blobs,circles,cluto_t4,cluto_t5,cluto_t7,cabspot,crash}, --dataset {moons,blobs,circles,cluto_t4,cluto_t5,cluto_t7,cabspot,crash}
                        Dataset, default moons
  -s SEED, --seed SEED  Random Seed, default 0
  --noise {Laplace,Geometric,Gaussian}
                        Noise to generate DP histogram, default Laplace
  --epsilon EPSILON     Epsilon for DP mechanisms, default 1
  --delta DELTA         Delta for DP mechanisms, default 0
  --beta BETA           Beta for Calculating Error Bounds, default 0.5
  -p, --plot            plot results
  --skip_dbscan
  --skip_kmeans
  --skip_dp_dbscan
```

## Run Experiments

```
mkdir output
python main.py -d circles >> output/circles.out 2>&1
python main.py -d moons >> output/moons.out 2>&1
python main.py -d blobs >> output/blobs.out 2>&1
python main.py -d cluto_t4 >> output/cluto_t4.out 2>&1
python main.py -d cluto_t5 >> output/cluto_t5.out 2>&1
python main.py -d cluto_t7 >> output/cluto_t7.out 2>&1
python main.py -d crash >> output/crash.out 2>&1
python main.py -d cabspot >> output/cabspot.out 2>&1   # takes minutes
```
