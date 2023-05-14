# Dataset setup

Here you can find instructions to setup datasets for use with this code.

## Cambridge and 7-Scenes

We provide the script [datasetup.py](datasetup.py) for setting up Cambridge and 7-Scenes datasets. The script can be
called with either the name of the dataset to setup, *e.g.*, `7-Scenes`, or the name of a specific scene, *e.g.*,
`KingsCollege`. For example, if you want to setup the whole Cambridge dataset:
```shell
python datasetup.py Cambridge
```
Or if you want to only setup the *chess* scene of 7-Scenes dataset:
```shell
python datasetup.py chess
```
All possibilities can be accessed by running:
```shell
python datasetup.py -h
```
