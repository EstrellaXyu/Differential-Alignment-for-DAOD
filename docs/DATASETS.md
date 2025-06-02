# Benchmark datasets setup

You can download needed datasets in our paper following our guide.
## Cityscapes &rarr; Foggy Cityscapes

**Images:** Downloading Cityscapes and Foggy Cityscapes on the [Cityscapes website](https://www.cityscapes-dataset.com/), corresponding to the leftImg8bit_trainvaltest.zip and leftImg8bit_trainvaltest_foggy.zip.

**Labels:** For the bounding boxes annotations, we provide JSON annotations files for preproducibility.

[Cityscapes train labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/cityscapes_train_instances.json)

[Cityscapes val labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/cityscapes_val_instances.json)

[Foggy Cityscapes train labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/cityscapes_train_instances_foggyALL.json)

[Foggy Cityscapes val labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/cityscapes_val_instances_foggyALL.json)

## Cityscapes &rarr; BDD100K

**Cityscapes images**: Correspoinding aforementioned the leftImg8bit dictionary.

**BDD100K images**: Download the BDD100K-daytime images [here](https://dl.cv.ethz.ch/bdd100k/data/)

**Labels**: We provide the annotation files for BDD100K-daytime here:

[BDD100K-daytime train labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/bdd100k_daytime_train_cocostyle.json)

[BDD100K-daytime val labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/bdd100k_daytime_val_cocostyle.json)

## Sim10k &rarr; Cityscapes

**Sim10k images:** Download the Sim10k images [here](https://deepblue.lib.umich.edu/data/downloads/ks65hc58r).

**Cityscapes images:** Correspoinding aforementioned the leftImg8bit dictionary.

**Labels:** We provide labels postprocessed for this task here:

[Sim10k cars train labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/sim10k_car_annotations.json)

[Cityscapes cars train labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/cityscapes_train_instances_cars.json)

[Cityscapes cars val labels](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/releases/download/v1.0/cityscapes_val_instances_cars.json)

And all datasets are expected to be organized in the following structure:

```bash
datasets/
    cityscapes/
        leftImg8bit/
        leftImg8bit_foggy/
        annotations/
            cityscapes_train_instances.json
            ...
    sim10k/
        images/
        coco_car_annotations.json
    bdd100k/
        images/
        annotations/
            bdd100k_daytime_train_cocostyle.json
            ...
```

After organizing the dataset, you can preceed to configure the corresponding dictionary paths in the [dataset.py](../DA2OD/datasets.py) like:

```bash
# Cityscapes 
register_coco_instances("cityscapes_train", {},         "/PATH/TO/DATASETS/cityscapes/annotations/cityscapes_train_instances.json",                  "/PATH/TO/DATASETS/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "/PATH/TO/DATASETS/cityscapes/annotations/cityscapes_val_instances.json",                    "/PATH/TO/DATASETS/cityscapes/leftImg8bit/val/")
```