# Pytorch Classification

Framework to manage, prepare, train and evaluate models

## Structure of the data environment

All data, models, data structures, charts, log/csv files etc. are stored in the folder data within the main directory and have the following structure:

* raw
* prepared
* processed

This folder and its subfolders must be created independently and are **not** part of this project.

### Raw

The raw folder contains the (original) data to be learned and validated. The structure of the folder is as follows:

`/data/raw/[superclass]/[class]/[data-file]`

* [superclass]: Superclass contains all classes that should be learned
* [class]: The classes to be learned
* [data-file]: The files to be learned

### Prepared

The prepared folder contains the prepared data. Prepared data is structured for learning, whereas raw data is not. The structuring can be, for example:

* ratio val/train data (e.g. 80% training / 20% validation data)
* unbalanced/balanced data (the number of data within the classes is balanced or is not)
* grouped classifiers (each group gets its own model)
* binary classifiers (each class gets its own model - true/false model)

To save storage space, the data in this order are merely linked to the raw data within the raw folder.

### Processed

This folder is used to save the calculated results, models and data. These data and models are then used later for evaluation after the calculation.

## Prepare data

### Manage prepare folder

#### Build ratio folders

TODO

#### Build binary data folders

```shell
user$ src/prepare-binary-data.py
```

### Prepare working environment (processed folder)

TODO

## Train model

```shell
user$ bin/train \
    --arch [the-model-to-be-used]* \
    --epochs [number-of-epochs-to-be-learned] \
    --learning-rate [learning-rate] \
    --learning-rate-decrease-factor [learning-rate-decrease-factor] \
    --learning-rate-decrease-after [learning-rate-decrease-after] \
    --linear-layer [linear-layer] \
    --pretrained \
    --batch-size [batch-size] \
    --session-name [the-label-to-be-used-for-validations] \
    --csv-path-settings [the-place-where-the-settings-should-be-written] \
    --csv-path-summary [the-place-where-the-most-important-outputs-should-be-written] \
    --csv-path-summary-full [the-place-where-all-outputs-should-be-written] \
    --model-path [the-location-where-the-model-is-to-be-saved] \
    --print-freq 1 \
    [the-location-of-the-data-to-be-trained-and-validated]
```

Possible models are:

* alexnet
* densenet121, densenet161, densenet169, densenet201
* inception_v3
* resnet101, resnet152
* resnet18, resnet34, resnet50
* squeezenet1_0, squeezenet1_1
* vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
* ...

An overview of all usable models can be found here: https://pytorch.org/docs/stable/torchvision/models.html

Parameter recommendations:

* epochs: 21
* learning-rate: 0.001
* learning-rate-decrease-factor: 0.1
* learning-rate-decrease-after: 7
* linear-layer: The number of data to be classified
* batch-size: Depending on the available memory for the GPU, the number of elements to be learned simultaneously (e.g. 8)

## Overview of already trained models

```shell
user$ src/evaluate-processed-data.py data/processed/food/unbalanced/90_10/elements/all -om pragmatic

+---------------+--------+-------+-------+----+----+----+----+----------+------------+
| model         | acc 1  | mc    | label | ep | te | be | bs | duration | size       |
+---------------+--------+-------+-------+----+----+----+----+----------+------------+
|   densenet201 | 86.27% | food  | all   | 90 | 90 | 86 |  8 | 10:00:10 |  140.01 MB |
|     resnet152 | 82.64% | food  | all   | 21 | 21 | 17 |  4 | 03:50:24 |  445.19 MB |
|   densenet201 | 82.17% | food  | all   | 21 | 21 | 21 |  4 | 03:22:56 |  140.01 MB |
|         vgg19 | 80.75% | food  | all   | 21 | 21 | 21 |  8 | 02:43:09 | 1066.41 MB |
|      resnet18 | 80.48% | food  | all   | 21 | 21 | 15 |  8 | 00:31:56 |   85.53 MB |
+---------------+--------+-------+-------+----+----+----+----+----------+------------+
|      resnet18 | 79.88% | food  | all   | 21 | 21 | 18 | 16 | 00:24:58 |   85.53 MB |
|         vgg16 | 79.41% | food  | all   | 21 | 21 | 15 |  8 | 02:27:01 | 1025.90 MB |
|      resnet18 | 78.13% | food  | all   | 21 | 21 | 17 | 32 | 00:21:05 |   85.53 MB |
| squeezenet1_0 | 71.40% | food  | all   | 21 | 21 | 19 |  8 | 00:21:16 |    5.83 MB |
| squeezenet1_1 | 70.86% | food  | all   | 21 | 21 | 20 |  4 | 00:28:39 |    5.73 MB |
+---------------+--------+-------+-------+----+----+----+----+----------+------------+
```

## Evaluate model

### Write evaluation csv

```shell
user$ bin/train \
    --evaluate \
    --csv-path-validated auto \
    --resume [path-to-the-model-to-be-validated] \
    [path-to-folder-with-data-to-be-validated]
```

This command generates a CSV file with the result of the model's evaluation.

## Create charts

### Learning success charts

TODO

### Confusion matrix charts

```shell
user$ src/build-confusion-matrix.py
```

### MDS Charts

```shell
user$ src/build-mds-chart.py
```

## Full example to use

TODO

## A. Authors

* **Björn Hempel** - *Initial work* - [Björn Hempel](https://github.com/bjoern-hempel)

## B. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## C. Closing words

Have fun! :)
