# Pytorch Classification

Framework to manage, prepare, train and evaluate models

## Structure of the data environment

All data, models, data structures, charts, log/csv files etc. are stored in the folder data in the main directory and have the following structure:

* raw
* prepared
* processed

### Raw

The raw folder contains the (original) data to be learned and validated.

### Prepared

The prepared folder contains the prepared data. Prepared data is structured for learning. The structuring can be, for example:

* ratio val/train data
* unbalanced/balanced data
* grouped classifiers
* binary classifiers

To save storage space, the data in this order are merely linked to the raw data within the raw folder.

### Processed

This folder is used to save the calculated results, models and data. These data and models are then used later for evaluation after the calculation.

## Prepare data

TODO

## Train model

```shell
bin/train \
    --arch [the-model-to-be-used]* \
    --epochs [number-of-epochs-to-be-learned] \
    --learning-rate [learning-rate] \
    --learning-rate-decrease-factor [learning-rate-decrease-factor] \
    --learning-rate-decrease-after [learning-rate-decrease-after] \
    --linear-layer [linear-layer] \
    --pretrained \
    --batch-size [batch-size] \
    --session-name all \
    --csv-path-settings [the-place-where-the-settings-should-be-written] \
    --csv-path-summary [the-place-where-the-most-important-outputs-should-be-written] \
    --csv-path-summary-full [the-place-where-all-outputs-should-be-written] \
    --model-path [the-location-where-the-model-is-to-be-saved] \
    --print-freq 1 \
    [the-location-of-the-data-to-be-trained-and-validated]
```

Possible models are:

* alexnet
* vgg11
* vgg11_bn
* ...

An overview of all usable models can be found here: https://pytorch.org/docs/stable/torchvision/models.html

Parameter recommendations:

* epochs: 21
* learning-rate: 0.001
* learning-rate-decrease-factor: 0.1
* learning-rate-decrease-after: 7
* linear-layer: The number of data to be classified
* batch-size: Depending on the available memory for the GPU, the number of elements to be learned simultaneously (e.g. 8)

## Evaluate model

### Write evaluation csv

```shell
bin/train \
    --evaluate \
    --csv-path-validated auto \
    --resume [path-to-the-model-to-be-validated] \
    [path-to-folder-with-data-to-be-validated]
```

This command generates a CSV file with the result of the model's evaluation.

## Create charts

TODO

## A. Authors

* **Björn Hempel** - *Initial work* - [Björn Hempel](https://github.com/bjoern-hempel)

## B. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## C. Closing words

Have fun! :)
