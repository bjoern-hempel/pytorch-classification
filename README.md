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

TODO

## Evaluate model

TODO

## Create charts

TODO

## A. Authors

* **Björn Hempel** - *Initial work* - [Björn Hempel](https://github.com/bjoern-hempel)

## B. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## C. Closing words

Have fun! :)
