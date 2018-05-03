# Evaluation-of-Machine-Learning-Based-Storage-Control-Algorithms-for-the-EEX-Day-Ahead-Electricity-Ma
corresponding code to my master's thesis.
---

**This is a Draft**


## Content
1. Overview
2. Load the Data from OpenPowerSystemData.org
3. Data Preparation
4. storage logic


## 1. Overview

_not implemented yet_

## 2. Load the Data from Load the Data from OpenPowerSystemData.org
The corresponding script to this chapter is called **OPSD_Import.py**
This script performs X steps:
* opening the local file of market data
* parsing the date stamp
* creating dummy variables for the relevant columns (e.g. time related columns)
* calculating values of interest (residual load, total renewable generation etc...)
* dropping unwanted columns
* creating an overview file containing the remaining headers
* saving the new _clean_ data frame as excel and as .pickle for further actions

it is necessary to have an input file that is formatted like:
> time_series_60min_singleindex_filtered.csv

see [OPSD](https://open-power-system-data.org/)'documentation on how to
obtain a similar file suited for your needs
... or edit the code if you're brave enough :wink:
However the input file should hold sufficient data

## Data Preparation
The corresponding script to this chapter is called **data_preparation.py**
the idea behind this "additional" was shorten the preparation process.
This allows to test different techniques and parameters during without
cleaning the raw input file every time.

Additionally, the result from the linear programming model (LPM) get
processed and are included after this step.

Finally, this script's _get\_dict()_ function  returns a handy dictionary,
that contains all relevant data frames
(e.g. each features & labels for testing, training & evaluation)
as well as the performance of the LPM

The script relies on three inputs:
* the pickle from OPSD_import.py (00_OPSD_DATA.pickle)
holding the _clean_ market information

* the output from the LPM(results_GAMS_10-15_TOTAL_single_wo_INFO_.csv)
holding the storage actions for the optimal storage behavior

* the script Storagelocgic
this script simulates a storage behaviour. It's used to calculate the
profit of the LPM

## 4. Storage Logic
The storage logic performs the virtual storage process and assures that
all physical restrictions are met.
Additionally, it saves the history of a time series of storage signals

## 5.Linear Optimization Model(LPM)
The LPM was realized in GAMS.
It financially optimizes the storage behavior of an given battery system.
The output of this model is for one part the reference profit a battery can yield.
on the other hand it provides the input data for the machine learning algorithms.

## 6. Hyper-Parameter Tuning
These scripts perform the respective hyper-parameter tuning for the classifiers.
The hyper-parameters are used to train the classifiers.

## 7. Evaluation
The evaluation provides performs the training and evaluation of the machine learning algorithms
and provides summary files as well as confusion matrices for the evaluation process

## 8. Util Scripts
_not yet implemented_


