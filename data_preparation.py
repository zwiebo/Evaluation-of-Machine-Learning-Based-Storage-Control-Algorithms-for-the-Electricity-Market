"""
# Created by Lukas on 10.12.2017#
Topic:

Task: Preparing of the input DataFrame
        - adding Lag variables
        - Labeling the dada (linear programming results)
        - Train/Test/Evaluation - Split
"""
import pandas as pd
from StorageLogic import runLogic


def get_dict():
    """
    This function performs data preparation
    :return:dict with all necessary values for the training of classifiers
    """
    def make_label(val, threshold=0.1):
        # label the results from the LP (Classification)
        if val > threshold:
            return 1
        else:
            return 0

    # load the processed OPSD Dataframe
    dataFrame = pd.read_pickle("00_OPSD_DATA.pickle")

    # a list of elemts with non numeric values
    # including the absolute day variable
    del_list = ["absolute_hour",
                'utc_timestamp',
                'Time',
                'day']

    for element in del_list:
            dataFrame.drop(element, axis=1, inplace=True)

    # add lag variables for load and day ahead price
    for shift in range(1, 25):
        dataFrame["DE_price_day_ahead_T-{}".format(shift)] = dataFrame["DE_price_day_ahead"].shift(shift)
        dataFrame["DE_load__T-{}".format(shift)] = dataFrame["DE_load_"].shift(shift)
        # shift for value of last same time last week
    dataFrame["DE_price_day_ahead_T-{}".format("last week")] = dataFrame["DE_price_day_ahead"].shift(168)
    dataFrame["DE_load__T-{}".format("last_week")] = dataFrame["DE_load_"].shift(168)


    # make moving average
    rolling_means = [4, 24, 168]
    for RM in rolling_means:
        dataFrame["DE_price_day_ahead_RM-{}".format(RM)] = \
            dataFrame["DE_price_day_ahead"].\
                rolling(window=RM, min_periods=1).mean()

        dataFrame["DE_load__RM-{}".format(RM)] = \
            dataFrame["DE_load_"]. \
                rolling(window=RM, min_periods=1).mean()

        dataFrame["DE_residual_load__RM-{}".format(RM)] = \
            dataFrame["DE_residual_load"]. \
                rolling(window=RM, min_periods=1).mean()

    # set time frame
    dataFrame = dataFrame[dataFrame.year >= 2010]
    dataFrame = dataFrame[dataFrame.year <= 2015]

    # shift the forecast columns, so that the are in the correct row
    dataFrame["forecast_wind"] = dataFrame["forecast_wind"].shift(-1)
    dataFrame["forecast_solar"] = dataFrame["forecast_solar"].shift(-1)
    dataFrame["forecast_total"] = dataFrame["forecast_total"].shift(-1)

    # load the results file from the LP and shift it
    GAMS_df = pd.read_csv("GAMS\\results_GAMS_10-15_TOTAL_single_wo_INFO_.csv", sep=";")
    GAMS_df = GAMS_df[["Feed_in", "Feed_out"]]
    GAMS_df = GAMS_df.shift(-1)

    # combine features and labels
    dataFrame = pd.DataFrame.join(dataFrame.reset_index(), GAMS_df)
    dataFrame.drop('cet_cest_timestamp', axis=1, inplace=True)

    # transform the label into categorical data
    dataFrame["Feed_in"] = list(map(make_label, dataFrame.loc[:, "Feed_in"]))
    dataFrame["Feed_out"] = list(map(make_label, dataFrame.loc[:, "Feed_out"]))
    # correct setting of labels
    dataFrame.loc[:, "behave"] = dataFrame.Feed_in - dataFrame.Feed_out
    dataFrame.drop("Feed_in", axis=1, inplace=True)
    dataFrame.drop("Feed_out", axis=1, inplace=True)

    # replace "nan"  values with last valid value
    # replace remaining "nan" with 0
    dataFrame.fillna(method="ffill", inplace=True)
    dataFrame.fillna(0, inplace=True)

    # slice the data frames into train/test and evaluation sets
    df_10_14 = dataFrame[dataFrame["year"] <= 2014].copy()
    df_14 = dataFrame[dataFrame["year"] == 2014].copy()
    df_15 = dataFrame[dataFrame["year"] == 2015].copy()

    # split features and labels for...
    # ...optimization (2010- 2014)
    X = df_10_14.drop("behave", axis=1)
    y = df_10_14["behave"]

    # ...optimization (2014)
    X_14 = df_14.drop("behave", axis=1)
    y_14 = df_14["behave"]

    # ..evaluation
    X_eval = df_15.drop("behave", axis=1)
    y_eval = df_15["behave"]

    # shift the prices, to match the features
    prices_eval_10_14 = df_10_14["DE_price_day_ahead"].shift(-1)
    prices_eval_15 = df_15["DE_price_day_ahead"].shift(-1)
    prices_eval_14 = df_14["DE_price_day_ahead"].shift(-1)

    # calculate the the profit of the LPM according the the storage logic
    GAMS_result_10_14 = runLogic("GAMS",
                                 price_series=prices_eval_10_14,
                                 signals=y,
                                 signal_format="1").balance
    GAMS_result_14 = runLogic("GAMS",
                                 price_series=prices_eval_14,
                                 signals=y_14,
                                 signal_format="1").balance
    GAMS_result_15 = runLogic("GAMS",
                              price_series=prices_eval_15,
                              signals=y_eval,
                              signal_format="1").balance

    # populate a dict to return
    return_dict = {"df_10_14": df_10_14,
                   "df_15": df_15,
                   "X": X,
                   "y": y,
                   "X_14": X_14,
                   "y_14": y_14,
                   "X_eval": X_eval,
                   "y_eval": y_eval,
                   "prices_eval_10_14": prices_eval_10_14,
                   "prices_eval_14": prices_eval_14,
                   "prices_eval_15": prices_eval_15,
                   "GAMS_result_10_14":GAMS_result_10_14,
                   "GAMS_result_14":GAMS_result_14,
                   "GAMS_result_15":GAMS_result_15
                   }
    return return_dict

if __name__ == "__main__":
    get_dict()