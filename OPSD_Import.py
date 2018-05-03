"""
#################
## OPSD IMPORT ##
#################

# This Script manages the data import from Open-Power-System-Data (OPSD)
# the source file is downloaded manually and from the OPSD homepage

# The Tasks of the script are:
#   - load the data from a .csv file into a "pandas" DataFrame
#   - transform the time stamp string to a time object
#   - delete unnecessary columns
#   - delete columns with to many empty values

#   - calculate some values of interest
        - residual load
        - combined forecasts
        - combined renewable generations
#   - save the DataFrame to a .xls -> OPSD_OWN_table.xls
#   - save a pickle for the next run :-)

"""

from datetime import datetime
import pandas as pd
import warnings


# ---------
# Functions
# ---------


def transform_date(time_stamp_str):
    """
    ----------
    # convert the time stamp string from the file to a machine readable object
    ----------
    """
    return datetime.strptime(time_stamp_str, "%Y-%m-%dT%H:%M:%SZ")


def format_workday(value):
    """
    Classify a variable as either workday or weekend-day
    :param value: number of day
    :return: 1 : workday,
            0 : weekend-day
    """
    if value <= 5:
        return 1
    else:

        return 0


def init_time_stamps(df):
    """
    format the timestamp and make variables for every component of the timestamp
    """
    df["absolute_hour"] = df.index
    df["Time"] = list(map(transform_date, df["utc_timestamp"]))
    df["hour"] = pd.DatetimeIndex(df["Time"]).hour
    df["day"] = pd.DatetimeIndex(df["Time"]).day
    df["month"] = pd.DatetimeIndex(df["Time"]).month
    df["year"] = pd.DatetimeIndex(df["Time"]).year
    df["weekday"] = pd.DatetimeIndex(df["Time"]).weekday
    df["workday"] = list(map(format_workday, df["weekday"]))

    return df


def create_header_overview(df2header, filename="00_Headers.txt"):
    """
    create a file with all headers of the DataFrame for a better overview
    """

    with open(filename, "w") as f:
        for header in list(df2header):
            f.write("{}\n".format(header))


if __name__ == "__main__":

    # file import
    original_file = "time_series_60min_singleindex_filtered.csv"

    # load Dataframe
    df = pd.read_csv(original_file, index_col=1, parse_dates=True)
    da = init_time_stamps(df)

    # drop values with lots of NAN
    df.drop('interpolated_values', axis=1, inplace=True)

    # off and onshore generation are summarized in in wind generation
    df.drop("DE_wind_offshore_generation", axis=1, inplace=True)
    df.drop("DE_wind_onshore_generation", axis=1, inplace=True)
    df.drop("DE_50hertz_wind_offshore_forecast", axis=1, inplace=True)
    df.drop("DE_50hertz_wind_onshore_forecast", axis=1, inplace=True)
    df.drop("DE_50hertz_wind_offshore_generation", axis=1, inplace=True)
    df.drop("DE_50hertz_wind_onshore_generation", axis=1, inplace=True)

    # calculate the residual load
    #   fill "nan" with zeros
    #   subtract the solar and wind generation from the total load
    df["DE_solar_generation"] = df["DE_solar_generation"].fillna(value=0)

    # calculate the total renewable generation
    df["renewable_generation"] = df["DE_wind_generation"] + df["DE_solar_generation"]
    df["DE_residual_load"] = df["DE_load_"] - df["renewable_generation"]

    # calculate the total wind forecast
    df["forecast_wind"] = df["DE_50hertz_wind_forecast"] + \
                          df["DE_amprion_wind_forecast"] + \
                          df["DE_tennet_wind_forecast"] + \
                          df["DE_transnetbw_wind_forecast"]

    # calculate the total solar forecast
    df["forecast_solar"] = df["DE_50hertz_solar_forecast"] + \
                           df["DE_amprion_solar_forecast"] + \
                           df["DE_tennet_solar_forecast"] + \
                           df["DE_transnetbw_solar_forecast"]

    # calculate the total forecast
    df["forecast_total"] = df["forecast_wind"] + df["forecast_wind"]

    # a list of the individual TSO's forecast and generation reports to be dropped...
    drop_list = ["DE_50hertz_wind_forecast",
                 "DE_amprion_wind_forecast",
                 "DE_tennet_wind_forecast",
                 "DE_transnetbw_wind_forecast",
                 "DE_50hertz_solar_forecast",
                 "DE_amprion_solar_forecast",
                 "DE_tennet_solar_forecast",
                 "DE_transnetbw_solar_forecast",
                 'DE_50hertz_solar_generation',
                 'DE_50hertz_wind_generation',
                 'DE_amprion_solar_generation',
                 'DE_amprion_wind_generation',
                 'DE_amprion_wind_onshore_generation',
                 'DE_tennet_solar_generation',
                 'DE_tennet_wind_generation',
                 'DE_tennet_wind_offshore_generation',
                 'DE_tennet_wind_onshore_generation',
                 'DE_transnetbw_solar_generation',
                 'DE_transnetbw_wind_generation',
                 'DE_transnetbw_wind_onshore_generation']

    # ...is used to drop these columns
    for element in drop_list:
        df.drop(element, axis=1, inplace=True)

    # generate dummy variables
    dummy_list = ['hour',
                  'month',
                  'weekday']
    for element in dummy_list:
        dummy_table = pd.get_dummies(df[element], prefix=element, prefix_sep="_")
        df = pd.concat([df, dummy_table], axis=1)

    # save the resulting df
    df.to_pickle("00_OPSD_DATA.pickle")
    df.to_excel("00_OPSD_DATA.xls")
    create_header_overview(df, "Info\\Headers_downsized.txt")
