"""
# Created by Lukas on 05.05.2017#
Topic: StorageLogic
Task: initialize a battery with with
Description:
the battery object allows to simulate a trading process,
while honoring the physical limitations of the EES
simultaneously the objects keeps record of all actions
"""

import datetime
import pandas as pd
import numpy as np

##################################################
#                >>  Objects  <<                 #
##################################################


class battery:
    def __init__(self,
                 self_discharge=0.007,
                 volume=1,
                 efficiency=0.825,
                 storage_speed=0.25,
                 balance=0,
                 storage_level=0,
                 signal_modificator=0,
                 initializer="Annon",
                 max_load_cycles = 7000,
                 investment_cost=700000,
                 ):

        self.investment_cost = investment_cost
        self.max_load_cycles = max_load_cycles
        self.minimal_gain_per_load_cycle = investment_cost/max_load_cycles
        self.efficiency = efficiency
        self.signal_modificator = signal_modificator
        self.balance = balance
        self.storage_level = storage_level
        self.volume = volume
        self.storage_speed = storage_speed
        self.self_discharge = self_discharge
        self.initializer = initializer  # Todo implement

        self.history = {
            "storagelevel": [],
            "delta_storage_level": [],
            "balance": [],
            "performance": [],
            "price": [],
            "signal": [],
            "activity": []}

    def mk_report_dataFrame(self, suffix=False, Path="BAT_History", filename=False, date=False):
        """
        this function transforms the .self.history to a pandas' DataFrame
        :param suffix: append a name specifying the algorithm.
        :param Path: Path to the output file. default "BAT_History"
        :param filename: if true save as BAT_History_{}.csv
        :param date: adds the date to the file name
        :return: df with history
        """

        if not suffix:
            suffix = self.initializer
        return_df = pd.DataFrame(self.history)
        return_df = return_df.add_suffix(("___"+suffix))
        if filename:
            filename = "{}\\BAT_History_{}".format(Path, filename)
            if date:
                filename += "_" + str(datetime.date.today())
            filename += ".csv"
            return_df.to_csv(filename, sep=";")

        return return_df

    def store_energy(self, current_price, signal):
        """
        calculate the storing process
        document the storing process
        :param current_price: input price of bat.behave
        :param signal: signal
        :return:
        """
        storable_energy = min(signal, self.volume - self.storage_level)

        # calculate the performance
        performance = storable_energy/self.efficiency * current_price * -1
        self.balance += performance
        # physical storing
        self.storage_level += storable_energy

        # update history
        self.history["performance"].append(performance)
        self.history["balance"].append(self.balance)
        self.history["delta_storage_level"].append(storable_energy)

        # plausibility check
        if self.storage_level > self.volume:
            print("LogicalError")
            quit()

    def sell_energy(self, current_price, signal):
        """
        Calculate the selling process
        document the selling process
        :param current_price: input price of bat.behave
        :param signal: signal
        :return:
        """
        sellable_energy = min(signal * -1, self.storage_level)
        # Todo maxstorage speed  missing because of GAMS TESTS storable_energy = min(signal * self.max_storage_speed, self.volume - self.storage_level)
        # calculate the performance
        performance = sellable_energy * current_price
        self.balance += performance

        # physical storing
        self.storage_level -= sellable_energy

        # update history
        self.history["performance"].append(performance)
        self.history["balance"].append(self.balance)
        self.history["delta_storage_level"].append(sellable_energy * -1)

        # plausibility check
        if self.storage_level < 0:
            print("Error1")

    def behave(self, signal, current_price):

        self.history["price"].append(current_price)
        self.history["signal"].append(signal)

        if signal > 0 and self.storage_level < self.volume:
            self.store_energy(current_price, signal)
            self.history["activity"].append("storing..")
            # store

        elif signal < 0 and self.storage_level > 0:
            self.sell_energy(current_price, signal)
            self.history["activity"].append("selling..")
            # sell

        elif signal == 0:
            # wait
            self.history["performance"].append(0)
            self.history["balance"].append(self.balance)
            self.history["delta_storage_level"].append(0)
            self.history["activity"].append("waiting..")
        else:
            # storage cap reached
            self.history["performance"].append(0)
            self.history["balance"].append(self.balance)
            self.history["delta_storage_level"].append(0)
            if self.storage_level == 1:
                self.history["activity"].append("I'm full")
                # "will never happen"

            if self.storage_level == 0:
                self.history["activity"].append("I'm empty")

        if (signal < -1) or (signal > 1):
            # Error
            print("Error, signal is ", signal)
            exit()
        self.history["storagelevel"].append(self.storage_level)
        ## self-discharge is in %
        self.storage_level *= (1 - self.self_discharge/100)

##################################################
#               >>  Functions  <<                #
##################################################

def simple_signal(price_series, sell_price=40, buy_price=40, maxStorageSpeed=1):
    """
    This function evaluates the price series and produces simple storage signals
    :param price_series: a times series of prices (pd.Series,list etc)
    :param sell_price: lower threshold for selling energy
    :param buy_price: upperthreshold for buying energy
    :return:list of signals for Storagelogic.py
    """

    signal = []
    for price in price_series:
        if price < buy_price:
            signal.append(maxStorageSpeed)
        elif price > sell_price:
            signal.append(maxStorageSpeed * -1)
        else:
            signal.append(0)
    return signal

def runSimpleLogic(price_series, initializer="SimpleLogic", buy_price=35, sell_price=38):
    """
    Start a run with an price Series
    Compute the signal by itself, depending on given parameters of buy-price and sell_price
    :param initializer: name of the initializer of the Battery
    :param price_series:
    :param buy_price:
    :param sell_price:
    :return: A Bettery, which already performed a cycle run through the priceseries
    """

    simple = simple_signal(price_series=price_series, buy_price=buy_price, sell_price=sell_price)
    bat = battery(initializer=initializer)
    for signal, price in zip(simple, price_series):

        bat.behave(signal, price)
    return bat

def runLogic(initializer, price_series, signals, signal_format):
    """
    Run the batteryLogic with given price_series and signals
    :param initializer: name of the initializer of the Battery
    :param price_series:
    :param signals:
    :param signal_format:to format the input signal to a consistent value
    :return: A Bettery, which already performed a cycle run through the price series
    """
    bat = battery(initializer=initializer)
    signals = np.array(signals)
    price_series = np.array(price_series)
    if signal_format == "0.1":
        for signal, price in zip(signals, price_series):
            bat.behave(signal, price)
    if signal_format == "1":
        price_series[-1] = 0
        assert len(signals) == len(price_series)

        for signal, price in zip(signals, price_series):
            bat.behave(signal*bat.storage_speed, price)
    else:
        raise TypeError("Signaltype is not correct!")
    return bat