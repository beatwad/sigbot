"""
This module provides a functionality for collecting and
processing data for the model and making a prediction with
this model
"""

import json

import joblib
import pandas as pd

from indicators import indicators
from loguru import logger


class Model:
    """
    A class to represent a trading model.

    Attributes
    ----------
    configs : dict
        Configuration settings for the model.
    model_lgb : object
        Loaded LightGBM model for predictions.
    patterns_to_predict : list
        List of signal patterns for which models will make predictions.
    favorite_exchanges : list
        List of favorite exchanges to filter predictions.
    feature_dict : dict
        Dictionary of features which are used for model prediction.
    time_to_predict_buy : list
        List of hours when the model is allowed to make buy predictions.
    time_to_predict_sell : list
        List of hours when the model is allowed to make sell predictions.
    cols_to_scale : list
        List of numerical columns to be scaled.
    pred_thresh : float
        Prediction threshold for making a trade decision.
    """

    def __init__(self, **configs):
        self.configs = configs
        # load buy and sell models
        model_lgb_path = configs["Model"]["params"]["model_path"]
        self.model_lgb = joblib.load(model_lgb_path)
        # list of patterns for which models will make predictions
        self.patterns_to_predict = configs["Model"]["params"]["patterns_to_predict"]
        self.favorite_exchanges = configs["Telegram"]["params"]["favorite_exchanges"]
        # load feature lists for buy and sell models
        feature_path = configs["Model"]["params"]["feature_path"]
        with open(feature_path) as f:
            self.feature_dict = json.load(f)
        # time (hour) when model is allowed to predict
        self.time_to_predict_buy = configs["Model"]["params"]["time_to_predict_buy"]
        self.time_to_predict_sell = configs["Model"]["params"]["time_to_predict_sell"]
        self.cols_to_scale = configs["Model"]["params"]["cols_to_scale"]
        self.pred_thresh = configs["Model"]["params"]["pred_thresh"]

    def prepare_data(
        self,
        df: pd.DataFrame,
        btcd: pd.DataFrame,
        btcdom: pd.DataFrame,
        signal_points: list,
        ttype: str,
        exchange_name: str,
    ) -> pd.DataFrame:
        """
        Get data from ticker DataFrame and prepare it for model prediction.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ticker data.
        btcd : pd.DataFrame
            DataFrame containing BTC dominance data.
        btcdom : pd.DataFrame
            DataFrame containing BTC dominance data for the market.
        signal_points : list
            List of signal points to analyze.
        ttype : str
            Type of trade, either 'buy' or 'sell'.
        exchange_name : str
            Name of the exchange to filter predictions by exchange.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing prepared data for model prediction.
        """
        btcd_cols = list(btcd.columns)
        btcdom_cols = list(btcdom.columns)
        rows = []
        tmp_df = df.copy()
        # add CCI and SAR indicators
        cci = indicators.CCI(ttype, self.configs)
        tmp_df = cci.get_indicator(tmp_df, "", "", 0)
        # add SAR
        sar = indicators.SAR(ttype, self.configs)
        tmp_df = sar.get_indicator(tmp_df, "", "", 0)
        # bring columns with highly different absolute values
        # (for different tickers) to similar scale
        for c in self.cols_to_scale:
            tmp_df[c] = tmp_df[c].pct_change() * 100
        # merge with BTC dominance dataframes
        tmp_df[btcd_cols] = pd.merge(tmp_df[["time"]], btcd[btcd_cols], how="left", on="time")
        tmp_df[btcdom_cols] = pd.merge(tmp_df[["time"]], btcdom[btcdom_cols], how="left", on="time")
        btcd_btcdom_cols = btcd_cols + btcdom_cols[1:]
        tmp_df[btcd_btcdom_cols] = tmp_df[btcd_btcdom_cols].ffill()
        tmp_df["weekday"] = 0
        tmp_df["hour"] = 0
        # create dataframe for prediction
        for i, point in enumerate(signal_points):
            ticker = point[0]
            point_idx = point[2]
            pattern = point[5]
            point_time = tmp_df.iloc[point_idx, tmp_df.columns.get_loc("time")]
            # predict only at selected hours
            if ttype == "buy" and point_time.hour not in self.time_to_predict_buy:
                continue
            if ttype == "sell" and point_time.hour not in self.time_to_predict_sell:
                continue
            # predict only for selected patterns and exchanges
            if (
                pattern not in self.patterns_to_predict
                or exchange_name not in self.favorite_exchanges
            ):
                continue
            # add weekday and hour features
            tmp_df.loc[tmp_df["time"] == point_time, "weekday"] = point_time.weekday()
            tmp_df.loc[tmp_df["time"] == point_time, "hour"] = point_time.hour
            row = pd.DataFrame()
            for key, features in self.feature_dict.items():
                if key.isdigit():
                    try:
                        tmp_row = tmp_df.loc[
                            tmp_df["time"] == point_time - pd.to_timedelta(int(key), unit="h"),
                            features,
                        ].reset_index(drop=True)
                    except KeyError as key_err:
                        logger.exception(key_err)
                        return pd.DataFrame()
                    row = pd.concat([row, tmp_row], axis=1)
            # if row contains NaNs - skip it
            if row.isnull().sum().sum() > 0:
                logger.info(
                    f"Ticker {ticker} signal with time " f"{point_time} contains NaNs, skip it"
                )
                continue
            row.columns = self.feature_dict["features"]
            # add number of signal point for which prediction is made
            row["sig_point_num"] = i
            rows.append(row)

        if len(rows) > 0:
            rows = pd.concat(rows).reset_index(drop=True)
        else:
            rows = pd.DataFrame()
        return rows

    def make_prediction(
        self,
        df: pd.DataFrame,
        btcd: pd.DataFrame,
        btcdom: pd.DataFrame,
        signal_points: list,
        ttype: str,
        exchange_name: str,
    ) -> list:
        """
        Make prediction using the model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ticker data.
        btcd : pd.DataFrame
            DataFrame containing BTC dominance data.
        btcdom : pd.DataFrame
            DataFrame containing BTC dominance data for the market.
        signal_points : list
            List of signal points to analyze.
        ttype : str
            Type of trade, either 'buy' or 'sell'.
        exchange_name : str
            Name of the exchange to filter predictions by exchange.

        Returns
        -------
        list
            Updated list of signal points with model prediction scores.
        """
        rows = self.prepare_data(df, btcd, btcdom, signal_points, ttype, exchange_name)
        if rows.shape[0] == 0:
            return signal_points
        # make predictions and average them
        preds = self.model_lgb.predict_proba(rows.iloc[:, :-1])
        # transform predictions to a list
        preds = preds[:, 1].ravel().tolist()
        sig_point_nums = rows["sig_point_num"].tolist()
        # add predictions to signal points
        for s_p_n, pred in zip(sig_point_nums, preds):
            logger.info(f"Prediction score for ticker {signal_points[s_p_n][0]} is {pred}")
            # if pred > self.pred_thresh:
            signal_points[s_p_n][9] = pred
        return signal_points
