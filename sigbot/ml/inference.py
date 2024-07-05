import json
import joblib
import pandas as pd
from indicators import indicators
from log.log import logger


class Model:
    def __init__(self, **configs):
        self.configs = configs
        # load buy and sell models
        model_lgb_path = configs['Model']['params']['model_path']
        self.model_lgb = joblib.load(model_lgb_path)
        # list of patterns for which models will make predictions
        self.patterns_to_predict = configs['Model']['params']['patterns_to_predict']
        self.favorite_exchanges = configs['Telegram']['params']['favorite_exchanges']
        # load feature lists for buy and sell models
        feature_path = configs['Model']['params']['feature_path']
        with open(feature_path) as f:
            self.feature_dict = json.load(f)
        # models relative weights
        self.weight = configs['Model']['params']['weight']
        # time (hour) when model is allowed to predict
        self.time_to_predict_buy = configs['Model']['params']['time_to_predict_buy']
        self.time_to_predict_sell = configs['Model']['params']['time_to_predict_sell']
        self.cols_to_scale = configs['Model']['params']['cols_to_scale']
        self.pred_thresh = configs['Model']['params']['pred_thresh']

    def prepare_data(self, df: pd.DataFrame, btcd: pd.DataFrame, btcdom: pd.DataFrame,
                     signal_points: list, ttype: str, exchange_name: str) -> pd.DataFrame:
        """ Get data from ticker dataframe and prepare it for model prediction """
        btcd_cols = list(btcd.columns)
        btcdom_cols = list(btcdom.columns)
        rows = pd.DataFrame()
        tmp_df = df.copy()
        # add CCI and SAR indicators
        cci = indicators.CCI(ttype, self.configs)
        tmp_df = cci.get_indicator(tmp_df, '', '', 0)
        # add SAR
        sar = indicators.SAR(ttype, self.configs)
        tmp_df = sar.get_indicator(tmp_df, '', '', 0)
        # bring columns with highly different absolute values (for different tickers) to similar scale
        for c in self.cols_to_scale:
            tmp_df[c] = tmp_df[c].pct_change() * 100
        # merge with BTC dominance dataframes
        tmp_df[btcd_cols] = pd.merge(tmp_df[['time']], btcd[btcd_cols], how='left', on='time')
        tmp_df[btcdom_cols] = pd.merge(tmp_df[['time']], btcdom[btcdom_cols], how='left', on='time')
        btcd_btcdom_cols = btcd_cols + btcdom_cols[1:]
        tmp_df[btcd_btcdom_cols] = tmp_df[btcd_btcdom_cols].ffill()
        # create dataframe for prediction
        for i, point in enumerate(signal_points):
            ticker = point[0]
            point_idx = point[2]
            pattern = point[5]
            point_time = tmp_df.iloc[point_idx, tmp_df.columns.get_loc('time')]
            # predict only at selected hours
            if ttype == 'buy' and point_time.hour not in self.time_to_predict_buy:
                logger.info(f'Hour {point_time.hour} is not in list of hours when model can predict for buy trades')
                continue
            if ttype == 'sell' and point_time.hour not in self.time_to_predict_sell:
                logger.info(f'Hour {point_time.hour} is not in list of hours when model can predict for sell trades')
                continue
            # predict only for selected patterns and exchanges
            if pattern not in self.patterns_to_predict or exchange_name not in self.favorite_exchanges:
                logger.info(f'Hour {exchange_name} is not in the list of favorite exchanges '
                            f'or {pattern} not in the list of favorite patterns')
                continue
            row = pd.DataFrame()
            for key, features in self.feature_dict.items():
                if key.isdigit():
                    try:
                        tmp_row = tmp_df.loc[tmp_df['time'] == point_time -
                                             pd.to_timedelta(int(key), unit='h'), features].reset_index(drop=True)
                    except KeyError as key_err:
                        logger.exception(key_err)
                        return pd.DataFrame()
                    row = pd.concat([row, tmp_row], axis=1)
            row['weekday'] = point_time.weekday()
            row['hour'] = point_time.hour
            row.columns = self.feature_dict['features']
            # add number of signal point for which prediction is made
            row['sig_point_num'] = 0
            rows = pd.concat([rows, row])
            rows.iloc[-1, rows.columns.get_loc('sig_point_num')] = i
        rows = rows.reset_index(drop=True).fillna(0)
        return rows

    def make_prediction(self, df: pd.DataFrame, btcd: pd.DataFrame, btcdom: pd.DataFrame,
                        signal_points: list, ttype: str, exchange_name: str) -> list:
        """ Make prediction with model """
        rows = self.prepare_data(df, btcd, btcdom, signal_points, ttype, exchange_name)
        if rows.shape[0] == 0:
            return signal_points
        # make predictions and average them
        preds = self.model_lgb.predict_proba(rows.iloc[:, :-1])
        # transform predictions to a list
        preds = preds[:, 1].ravel().tolist()
        sig_point_nums = rows['sig_point_num'].tolist()
        # add predictions to signal points
        for s_p_n, pred in zip(sig_point_nums, preds):
            logger.info(f'Prediction score for ticker {signal_points[s_p_n][0]} is {pred}')
            if pred > self.pred_thresh:
                signal_points[s_p_n][9] = pred
        return signal_points
