import json
import joblib
import pandas as pd
import numpy as np


class Model:
    def __init__(self, **configs):
        # load buy and sell models
        model_lgb_path = configs['Model']['params']['model_lgb_path']
        model_svc_path = configs['Model']['params']['model_svc_path']
        self.model_lgb = joblib.load(model_lgb_path)
        self.model_svc = joblib.load(model_svc_path)
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

    def prepare_data(self, df: pd.DataFrame, signal_points: list, ttype: str) -> pd.DataFrame:
        """ Get data from ticker dataframe and prepare it for model prediction """
        rows = pd.DataFrame()
        # bring columns with highly different absolute values (for different tickers) to similar scale
        tmp_df = df[self.cols_to_scale].copy()
        for c in self.cols_to_scale:
            df[c] = df[c].pct_change() * 100
        # create dataframe for prediction
        for i, point in enumerate(signal_points):
            row = pd.DataFrame()
            point_idx = point[2]
            point_time = df.iloc[point_idx, df.columns.get_loc('time')]
            for key, features in self.feature_dict.items():
                if key.isdigit():
                    try:
                        tmp_row = df.loc[df['time'] == point_time -
                                         pd.to_timedelta(int(key), unit='h'), features].reset_index(drop=True)
                    except KeyError:
                        return pd.DataFrame()
                    row = pd.concat([row, tmp_row], axis=1)
            row['weekday'] = point_time.weekday()
            row.columns = self.feature_dict['features'] + ['weekday']
            # add number of signal point for which prediction is made
            row['sig_point_num'] = 0
            # do not predict at "bad" hours
            if ttype == 'buy' and point_time.hour not in self.time_to_predict_buy:
                continue
            if ttype == 'sell' and point_time.hour not in self.time_to_predict_sell:
                continue
            # predict only for favorite exchanges
            pattern = point[5]
            exchange_list = point[7]
            # if pattern in self.patterns_to_predict and set(exchange_list).intersection(set(self.favorite_exchanges)): !!!
            if pattern in self.patterns_to_predict:
                rows = pd.concat([rows, row])
                rows.iloc[-1, rows.columns.get_loc('sig_point_num')] = i
        rows = rows.reset_index(drop=True).fillna(0)
        df[self.cols_to_scale] = tmp_df[self.cols_to_scale]
        return rows

    def make_prediction(self, df: pd.DataFrame, signal_points: list, ttype: str) -> list:
        """ Make prediction with model """
        rows = self.prepare_data(df, signal_points, ttype)
        if rows.shape[0] == 0:
            return signal_points
        # make predictions and average them
        preds_svc = self.model_svc.predict_proba(rows.iloc[:, :-1])
        preds_lgb = self.model_lgb.predict_proba(rows.iloc[:, :-1])
        preds = np.average([preds_svc, preds_lgb], weights=[self.weight, 1 - self.weight], axis=0)
        # transform predictions to a list
        preds = preds[:, 1].ravel().tolist()
        sig_point_nums = rows['sig_point_num'].tolist()
        # add predictions to signal points
        for s_p_n, pred in zip(sig_point_nums, preds):
            signal_points[s_p_n][9] = pred
        return signal_points
