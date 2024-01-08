import json
import joblib
import pandas as pd
import numpy as np


class Model:
    def __init__(self, **configs):
        # load buy and sell models
        buy_model_path = configs['Model']['params']['model_path_buy']
        sell_model_path = configs['Model']['params']['model_path_sell']
        self.buy_model = joblib.load(buy_model_path)
        self.sell_model = joblib.load(sell_model_path)
        # list of patterns for which models will make predictions
        self.patterns_to_predict = configs['Model']['params']['patterns_to_predict']
        self.favorite_exchanges = configs['Telegram']['params']['favorite_exchanges']
        # load feature lists for buy and sell models
        buy_feature_path = configs['Model']['params']['feature_path_buy']
        sell_feature_path = configs['Model']['params']['feature_path_sell']
        with open(buy_feature_path) as f:
            self.buy_feature_dict = json.load(f)
        with open(sell_feature_path) as f:
            self.sell_feature_dict = json.load(f)

    def prepare_data(self, df: pd.DataFrame, signal_points: list, ttype: str) -> pd.DataFrame:
        """ Get data from ticker dataframe and prepare it for model prediction """
        last_time = df['time'].max()
        row, rows = pd.DataFrame(), pd.DataFrame()

        if ttype == 'buy':
            feature_dict = self.buy_feature_dict
        else:
            feature_dict = self.sell_feature_dict

        for key, features in feature_dict.items():
            if key.isdigit():
                try:
                    tmp_row = df.loc[df['time'] == last_time -
                                     pd.to_timedelta(int(key), unit='h'), features].reset_index(drop=True)
                except KeyError:
                    return pd.DataFrame()
                row = pd.concat([row, tmp_row], axis=1)
        row.columns = feature_dict['features']
        # add number of signal point for which prediction is made
        row['sig_point_num'] = 0
        # make predictions only for patterns, which are suitable for ML model prediction
        # and only for exchanges where we trade
        patterns = list()
        for i, point in enumerate(signal_points):
            pattern = point[5]
            exchange_list = point[7]
            if pattern in self.patterns_to_predict and set(exchange_list).intersection(set(self.favorite_exchanges)):
                patterns.append(pattern)
                rows = pd.concat([rows, row])
                rows.iloc[-1, rows.columns.get_loc('sig_point_num')] = i
        rows.reset_index(inplace=True, drop=True)
        return rows

    def make_prediction(self, df: pd.DataFrame, signal_points: list, ttype: str) -> list:
        """ Make prediction with model """
        if ttype == 'buy':
            model = self.buy_model
        else:
            model = self.sell_model
        rows = self.prepare_data(df, signal_points, ttype)
        if rows.shape[0] == 0:
            return signal_points
        preds = np.zeros([len(rows), 1])
        preds[:, 0] = model.predict_proba(rows.iloc[:, :-1])[:, 1]
        # transform predictions to list
        preds = preds.ravel().tolist()
        sig_point_nums = rows['sig_point_num'].tolist()
        # add predictions to signal points
        for s_p_n, pred in zip(sig_point_nums, preds):
            signal_points[s_p_n][9] = pred
        return signal_points
