import json
import joblib
import pandas as pd
import numpy as np


class Model:
    def __init__(self, **configs):
        model_path = configs['Model']['params']['model_path']
        self.model = joblib.load(model_path)
        feature_path = configs['Model']['params']['feature_path']
        with open(feature_path) as f:
            self.feature_dict = json.load(f)
        # threshold that filters model predictions with low confidence
        self.high_bound = configs['Model']['params']['high_bound']

    def prepare_data(self, df: pd.DataFrame, signal_points: list) -> pd.DataFrame:
        """ Get data from ticker dataframe and prepare it for model prediction """
        last_time = df['time'].max()
        row, rows = pd.DataFrame(), pd.DataFrame()

        for key, features in self.feature_dict.items():
            if key.isdigit():
                tmp_row = df.loc[df['time'] == last_time - pd.to_timedelta(int(key), unit='h'), features].reset_index(drop=True)
                row = pd.concat([row, tmp_row], axis=1)
        row['Pattern_Trend'] = 0
        row['STOCH_RSI'] = 0
        row.columns = self.feature_dict['features']
        # make predictions only for patterns, which are suitable for ML model prediction
        patterns = list()
        for point in signal_points:
            pattern = point[5]
            if pattern in ['Pattern_Trend', 'STOCH_RSI', 'MACD']:
                patterns.append(pattern)
                rows = pd.concat([rows, row])
        rows.reset_index(inplace=True, drop=True)
        # for every pattern in a signal points list add its own row and mark corresponding pattern feature with 1
        for i, pattern in enumerate(patterns):
            if pattern == 'Pattern_Trend' or pattern == 'STOCH_RSI':
                rows.iloc[i, rows.columns.get_loc(pattern)] = 1
        return rows

    def make_prediction(self, df: pd.DataFrame, signal_points: list) -> list:
        """ Make prediction with model """
        rows = self.prepare_data(df, signal_points)
        preds = np.zeros([len(rows), 1])
        preds[:, 0] = self.model.predict_proba(rows)[:, 1]
        # select only highly confident prediction
        preds = np.where(preds >= self.high_bound, preds, 0).squeeze()
        return preds.tolist()
