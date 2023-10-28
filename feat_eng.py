import pandas as pd
import numpy as np
import gc


def make_features(df):
    # parse the timestamp and create an "hour" feature
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df["hour"] = df["timestamp"].dt.hour
        
    
    # Rolling window algo
    result_dfs = []
    for _, group in df.groupby('series_id'):
        processed_group = rolling_window_algo(group)
        result_dfs.append(processed_group)

    algo_df = pd.concat(result_dfs)
    df = algo_df.dropna(axis = 0, subset=['hour'])
    del algo_df, result_dfs;gc.collect()
    

    periods = 10
    df["anglez_diff"] = df.groupby('series_id')['anglez'].diff(periods).abs().fillna(method="bfill").astype('float16')
    df["anglez"] = abs(df["anglez"])
    df["enmo_diff"] = df.groupby('series_id')['enmo'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["anglez_rolling_mean"] = df["anglez"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_mean"] = df["enmo"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_rolling_max"] = df["anglez"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_max"] = df["enmo"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_rolling_std"] = df["anglez"].rolling(periods,center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_std"] = df["enmo"].rolling(periods,center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling_mean"] = df["anglez_diff"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling_mean"] = df["enmo_diff"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling_max"] = df["anglez_diff"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling_max"] = df["enmo_diff"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    
    return df

features = ["hour",
            "anglez",
            "anglez_rolling_mean",
            "anglez_rolling_max",
            "anglez_rolling_std",
            "anglez_diff",
            "anglez_diff_rolling_mean",
            "anglez_diff_rolling_max",
            "enmo",
            "enmo_rolling_mean",
            "enmo_rolling_max",
            "enmo_rolling_std",
            "enmo_diff",
            "enmo_diff_rolling_mean",
            "enmo_diff_rolling_max",
            "rolling_algo_awake",
           ]

def rolling_window_algo(df_group):
    
    # Steps 3-5
    df_group["anglez_diff"] = df_group['anglez'].diff().abs().fillna(method="bfill").astype('float16')
    df_group['rolling_median'] = df_group['anglez_diff'].rolling(window=60).median()
    # Steps 6-7
    df_group['day_id'] = (df_group['timestamp'] - pd.Timedelta(hours=12)).dt.date
    thresholds = df_group.groupby('day_id')['rolling_median'].quantile(0.1) * 15
    df_group['threshold'] = df_group['day_id'].map(thresholds)
    below_threshold_series = (df_group['rolling_median'] < df_group['threshold']).astype('int')
    df_group['below_threshold'] = below_threshold_series
    block_diff_series = below_threshold_series.diff()
    df_group['block_diff'] = block_diff_series
    df_group['block_start'] = (block_diff_series == 1).astype('int')
    df_group['block_end'] = (block_diff_series == -1).astype('int')
    if below_threshold_series.iloc[0] == 1:
        df_group.at[0, 'block_start'] = 1
    if below_threshold_series.iloc[-1] == 1:
        df_group.at[-1, 'block_end'] = 1
    block_start_times = df_group.loc[df_group['block_start'] == 1, 'timestamp'].values
    block_end_times = df_group.loc[df_group['block_end'] == 1, 'timestamp'].values
    block_durations = block_end_times - block_start_times
    valid_block_mask = block_durations > np.timedelta64(30, 'm')
    df_group['valid_block'] = 0
    df_group.loc[df_group['block_start'] == 1, 'valid_block'] = valid_block_mask.astype(int)
    df_group.loc[df_group['block_end'] == 1, 'valid_block'] = valid_block_mask.astype(int)
    # Step 8
    gap_durations = block_start_times[1:] - block_end_times[:-1]
    valid_gap_mask = gap_durations < np.timedelta64(60, 'm')
    gap_start_indices = df_group[df_group['block_end'] == 1].index[:-1][valid_gap_mask]
    gap_end_indices = df_group[df_group['block_start'] == 1].index[1:][valid_gap_mask]
    df_group.loc[gap_start_indices, 'valid_block'] = 1
    df_group.loc[gap_end_indices, 'valid_block'] = 1
    # Step 9
    cum_valid_block = df_group['valid_block'].cumsum()
    sleep_period_length = df_group.groupby(['day_id', cum_valid_block])['valid_block'].sum()
    longest_block_index = sleep_period_length.idxmax()
    df_group['main_sleep_period'] = 0
    df_group.loc[(df_group['day_id'] == longest_block_index[0]) & (cum_valid_block == longest_block_index[1]), 'main_sleep_period'] = 1
    df_group['rolling_algo_awake'] = (~(df_group['valid_block'] | df_group['main_sleep_period'])).astype(int).fillna(method="ffill")
    return df_group


