import pandas as pd
import numpy as np
import gc

def make_features(df):
    # parse the timestamp and create an "hour" feature

    df['series_id'] = df['series_id'].astype('category')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df["hour"] = df["timestamp"].dt.hour
    
    # Use .fillna(value = 0) or .interpolate(method='linear')?
    df["anglezdiff"] = df["anglez"].diff().abs().astype(np.float32)

    df.sort_values(['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    # Rolling window algo
    result_dfs = []
    for _, group in df.groupby('series_id'):
        processed_group = rolling_window_algo(group)
        result_dfs.append(processed_group)

    algo_df = pd.concat(result_dfs)
    df = algo_df.dropna(axis = 0, subset=['hour'])
    del algo_df, result_dfs;gc.collect()
    

    diff_periods = [10, 20, 30]
    for diff_period in diff_periods:
        df[f"anglez_diff_{diff_period}"] = df.groupby('series_id')['anglez'].diff(diff_periods).astype(np.float16)
        df[f"enmo_diff_{diff_period}"] = df.groupby('series_id')['enmo'].diff(diff_periods).astype(np.float16)
    
    for col in ['enmo', 'anglez', 'anglezdiff', 'enmo_diff_10', 'enmo_diff_20', 'enmo_diff_60', 'anglez_diff_10', 'anglez_diff_20', 'anglez_diff_60']:
        
        # periods in seconds        
        periods = [60, 360, 720] 
        
        for n in periods:
            
            rol_args = {'window':f'{n+5}s', 'min_periods':10, 'center':True}
            
            for agg in ['median', 'mean', 'max', 'min', 'var']:
                df[f'{col}_{agg}_{n}'] = df[col].rolling(**rol_args).agg(agg).astype(np.float16).values
                gc.collect()
            
            if n == max(periods):
                df[f'{col}_mad_{n}'] = (df[col] - df[f'{col}_median_{n}']).abs().rolling(**rol_args).median().astype(np.float16)
            
            df[f'{col}_amplit_{n}'] = df[f'{col}_max_{n}']-df[f'{col}_min_{n}']
            df[f'{col}_amplit_{n}_min'] = df[f'{col}_amplit_{n}'].rolling(**rol_args).min().astype(np.float16).values
            
            df[f'{col}_diff_{n}_max'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).max().astype(np.float16)
            df[f'{col}_diff_{n}_mean'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).mean().astype(np.float16)

    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    return df

def rolling_window_algo(df_group):
    
    # Steps 3-5
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

