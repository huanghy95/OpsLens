from .utils import get_generated_functions_excluding_called
def get_traceCoder_generated_functions():
    """
    Get all dynamically generated functions in the current module, excluding functions that are called by other functions.

    Returns:
        list: Returns a list of all dynamically generated function objects that are not called by other functions.
    """
    import sys
    import traceback

    try:
        # Get the current module object
        current_module = sys.modules[__name__]

        # Use the common utility function to get functions excluding called ones
        return get_generated_functions_excluding_called(current_module)

    except Exception as e:
        print(f"Error in get_traceCoder_generated_functions: {e}")
        print(traceback.format_exc())
        return []



import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
import numpy as np
from typing import Annotated, List, Dict, Any
def find_trace_chain_anomalies_aff37679(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    start_str: Annotated[str, "Start time in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:05:00'"],
    end_str: Annotated[str, "End time in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:25:00'"],
    count_drop_agg_freq_initial: Annotated[str, "Aggregation frequency for initial count, e.g., '1T' for 1 minute"] = "1T",
    count_drop_window_minutes: Annotated[int, "Window size for rolling sum in minutes, e.g., 3"] = 3,
    count_drop_threshold_ratio: Annotated[float, "Threshold ratio for count drop detection, e.g., 0.5 for 50%"] = 0.5,
    count_drop_min_components: Annotated[int, "Minimum number of components showing simultaneous drops to trigger an event, e.g., 1"] = 1,
    window_extension_minutes: Annotated[int, "Minutes to extend the window earlier for baseline calculation, e.g., 5"] = 5
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Detects 'broken chain' anomalies in trace data by identifying significant, simultaneous drops in span counts across multiple components. This can indicate a service outage or a broken call chain.

    Example:
    This tool analyzes span counts in time windows to find periods where the number of traces suddenly decreases.

    Parameters:
    - date_str: The date of the telemetry data to analyze.
    - start_str: The start of the diagnostic time window.
    - end_str: The end of the diagnostic time window.
    - count_drop_agg_freq_initial: The time interval for aggregating span counts initially.
    - count_drop_window_minutes: The size of the rolling window (in minutes) to sum the initial counts.
    - count_drop_threshold_ratio: The minimum relative drop in the rolling count to be considered an anomaly.
    - count_drop_min_components: The minimum number of components that must exhibit a drop simultaneously to trigger an anomaly event.
    - window_extension_minutes: How many minutes to extend the time window backward to ensure a stable baseline for comparison.

    Returns:
    - A list of anomaly event dictionaries, where each dictionary represents a detected broken chain event.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    trace_path = os.path.join(data_source_path, date_str, 'trace', 'trace_span.csv')
    
    tz = pytz.timezone('Asia/Shanghai')
    
    try:
        start_time = tz.localize(datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S'))
        end_time = tz.localize(datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError as e:
        raise ValueError(f"Invalid time format: {e}. Please use 'YYYY-MM-DD HH:MM:SS'.")

    extended_start_time = start_time - timedelta(minutes=window_extension_minutes)
    
    try:
        df = pd.read_csv(trace_path)
    except FileNotFoundError:
        return []

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(tz)
    
    df_filtered = df[(df['datetime'] >= extended_start_time) & (df['datetime'] <= end_time)]
    
    # Filter out irrelevant components
    excluded_components = ['dockerA1', 'dockerA2', 'dockerB1', 'dockerB2']
    df_filtered = df_filtered[~df_filtered['cmdb_id'].isin(excluded_components)]

    if df_filtered.empty:
        return []

    # Resample to get 1-minute span counts for each component
    span_counts = df_filtered.set_index('datetime').groupby('cmdb_id').resample(count_drop_agg_freq_initial).size()
    count_df = span_counts.unstack(level='cmdb_id').fillna(0)
    
    # Ensure all components from the original filtered dataframe are present as columns
    all_components = df_filtered['cmdb_id'].unique()
    for comp in all_components:
        if comp not in count_df.columns:
            count_df[comp] = 0
    count_df = count_df.sort_index(axis=1)


    # Calculate rolling sum over the specified window (e.g., 3 minutes)
    rolling_sum_df = count_df.rolling(window=f"{count_drop_window_minutes}T").sum()
    
    # Shift by the window size to get the previous, non-overlapping window's sum
    prev_rolling_sum_df = rolling_sum_df.shift(periods=count_drop_window_minutes)
    
    # Calculate the drop and the ratio. A positive ratio indicates a drop.
    delta = prev_rolling_sum_df - rolling_sum_df
    # Replace NaNs and infinities resulting from division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        drop_ratio = (delta / prev_rolling_sum_df).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Identify where the drop ratio exceeds the threshold
    is_anomaly = drop_ratio > count_drop_threshold_ratio
    
    # Count how many components are anomalous at each timestamp
    anomaly_counts = is_anomaly.sum(axis=1)
    
    # Find timestamps where the number of anomalous components meets the minimum requirement
    simultaneous_anomaly_times = anomaly_counts[anomaly_counts >= count_drop_min_components].index
    
    anomaly_events = []
    
    # Filter for anomalies within the original diagnostic window
    final_anomaly_times = simultaneous_anomaly_times[
        (simultaneous_anomaly_times >= start_time) & 
        (simultaneous_anomaly_times <= end_time)
    ]

    for ts in final_anomaly_times:
        anomalous_components_at_ts = is_anomaly.loc[ts]
        anomalous_components = anomalous_components_at_ts[anomalous_components_at_ts].index.tolist()
        
        for comp in anomalous_components:
            current_val = rolling_sum_df.loc[ts, comp]
            prev_val = prev_rolling_sum_df.loc[ts, comp]
            
            if prev_val > 0:
                actual_ratio = (prev_val - current_val) / prev_val
                description = (
                    f"Span count dropped by {actual_ratio:.2%} in a {count_drop_window_minutes}-minute window. "
                    f"From {int(prev_val)} to {int(current_val)}."
                )
                
                anomaly_event = {
                    "data_source": "Trace",
                    "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S'),
                    "cmdb_id": comp,
                    "description": description,
                }
                anomaly_events.append(anomaly_event)

    return anomaly_events
if __name__ == '__main__':
    date = "2021_03_04"
    start = "2021-03-04 18:05:00"
    end = "2021-03-04 18:25:00"
    
    # Using default thresholds as they are based on the problem description's guidelines.
    # COUNT_DROP_MIN_COMPONENTS = 1 to catch any potential issue.
    # COUNT_DROP_THRESHOLD_RATIO = 0.5 for a significant drop.
    anomaly_events = find_trace_chain_anomalies_aff37679(
        date_str=date,
        start_str=start,
        end_str=end,
        count_drop_threshold_ratio=0.5,
        count_drop_min_components=1
    )
    
    print(anomaly_events)
    if anomaly_events:
        df_out = pd.DataFrame(anomaly_events)
        df_out.to_csv("anomaly_events_trace.csv", index=False)



import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from typing import Annotated, List, Dict, Any
def find_trace_chain_anomalies_bd2bad31(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_06'"],
    start_time_str: Annotated[str, "Start time of the analysis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-06 06:00:00'"],
    end_time_str: Annotated[str, "End time of the analysis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-06 06:30:00'"],
    count_drop_agg_freq: Annotated[str, "Aggregation frequency for span counts. Pandas frequency string, e.g., '3min'."] = "3min",
    count_drop_threshold_ratio: Annotated[float, "Threshold ratio for detecting a significant drop in span count (e.g., 0.5 for a 50% drop)."] = 0.5,
    count_drop_min_components: Annotated[int, "Minimum number of components that must show a drop in the same time window to trigger an event."] = 1,
    window_extension_minutes: Annotated[int, "Minutes to extend the window earlier for baseline calculation."] = 6
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Detects 'trace broken chain' issues by identifying significant drops in span counts for components.
    It aggregates span counts in fixed time windows, compares the count of each window to the previous one,
    and flags an anomaly if the count drops below a specified threshold for a minimum number of components simultaneously.
    This tool focuses on network-related components: Tomcat, IG, and MG.

    Parameters:
    - date_str: The date of the telemetry data to analyze.
    - start_time_str: The start of the analysis window.
    - end_time_str: The end of the analysis window.
    - count_drop_agg_freq: The time window for aggregating span counts.
    - count_drop_threshold_ratio: The relative drop in count required to be considered an anomaly.
    - count_drop_min_components: The minimum number of components that must exhibit a drop to trigger an event.
    - window_extension_minutes: How many minutes to extend the time window backward to ensure a baseline is available.

    Returns:
    - A list of anomaly event dictionaries, each describing a detected trace broken chain event.
    """
    
    data_source_path = './coding/dataset/Bank/telemetry'
    trace_path = os.path.join(data_source_path, date_str, 'trace', 'trace_span.csv')
    
    tz = pytz.timezone('Asia/Shanghai')
    
    try:
        original_start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        original_end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        # This case should not happen with correct inputs but is good practice.
        return []

    extended_start_time = original_start_time - timedelta(minutes=window_extension_minutes)
    
    start_ms = int(extended_start_time.timestamp() * 1000)
    end_ms = int(original_end_time.timestamp() * 1000)

    try:
        df = pd.read_csv(trace_path)
    except FileNotFoundError:
        return []

    df_filtered = df[(df['timestamp'] >= start_ms) & (df['timestamp'] < end_ms)]
    
    target_components = ['Tomcat', 'IG', 'MG']
    df_filtered = df_filtered[df_filtered['cmdb_id'].str.contains('|'.join(target_components), na=False)]

    if df_filtered.empty:
        return []

    df_filtered['datetime'] = pd.to_datetime(df_filtered['timestamp'], unit='ms', utc=True).dt.tz_convert(tz)
    df_filtered = df_filtered.set_index('datetime')

    span_counts = df_filtered.groupby('cmdb_id').resample(count_drop_agg_freq, label='right').size().reset_index(name='span_count')
    
    all_anomalies = []
    for cmdb_id, group in span_counts.groupby('cmdb_id'):
        group = group.copy()
        group['previous_span_count'] = group['span_count'].shift(1)
        
        group = group.dropna(subset=['previous_span_count'])
        
        anomalous_drops = group[
            (group['previous_span_count'] > 0) & 
            (group['span_count'] < group['previous_span_count'] * (1 - count_drop_threshold_ratio))
        ]
        
        for _, row in anomalous_drops.iterrows():
            all_anomalies.append({
                'timestamp': row['datetime'],
                'cmdb_id': cmdb_id,
                'current_count': row['span_count'],
                'previous_count': row['previous_span_count']
            })

    if not all_anomalies:
        return []

    anomalies_df = pd.DataFrame(all_anomalies)
    
    anomaly_counts_by_time = anomalies_df.groupby('timestamp').size()
    triggered_timestamps = anomaly_counts_by_time[anomaly_counts_by_time >= count_drop_min_components].index

    final_anomalies_df = anomalies_df[anomalies_df['timestamp'].isin(triggered_timestamps)]

    final_anomalies_df = final_anomalies_df[
        (final_anomalies_df['timestamp'] > original_start_time) & 
        (final_anomalies_df['timestamp'] <= original_end_time)
    ]

    anomaly_events = []
    for _, row in final_anomalies_df.iterrows():
        delta = row['previous_count'] - row['current_count']
        if row['previous_count'] > 0:
            drop_ratio = delta / row['previous_count']
        else:
            drop_ratio = 0

        description = (f"'span_count' is abnormally low. "
                       f"Current count in window: {int(row['current_count'])}, "
                       f"Previous count in window: {int(row['previous_count'])}, "
                       f"Drop ratio: {drop_ratio:.2f}")

        anomaly_events.append({
            'data_source': 'Trace',
            'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'cmdb_id': row['cmdb_id'],
            'description': description
        })
        
    anomaly_events.sort(key=lambda x: (x['timestamp'], x['cmdb_id']))

    return anomaly_events
if __name__ == '__main__':
    date_str = '2021_03_06'
    start_time_str = '2021-03-06 06:00:00'
    end_time_str = '2021-03-06 06:30:00'
    
    anomaly_events = find_trace_chain_anomalies_bd2bad31(
        date_str=date_str,
        start_time_str=start_time_str,
        end_time_str=end_time_str,
        count_drop_threshold_ratio=0.50,
        count_drop_min_components=1
    )
    
    print(anomaly_events)



import pandas as pd
import os
from typing import Annotated, List, Dict, Any
from datetime import datetime
import pytz
def find_anomalous_spans_by_duration_e4e3b926(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_09'"],
    buffered_window_start_str: Annotated[str, "Start of the buffered window for baseline calculation (UTC+8), e.g., '2021-03-09 13:55:00'"],
    buffered_window_end_str: Annotated[str, "End of the buffered window for baseline calculation (UTC+8), e.g., '2021-03-09 14:35:00'"],
    failure_window_start_str: Annotated[str, "Start of the user-visible failure window for reporting (UTC+8), e.g., '2021-03-09 14:00:00'"],
    failure_window_end_str: Annotated[str, "End of the user-visible failure window for reporting (UTC+8), e.g., '2021-03-09 14:30:00'"],
    target_components: Annotated[List[str], "List of cmdb_ids to focus on, e.g., ['Tomcat04', 'Tomcat03', 'MG02', 'IG02', 'Tomcat01']"],
    std_multiplier: Annotated[float, "Multiplier for standard deviation to set the anomaly threshold, e.g., 15.0"] = 15.0
) -> List[Dict[str, Any]]:
    """
    Analyzes trace span data to find and group the most severe anomalous spans based on duration. It calculates a
    baseline (mean + N * std) from a buffered time window and reports spans exceeding this high threshold within a
    specified failure window. Anomalies are grouped by component and 1-minute intervals to reduce noise.

    Parameters:
    - date_str: The date of the telemetry data to analyze.
    - buffered_window_start_str: Start of the time window for baseline calculation (UTC+8).
    - buffered_window_end_str: End of the time window for baseline calculation (UTC+8).
    - failure_window_start_str: Start of the time window for reporting anomalies (UTC+8).
    - failure_window_end_str: End of the time window for reporting anomalies (UTC+8).
    - target_components: A list of specific cmdb_ids to analyze.
    - std_multiplier: The number of standard deviations above the mean to define an anomaly. A higher value focuses on more severe anomalies.

    Returns:
    - A list of dictionaries, where each dictionary represents a grouped anomalous event.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    trace_file_path = os.path.join(data_source_path, date_str, 'trace', 'trace_span.csv')

    tz = pytz.timezone('Asia/Shanghai')

    try:
        buffered_start_dt = tz.localize(datetime.strptime(buffered_window_start_str, '%Y-%m-%d %H:%M:%S'))
        buffered_end_dt = tz.localize(datetime.strptime(buffered_window_end_str, '%Y-%m-%d %H:%M:%S'))
        failure_start_dt = tz.localize(datetime.strptime(failure_window_start_str, '%Y-%m-%d %H:%M:%S'))
        failure_end_dt = tz.localize(datetime.strptime(failure_window_end_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError as e:
        raise ValueError(f"Incorrect date format, should be YYYY-MM-DD HH:MM:SS. Details: {e}")

    buffered_start_ts_ms = int(buffered_start_dt.timestamp() * 1000)
    buffered_end_ts_ms = int(buffered_end_dt.timestamp() * 1000)
    failure_start_ts_ms = int(failure_start_dt.timestamp() * 1000)
    failure_end_ts_ms = int(failure_end_dt.timestamp() * 1000)

    try:
        df = pd.read_csv(trace_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trace data file not found at {trace_file_path}")

    df_buffered = df[(df['timestamp'] >= buffered_start_ts_ms) & (df['timestamp'] <= buffered_end_ts_ms)].copy()

    if df_buffered.empty:
        return []

    mean_duration = df_buffered['duration'].mean()
    std_duration = df_buffered['duration'].std()
    threshold = mean_duration + std_multiplier * std_duration

    anomalous_spans = df_buffered[df_buffered['duration'] > threshold].copy()

    final_anomalies = anomalous_spans[
        (anomalous_spans['timestamp'] >= failure_start_ts_ms) &
        (anomalous_spans['timestamp'] <= failure_end_ts_ms) &
        (anomalous_spans['cmdb_id'].isin(target_components))
    ].copy()

    if final_anomalies.empty:
        return []

    final_anomalies['datetime'] = pd.to_datetime(final_anomalies['timestamp'], unit='ms', utc=True).dt.tz_convert(tz)
    final_anomalies['time_bin'] = final_anomalies['datetime'].dt.floor('1T')

    grouped_anomalies = final_anomalies.groupby(['cmdb_id', 'time_bin'])

    anomaly_events = []
    for (cmdb_id, time_bin), group in grouped_anomalies:
        count = len(group)
        max_duration = group['duration'].max()
        
        formatted_timestamp = time_bin.strftime('%Y-%m-%d %H:%M:%S')
        
        description = (
            f"{count} anomalous spans detected for {cmdb_id} in 1-minute window starting {formatted_timestamp}. "
            f"Max duration: {max_duration}ms, exceeding threshold of {threshold:.2f}ms."
        )

        anomaly_event = {
            "data_source": "Trace",
            "timestamp": formatted_timestamp,
            "cmdb_id": cmdb_id,
            "description": description,
        }
        anomaly_events.append(anomaly_event)
    
    anomaly_events.sort(key=lambda x: (x['timestamp'], x['cmdb_id']))

    return anomaly_events
if __name__ == '__main__':
    date = '2021_03_09'
    buf_start = '2021-03-09 13:55:00'
    buf_end = '2021-03-09 14:35:00'
    fail_start = '2021-03-09 14:00:00'
    fail_end = '2021-03-09 14:30:00'
    components = ['Tomcat04', 'Tomcat03', 'MG02', 'IG02', 'Tomcat01']

    anomaly_events_result = find_anomalous_spans_by_duration_e4e3b926(
        date_str=date,
        buffered_window_start_str=buf_start,
        buffered_window_end_str=buf_end,
        failure_window_start_str=fail_start,
        failure_window_end_str=fail_end,
        target_components=components,
        std_multiplier=15.0  # Increased threshold to focus on most severe anomalies
    )

    print(anomaly_events_result)
    
    if anomaly_events_result:
        output_df = pd.DataFrame(anomaly_events_result)
        output_df.to_csv("anomaly_events.csv", index=False)



import pandas as pd
import os
from typing import Annotated, List, Dict, Any
from datetime import datetime
import pytz
def diagnose_and_group_downstream_faulty_spans_95e3fbf3(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_09'"],
    buffered_window_start_str: Annotated[str, "Start of the buffered window for data loading (UTC+8), e.g., '2021-03-09 13:55:00'"],
    buffered_window_end_str: Annotated[str, "End of the buffered window for data loading (UTC+8), e.g., '2021-03-09 14:35:00'"],
    failure_window_start_str: Annotated[str, "Start of the user-visible failure window for reporting (UTC+8), e.g., '2021-03-09 14:00:00'"],
    failure_window_end_str: Annotated[str, "End of the user-visible failure window for reporting (UTC+8), e.g., '2021-03-09 14:30:00'"],
    target_components: Annotated[List[str], "List of cmdb_ids to focus on, e.g., ['Tomcat04', 'Tomcat03', 'MG02', 'IG02', 'Tomcat01']"],
    duration_threshold: Annotated[float, "The duration in ms to use as the anomaly threshold, e.g., 10500.0"]
) -> List[Dict[str, Any]]:
    """
    Identifies witness spans, finds the most-downstream faulty span for each trace, groups them by component
    and time, and reports those that overlap with a failure window as summarized trace diagnosis events.

    Parameters:
    - date_str: The date of the telemetry data.
    - buffered_window_start_str: Start of the time window for loading spans.
    - buffered_window_end_str: End of the time window for loading spans.
    - failure_window_start_str: Start of the user-visible failure window for reporting.
    - failure_window_end_str: End of the user-visible failure window for reporting.
    - target_components: A list of specific cmdb_ids to analyze.
    - duration_threshold: The duration in milliseconds to consider a span anomalous.

    Returns:
    - A list of dictionaries, each representing a grouped trace diagnosis event.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    trace_file_path = os.path.join(data_source_path, date_str, 'trace', 'trace_span.csv')
    tz = pytz.timezone('Asia/Shanghai')

    # Convert window strings to millisecond timestamps
    try:
        buffered_start_ts = int(tz.localize(datetime.strptime(buffered_window_start_str, '%Y-%m-%d %H:%M:%S')).timestamp() * 1000)
        buffered_end_ts = int(tz.localize(datetime.strptime(buffered_window_end_str, '%Y-%m-%d %H:%M:%S')).timestamp() * 1000)
        failure_start_ts = int(tz.localize(datetime.strptime(failure_window_start_str, '%Y-%m-%d %H:%M:%S')).timestamp() * 1000)
        failure_end_ts = int(tz.localize(datetime.strptime(failure_window_end_str, '%Y-%m-%d %H:%M:%S')).timestamp() * 1000)
    except ValueError as e:
        raise ValueError(f"Incorrect date format, should be YYYY-MM-DD HH:MM:SS. Details: {e}")

    # Load and filter data by the buffered window
    try:
        df = pd.read_csv(trace_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trace data file not found at {trace_file_path}")

    df_buffered = df[(df['timestamp'] >= buffered_start_ts) & (df['timestamp'] <= buffered_end_ts)].copy()

    # Identify initial "witness spans"
    witness_spans_df = df_buffered[
        (df_buffered['duration'] > duration_threshold) &
        (df_buffered['cmdb_id'].isin(target_components))
    ].copy()

    if witness_spans_df.empty:
        return []

    # Prepare raw witness span data
    witness_spans_df['start_ts_ms'] = witness_spans_df['timestamp']
    witness_spans_df['end_ts_ms'] = witness_spans_df['timestamp'] + witness_spans_df['duration']
    witness_spans_df['duration_ms'] = witness_spans_df['duration']
    
    output_cols = ['trace_id', 'span_id', 'cmdb_id', 'start_ts_ms', 'end_ts_ms', 'duration_ms']
    witness_spans_df[output_cols].to_csv('witness_spans.csv', index=False)

    # Identify the most-downstream faulty span for each trace
    most_downstream_spans = []
    for trace_id in witness_spans_df['trace_id'].unique():
        trace_witnesses = witness_spans_df[witness_spans_df['trace_id'] == trace_id]
        parent_ids_in_faulty_trace = set(trace_witnesses['parent_id'])
        
        for _, span in trace_witnesses.iterrows():
            if span['span_id'] not in parent_ids_in_faulty_trace:
                most_downstream_spans.append(span)

    if not most_downstream_spans:
        return []
        
    most_downstream_df = pd.DataFrame(most_downstream_spans).drop_duplicates(subset=['trace_id', 'span_id'])

    # Filter for spans overlapping the user-visible failure window
    overlapping_anomalies = most_downstream_df[
        (most_downstream_df['start_ts_ms'] < failure_end_ts) &
        (most_downstream_df['end_ts_ms'] > failure_start_ts)
    ].copy()

    if overlapping_anomalies.empty:
        return []

    # Group anomalies by component and time bin
    overlapping_anomalies['datetime'] = pd.to_datetime(overlapping_anomalies['start_ts_ms'], unit='ms', utc=True).dt.tz_convert(tz)
    overlapping_anomalies['time_bin'] = overlapping_anomalies['datetime'].dt.floor('1T')
    
    grouped_anomalies = overlapping_anomalies.groupby(['cmdb_id', 'time_bin'])

    anomaly_events = []
    for (cmdb_id, time_bin), group in grouped_anomalies:
        count = len(group)
        max_duration = group['duration_ms'].max()
        formatted_timestamp = time_bin.strftime('%Y-%m-%d %H:%M:%S')
        
        description = (
            f"{count} most-downstream faulty spans found for {cmdb_id} in 1-minute window starting {formatted_timestamp}. "
            f"Max duration: {max_duration}ms, exceeding threshold of {duration_threshold:.2f}ms."
        )
        
        anomaly_event = {
            "data_source": "Trace",
            "timestamp": formatted_timestamp,
            "cmdb_id": cmdb_id,
            "description": description,
        }
        anomaly_events.append(anomaly_event)
        
    anomaly_events.sort(key=lambda x: (x['timestamp'], x['cmdb_id']))
    
    return anomaly_events
if __name__ == '__main__':
    anomaly_events_result = diagnose_and_group_downstream_faulty_spans_95e3fbf3(
        date_str='2021_03_09',
        buffered_window_start_str='2021-03-09 13:55:00',
        buffered_window_end_str='2021-03-09 14:35:00',
        failure_window_start_str='2021-03-09 14:00:00',
        failure_window_end_str='2021-03-09 14:30:00',
        target_components=['Tomcat04', 'Tomcat03', 'MG02', 'IG02', 'Tomcat01'],
        duration_threshold=10500.0  # Increased threshold
    )
    
    print(anomaly_events_result)
    if anomaly_events_result:
        output_df = pd.DataFrame(anomaly_events_result)
        output_df.to_csv("anomaly_events.csv", index=False)



import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
from typing import Annotated, List, Dict, Any, Optional
def detect_span_duration_anomalies_72617c2e(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    start_time_str: Annotated[str, "Start time of the diagnostic window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnostic window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:30:00'"],
    target_cmdb_ids: Annotated[Optional[List[str]], "List of specific component IDs to focus on. If None, analyzes all relevant components."] = None,
    window_extension_minutes: Annotated[int, "Minutes to extend the window for context, e.g., 5."] = 5,
    percentile_threshold: Annotated[float, "Percentile for duration threshold (0-100), e.g., 99.0."] = 99.0,
    std_dev_multiplier: Annotated[float, "Multiplier for standard deviation threshold, e.g., 4.0."] = 4.0,
    absolute_duration_threshold_ms: Annotated[int, "Absolute duration threshold in milliseconds, e.g., 3000."] = 3000,
    grouping_window: Annotated[str, "Time window for grouping anomalies, e.g., '10T' for 10 minutes."] = '10T',
    min_anomalous_spans_in_group: Annotated[int, "Minimum number of anomalous spans required to create a group event, e.g., 5."] = 5
) -> List[Dict[str, Any]]:
    """
    Detects, groups, and filters severe span duration anomalies using aggressive consolidation.

    This tool identifies spans with critically high durations and uses wide temporal grouping to
    summarize persistent issues and minimize alert volume. The process is as follows:
    1.  Load the entire day's trace data to compute robust baseline duration statistics.
    2.  Identify spans as anomalous only if they exceed extremely strict thresholds (e.g., 99th percentile, mean+4*std, or a very high absolute value like 3000ms).
    3.  Filter for "most-downstream" anomalous spans to pinpoint the root cause.
    4.  Group these severe anomalies by component and a very wide time window (e.g., 10 minutes).
    5.  Create a summary event for a group ONLY IF it contains a significant number of anomalous spans (e.g., 5 or more).
    6.  Report only the final, highly-filtered, and consolidated anomalies that fall within the original diagnostic time window.

    Parameters:
    - date_str: The date of the telemetry data to analyze ('YYYY_MM_DD').
    - start_time_str: The start of the primary diagnostic window ('YYYY-MM-DD HH:MM:SS').
    - end_time_str: The end of the primary diagnostic window ('YYYY-MM-DD HH:MM:SS').
    - target_cmdb_ids: Optional list of cmdb_ids to analyze.
    - window_extension_minutes: How many minutes to look back/forward for context.
    - percentile_threshold: The high percentile used for the statistical threshold.
    - std_dev_multiplier: The high multiplier for the standard deviation threshold.
    - absolute_duration_threshold_ms: A very high absolute cutoff for span duration.
    - grouping_window: The wide time frequency for grouping anomalies (pandas frequency string).
    - min_anomalous_spans_in_group: The minimum number of anomalies in a group to be reported.

    Returns:
    - A list of dictionaries, where each dictionary represents a highly significant, consolidated anomaly event.
    """
    if target_cmdb_ids is None:
        target_cmdb_ids = ['Tomcat03', 'Redis02', 'MG01', 'IG02', 'MG02', 'IG01']

    data_source_path = './coding/dataset/Bank/telemetry'
    trace_path = os.path.join(data_source_path, date_str, 'trace', 'trace_span.csv')

    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace data file not found at: {trace_path}")

    tz = pytz.timezone('Asia/Shanghai')

    try:
        original_start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        original_end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError as e:
        raise ValueError(f"Incorrect time format. Please use 'YYYY-MM-DD HH:MM:SS'. Error: {e}")

    extended_start_time = original_start_time - timedelta(minutes=window_extension_minutes)
    extended_end_time = original_end_time + timedelta(minutes=window_extension_minutes)

    df_full = pd.read_csv(trace_path)
    df_full = df_full[~df_full['cmdb_id'].str.contains('docker', na=False)]
    df_full['datetime'] = pd.to_datetime(df_full['timestamp'], unit='ms', utc=True).dt.tz_convert(tz)

    baseline_stats = df_full.groupby('cmdb_id')['duration'].agg(
        mean='mean',
        std='std',
        p_thresh=lambda x: x.quantile(percentile_threshold / 100.0)
    ).fillna(0)

    df_window = df_full[
        (df_full['datetime'] >= extended_start_time) &
        (df_full['datetime'] < extended_end_time)
    ].copy()

    if target_cmdb_ids:
        df_window = df_window[df_window['cmdb_id'].isin(target_cmdb_ids)]

    if df_window.empty:
        return []

    df_window = df_window.join(baseline_stats, on='cmdb_id')

    df_window['is_anomalous'] = (
        (df_window['duration'] > df_window['p_thresh']) & (df_window['p_thresh'] > 0) |
        (df_window['duration'] > (df_window['mean'] + std_dev_multiplier * df_window['std'])) & (df_window['std'] > 0) |
        (df_window['duration'] > absolute_duration_threshold_ms)
    )

    df_anomalous = df_window[df_window['is_anomalous']].copy()

    if df_anomalous.empty:
        return []

    parent_ids = set(df_full['parent_id'].dropna())
    df_anomalous['is_downstream'] = ~df_anomalous['span_id'].isin(parent_ids)
    
    df_downstream_anomalous = df_anomalous[df_anomalous['is_downstream']]

    df_final = df_downstream_anomalous[
        (df_downstream_anomalous['datetime'] >= original_start_time) &
        (df_downstream_anomalous['datetime'] < original_end_time)
    ]

    if df_final.empty:
        return []

    anomaly_events = []
    grouped = df_final.groupby(['cmdb_id', pd.Grouper(key='datetime', freq=grouping_window)])

    for (cmdb_id, timestamp_group), group in grouped:
        if group.empty:
            continue
        
        count = len(group)
        if count < min_anomalous_spans_in_group:
            continue
        
        max_duration = group['duration'].max()
        avg_duration = group['duration'].mean()
        
        description = (
            f"{count} anomalous downstream spans detected for {cmdb_id} in this {grouping_window.replace('T','-minute')} window. "
            f"Max duration: {max_duration}ms, Avg duration: {avg_duration:.0f}ms."
        )
        
        anomaly_events.append({
            "data_source": "Trace",
            "timestamp": timestamp_group.strftime('%Y-%m-%d %H:%M:%S'),
            "cmdb_id": cmdb_id,
            "description": description
        })
        
    return anomaly_events
if __name__ == '__main__':
    date = '2021_03_04'
    start_time = '2021-03-04 18:00:00'
    end_time = '2021-03-04 18:30:00'
    
    target_components = ['Tomcat03', 'Redis02', 'MG01', 'IG02', 'MG02', 'IG01']

    # Use extremely strict thresholds and very wide grouping to get a high-level summary.
    anomaly_events = detect_span_duration_anomalies_72617c2e(
        date_str=date,
        start_time_str=start_time,
        end_time_str=end_time,
        target_cmdb_ids=target_components,
        absolute_duration_threshold_ms=3000,  # Increased to 3000ms
        percentile_threshold=99.0,
        std_dev_multiplier=4.0,
        grouping_window='10T',                # Widened to 10 minutes
        min_anomalous_spans_in_group=5        # Require at least 5 anomalies in a group
    )

    print(anomaly_events)


