from .utils import  get_generated_functions_excluding_called
def get_metricCoder_generated_functions():
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
        print(f"Error in get_metricCoder_generated_functions: {e}")
        print(traceback.format_exc())
        return []

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_aaddc52f(
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:30:00'"],
    date_str: Annotated[str, "Date string for locating data files, in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    stable_window_size: Annotated[int, "Number of data points to define a stable phase."] = 5,
    recovery_window_size: Annotated[int, "Number of data points to define a recovery phase."] = 5,
    anomaly_window_min: Annotated[int, "Minimum number of data points for an anomaly phase."] = 1,
    anomaly_window_max: Annotated[int, "Maximum number of data points for an anomaly phase."] = 8,
    stable_std_threshold: Annotated[float, "Standard deviation threshold for a phase to be considered stable (on a 0-100 normalized scale)."] = 10.0,
    jump_threshold_factor: Annotated[float, "A factor to determine a significant jump. The jump value must be greater than this factor times the stable phase's standard deviation."] = 3.0,
    noise_reduction_w: Annotated[float, "Weight for noise reduction within a metric. Anomalies with a delta less than w * max_delta for that metric are removed."] = 0.6,
    cluster_minutes_k: Annotated[int, "Time window in minutes to group anomalies into a single cluster."] = 3,
    max_clusters_nc: Annotated[int, "If a metric has more than this number of anomaly clusters, it's considered noisy and its anomalies are discarded."] = 3,
    final_filter_x: Annotated[float, "Final noise reduction factor. Anomalies with a delta less than x * overall_max_delta are removed."] = 0.2
) -> List[Dict[str, Any]]:
    """
    Detects metric anomalies by analyzing time-series data for significant spike or drop patterns within a specified time window.

    This tool performs a multi-stage analysis:
    1.  Loads metric data for a specified date.
    2.  For each relevant KPI, it normalizes the data across all components to a 0-100 scale.
    3.  It searches for an anomaly pattern defined by three phases: Stable -> Anomaly -> Recovery.
    4.  Anomalies are filtered based on their significance (delta) and temporal clustering to reduce noise.
    5.  The final list of anomalies is ranked by severity (delta) to highlight the most critical events.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity. Returns an empty list if no significant anomalies are found.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    
    TARGET_KPIS = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    tz = pytz.timezone('Asia/Shanghai')
    diag_start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    diag_end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    
    extended_start_time = diag_start_time - timedelta(minutes=15)
    extended_end_time = diag_end_time + timedelta(minutes=15)

    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Data file not found: {metric_path}")

    df = pd.read_csv(metric_path)
    df = df[df['kpi_name'].isin(TARGET_KPIS)]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)

    df = df[(df['timestamp'] >= extended_start_time) & (df['timestamp'] <= extended_end_time)]

    if df.empty:
        return []

    all_anomalies = []

    for kpi in TARGET_KPIS:
        kpi_df = df[df['kpi_name'] == kpi]
        if kpi_df.empty:
            continue

        pivot_df = kpi_df.pivot_table(index='timestamp', columns='cmdb_id', values='value').resample('1min').mean()
        pivot_df = pivot_df.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        
        if pivot_df.empty:
            continue

        # Normalization
        min_val, max_val = pivot_df.min().min(), pivot_df.max().max()
        if max_val == min_val:
            norm_df = pivot_df.apply(lambda x: 0)
        else:
            norm_df = (pivot_df - min_val) / (max_val - min_val) * 100

        kpi_anomalies = []
        
        diag_norm_df = norm_df[(norm_df.index >= diag_start_time) & (norm_df.index <= diag_end_time)]
        if diag_norm_df.empty:
            continue
            
        sorted_components = diag_norm_df.max().sort_values(ascending=False).index

        for cmdb_id in sorted_components:
            series = norm_df[cmdb_id].dropna()
            if len(series) < (stable_window_size + anomaly_window_min + recovery_window_size):
                continue

            component_events = []
            processed_indices = set()

            diag_series = series[(series.index >= diag_start_time) & (series.index <= diag_end_time)]
            sorted_points = diag_series.sort_values(ascending=False)

            for peak_time, peak_value in sorted_points.items():
                if peak_time in processed_indices or len(component_events) >= 2:
                    break
                
                peak_idx = series.index.get_loc(peak_time)

                for anom_len in range(anomaly_window_min, anomaly_window_max + 1):
                    for offset in range(anom_len):
                        start_idx = peak_idx - offset
                        end_idx = start_idx + anom_len - 1
                        
                        if start_idx < 0 or end_idx >= len(series): continue
                        
                        anomaly_start_time = series.index[start_idx]
                        anomaly_end_time = series.index[end_idx]

                        if not (diag_start_time <= anomaly_start_time <= diag_end_time and diag_start_time <= anomaly_end_time <= diag_end_time):
                            continue

                        # Stable phase
                        stable_end_idx = start_idx - 1
                        stable_start_idx = stable_end_idx - stable_window_size + 1
                        if stable_start_idx < 0: continue
                        
                        stable_series = series.iloc[stable_start_idx:stable_end_idx + 1]
                        if len(stable_series) != stable_window_size: continue
                        
                        stable_std = stable_series.std()
                        if stable_std > stable_std_threshold: continue

                        # Recovery phase
                        recovery_start_idx = end_idx + 1
                        recovery_end_idx = recovery_start_idx + recovery_window_size - 1
                        if recovery_end_idx >= len(series): continue

                        recovery_series = series.iloc[recovery_start_idx:recovery_end_idx + 1]
                        if len(recovery_series) != recovery_window_size: continue
                        if recovery_series.std() > stable_std_threshold: continue

                        # Sudden jump
                        jump = series.iloc[start_idx] - stable_series.iloc[-1]
                        if jump <= jump_threshold_factor * (stable_std if stable_std > 0 else 1):
                            continue
                        
                        # Event found
                        stable_mean = stable_series.mean()
                        anomaly_series = series.iloc[start_idx:end_idx+1]
                        actual_peak_val = anomaly_series.max()
                        delta = actual_peak_val - stable_mean
                        
                        if delta > 0:
                            event = {
                                'data_source': 'Metric',
                                'timestamp': anomaly_start_time,
                                'cmdb_id': cmdb_id,
                                'kpi_name': kpi,
                                'delta': delta,
                            }
                            component_events.append(event)
                            for i in range(stable_start_idx, recovery_end_idx + 1):
                                processed_indices.add(series.index[i])
                            break  # Move to next peak
                    if peak_time in processed_indices:
                        break
            
            kpi_anomalies.extend(component_events)

        # Noise reduction per metric
        if not kpi_anomalies:
            continue
        
        max_delta = max(event['delta'] for event in kpi_anomalies)
        denoised_anomalies = [event for event in kpi_anomalies if event['delta'] >= noise_reduction_w * max_delta]

        # Clustering
        if not denoised_anomalies:
            continue
            
        denoised_anomalies.sort(key=lambda x: x['timestamp'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for i in range(1, len(denoised_anomalies)):
                if (denoised_anomalies[i]['timestamp'] - current_cluster[-1]['timestamp']) <= timedelta(minutes=cluster_minutes_k):
                    current_cluster.append(denoised_anomalies[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [denoised_anomalies[i]]
            clusters.append(current_cluster)

        if 1 <= len(clusters) <= max_clusters_nc:
            all_anomalies.extend(denoised_anomalies)

    # Final filtering
    if not all_anomalies:
        return []

    overall_max_delta = max(event['delta'] for event in all_anomalies)
    final_anomalies = [event for event in all_anomalies if event['delta'] >= final_filter_x * overall_max_delta]

    # Sort and format
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)
    
    output_events = []
    for event in final_anomalies[:5]:
        output_events.append({
            'data_source': event['data_source'],
            'timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'cmdb_id': event['cmdb_id'],
            'description': f"'{event['kpi_name']}' is abnormally high, delta={event['delta']:.2f}"
        })

    return output_events
if __name__ == '__main__':
    start_time = "2021-03-04 18:00:00"
    end_time = "2021-03-04 18:30:00"
    date_str = "2021_03_04"
    
    anomaly_events = find_metric_anomalies_aaddc52f(start_time, end_time, date_str)
    
    print(anomaly_events)



import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_9bd4eead(
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-04 18:30:00'"],
    date_str: Annotated[str, "Date string for locating data files, in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    stable_window_size: Annotated[int, "Number of data points to define a stable phase."] = 5,
    recovery_window_size: Annotated[int, "Number of data points to define a recovery phase."] = 5,
    anomaly_window_min: Annotated[int, "Minimum number of data points for an anomaly phase."] = 1,
    anomaly_window_max: Annotated[int, "Maximum number of data points for an anomaly phase."] = 8,
    stable_std_threshold: Annotated[float, "Standard deviation threshold for a phase to be considered stable (on a 0-100 normalized scale)."] = 10.0,
    jump_threshold_factor: Annotated[float, "A factor to determine a significant jump. The jump value must be greater than this factor times the stable phase's standard deviation."] = 3.0,
    noise_reduction_w: Annotated[float, "Weight for noise reduction within a metric. Anomalies with a delta less than w * max_delta for that metric are removed."] = 0.6,
    cluster_minutes_k: Annotated[int, "Time window in minutes to group anomalies into a single cluster."] = 3,
    max_clusters_nc: Annotated[int, "If a metric has more than this number of anomaly clusters, it's considered noisy and its anomalies are discarded."] = 3,
    final_filter_x: Annotated[float, "Final noise reduction factor. Anomalies with a delta less than x * overall_max_delta are removed."] = 0.2
) -> List[Dict[str, Any]]:
    """
    Detects metric anomalies by analyzing time-series data for significant spike or drop patterns within a specified time window.

    This tool performs a multi-stage analysis:
    1.  Loads metric data for a specified date.
    2.  For each relevant KPI, it normalizes the data across all components to a 0-100 scale.
    3.  It searches for an anomaly pattern defined by three phases: Stable -> Anomaly -> Recovery.
    4.  Anomalies are filtered based on their significance (delta) and temporal clustering to reduce noise.
    5.  The final list of anomalies is ranked by severity (delta) to highlight the most critical events.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity. Returns an empty list if no significant anomalies are found.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    
    TARGET_KPIS = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    tz = pytz.timezone('Asia/Shanghai')
    diag_start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    diag_end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    
    extended_start_time = diag_start_time - timedelta(minutes=15)
    extended_end_time = diag_end_time + timedelta(minutes=15)

    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Data file not found: {metric_path}")

    df = pd.read_csv(metric_path)
    df = df[df['kpi_name'].isin(TARGET_KPIS)]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)

    df = df[(df['timestamp'] >= extended_start_time) & (df['timestamp'] <= extended_end_time)]

    if df.empty:
        return []

    all_anomalies = []

    for kpi in TARGET_KPIS:
        kpi_df = df[df['kpi_name'] == kpi]
        if kpi_df.empty:
            continue

        pivot_df = kpi_df.pivot_table(index='timestamp', columns='cmdb_id', values='value').resample('1min').mean()
        pivot_df = pivot_df.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        
        if pivot_df.empty:
            continue

        # Normalization
        min_val, max_val = pivot_df.min().min(), pivot_df.max().max()
        if max_val == min_val:
            norm_df = pivot_df.apply(lambda x: 0)
        else:
            norm_df = (pivot_df - min_val) / (max_val - min_val) * 100

        kpi_anomalies = []
        
        diag_norm_df = norm_df[(norm_df.index >= diag_start_time) & (norm_df.index <= diag_end_time)]
        if diag_norm_df.empty:
            continue
            
        sorted_components = diag_norm_df.max().sort_values(ascending=False).index

        for cmdb_id in sorted_components:
            series = norm_df[cmdb_id].dropna()
            if len(series) < (stable_window_size + anomaly_window_min + recovery_window_size):
                continue

            component_events = []
            processed_indices = set()

            diag_series = series[(series.index >= diag_start_time) & (series.index <= diag_end_time)]
            sorted_points = diag_series.sort_values(ascending=False)

            for peak_time, peak_value in sorted_points.items():
                if peak_time in processed_indices or len(component_events) >= 2:
                    break
                
                peak_idx = series.index.get_loc(peak_time)

                for anom_len in range(anomaly_window_min, anomaly_window_max + 1):
                    for offset in range(anom_len):
                        start_idx = peak_idx - offset
                        end_idx = start_idx + anom_len - 1
                        
                        if start_idx < 0 or end_idx >= len(series): continue
                        
                        anomaly_start_time = series.index[start_idx]
                        anomaly_end_time = series.index[end_idx]

                        if not (diag_start_time <= anomaly_start_time <= diag_end_time and diag_start_time <= anomaly_end_time <= diag_end_time):
                            continue

                        # Stable phase
                        stable_end_idx = start_idx - 1
                        stable_start_idx = stable_end_idx - stable_window_size + 1
                        if stable_start_idx < 0: continue
                        
                        stable_series = series.iloc[stable_start_idx:stable_end_idx + 1]
                        if len(stable_series) != stable_window_size: continue
                        
                        stable_std = stable_series.std()
                        if stable_std > stable_std_threshold: continue

                        # Recovery phase
                        recovery_start_idx = end_idx + 1
                        recovery_end_idx = recovery_start_idx + recovery_window_size - 1
                        if recovery_end_idx >= len(series): continue

                        recovery_series = series.iloc[recovery_start_idx:recovery_end_idx + 1]
                        if len(recovery_series) != recovery_window_size: continue
                        if recovery_series.std() > stable_std_threshold: continue

                        # Sudden jump
                        jump = series.iloc[start_idx] - stable_series.iloc[-1]
                        if jump <= jump_threshold_factor * (stable_std if stable_std > 0 else 1):
                            continue
                        
                        # Event found
                        stable_mean = stable_series.mean()
                        anomaly_series = series.iloc[start_idx:end_idx+1]
                        actual_peak_val = anomaly_series.max()
                        delta = actual_peak_val - stable_mean
                        
                        if delta > 0:
                            event = {
                                'data_source': 'Metric',
                                'timestamp': anomaly_start_time,
                                'cmdb_id': cmdb_id,
                                'kpi_name': kpi,
                                'delta': delta,
                            }
                            component_events.append(event)
                            for i in range(stable_start_idx, recovery_end_idx + 1):
                                processed_indices.add(series.index[i])
                            break  # Move to next peak
                    if peak_time in processed_indices:
                        break
            
            kpi_anomalies.extend(component_events)

        # Noise reduction per metric
        if not kpi_anomalies:
            continue
        
        max_delta = max(event['delta'] for event in kpi_anomalies)
        denoised_anomalies = [event for event in kpi_anomalies if event['delta'] >= noise_reduction_w * max_delta]

        # Clustering
        if not denoised_anomalies:
            continue
            
        denoised_anomalies.sort(key=lambda x: x['timestamp'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for i in range(1, len(denoised_anomalies)):
                if (denoised_anomalies[i]['timestamp'] - current_cluster[-1]['timestamp']) <= timedelta(minutes=cluster_minutes_k):
                    current_cluster.append(denoised_anomalies[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [denoised_anomalies[i]]
            clusters.append(current_cluster)

        if 1 <= len(clusters) <= max_clusters_nc:
            all_anomalies.extend(denoised_anomalies)

    # Final filtering
    if not all_anomalies:
        return []

    overall_max_delta = max(event['delta'] for event in all_anomalies)
    final_anomalies = [event for event in all_anomalies if event['delta'] >= final_filter_x * overall_max_delta]

    # Sort and format
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)
    
    output_events = []
    for event in final_anomalies[:10]:
        output_events.append({
            'data_source': event['data_source'],
            'timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'cmdb_id': event['cmdb_id'],
            'description': f"'{event['kpi_name']}' is abnormally high, delta={event['delta']:.2f}"
        })

    return output_events

if __name__ == '__main__':
    # Part 1: Rerun anomaly detection with expanded window and relaxed thresholds
    refined_start_time = "2021-03-04 17:55:00"
    refined_end_time = "2021-03-04 18:35:00"
    date_str = "2021_03_04"
    
    # Relaxed thresholds
    relaxed_stable_std = 15.0
    relaxed_jump_factor = 2.0
    relaxed_noise_w = 0.4
    relaxed_final_x = 0.1

    print("--- Refined Anomaly Detection Results ---")
    anomaly_events = find_metric_anomalies_9bd4eead(
        start_time_str=refined_start_time,
        end_time_str=refined_end_time,
        date_str=date_str,
        stable_std_threshold=relaxed_stable_std,
        jump_threshold_factor=relaxed_jump_factor,
        noise_reduction_w=relaxed_noise_w,
        final_filter_x=relaxed_final_x
    )
    print(anomaly_events)

    # Part 2: Extract raw time-series snippets for focused components
    snippets = [
        {'cmdb_id': 'Redis02', 'start_time': '18:05:00', 'end_time': '18:15:00'},
        {'cmdb_id': 'MG01', 'start_time': '18:10:00', 'end_time': '18:18:00'},
        {'cmdb_id': 'Mysql01', 'start_time': '18:18:00', 'end_time': '18:26:00'}
    ]
    




import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
from typing import Annotated, List, Dict, Any, Optional
def detect_metric_anomalies_from_files_51fa07e4(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_07'"],
    start_time_str: Annotated[str, "Diagnosis window start time in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-07 12:30:00'"],
    end_time_str: Annotated[str, "Diagnosis window end time in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-07 13:00:00'"],
    stable_window_size: Annotated[int, "Number of data points for the stable phase before an anomaly."] = 5,
    anomaly_window_size: Annotated[int, "Maximum number of data points for the anomaly phase (from start to peak)."] = 8,
    recovery_window_size: Annotated[int, "Number of data points for the recovery phase after an anomaly."] = 5,
    stable_std_thresh_factor: Annotated[float, "Threshold for standard deviation in the stable phase, as a factor of the stable mean."] = 1.5,
    ramp_up_thresh_factor: Annotated[float, "Factor to determine a sudden spike. Change must be > factor * stable_std."] = 2.0,
    noise_reduction_w: Annotated[float, "Threshold for metric-level noise reduction. Events with delta < w * max_delta are removed."] = 0.4,
    cluster_minutes_k: Annotated[int, "Time window in minutes to group anomalies into a single cluster."] = 3,
    max_clusters_n: Annotated[int, "Maximum number of clusters allowed for a metric to be considered a potential root cause."] = 3,
    final_noise_x: Annotated[float, "Threshold for final noise reduction. Events with delta < x * overall_max_delta are removed."] = 0.1
) -> List[Dict[str, Any]]:
    """
    Analyzes container metrics to detect anomalies based on a 'stable-anomaly-recovery' pattern.

    This tool loads metric data for a specified date and time window, normalizes the data for key performance indicators (KPIs),
    and scans for significant spikes or drops. It identifies anomalies by looking for a pattern of a stable period, followed by a
    sudden, sharp change, and then a return to a stable recovery state. The tool filters out noise, clusters related events
    to identify potential root cause metrics, and returns a ranked list of the most severe anomaly events.

    Parameters:
    - date_str: The date of the telemetry data to analyze (e.g., '2021_03_07').
    - start_time_str: The start of the diagnosis time window (e.g., '2021-03-07 12:30:00').
    - end_time_str: The end of the diagnosis time window (e.g., '2021-03-07 13:00:00').
    - stable_window_size: The length of the stable period before an anomaly.
    - anomaly_window_size: The maximum duration of the anomaly ramp-up/down phase.
    - recovery_window_size: The length of the stable period after an anomaly.
    - stable_std_thresh_factor: A factor to define stability based on standard deviation.
    - ramp_up_thresh_factor: A factor to define the suddenness of a change.
    - noise_reduction_w: A threshold for filtering minor anomalies within a single metric.
    - cluster_minutes_k: A time window for grouping anomalies into clusters.
    - max_clusters_n: The maximum number of clusters for a metric to be considered significant.
    - final_noise_x: A final threshold for filtering minor anomalies across all metrics.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity.
      Returns an empty list if no significant anomalies are found.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    tz = pytz.timezone('Asia/Shanghai')

    # Define time windows
    try:
        start_dt = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_dt = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS'.")

    extended_start_dt = start_dt - timedelta(minutes=15)
    extended_end_dt = end_dt + timedelta(minutes=15)

    # Load data
    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Metric file not found at {metric_path}")
    
    df = pd.read_csv(metric_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(tz)
    
    df = df[(df['datetime'] >= extended_start_dt) & (df['datetime'] <= extended_end_dt)]

    target_kpis = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc',
        'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]
    drop_anomaly_kpis = {'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem'}

    df_filtered = df[df['kpi_name'].isin(target_kpis)].copy()
    if df_filtered.empty:
        return []

    all_kpi_anomalies = []

    for kpi in df_filtered['kpi_name'].unique():
        kpi_df = df_filtered[df_filtered['kpi_name'] == kpi].copy()
        
        # Normalize data
        min_val, max_val = kpi_df['value'].min(), kpi_df['value'].max()
        if max_val == min_val:
            continue
        kpi_df.loc[:, 'norm_value'] = 100 * (kpi_df['value'] - min_val) / (max_val - min_val)

        # Pivot to get time series for each component
        pivoted = kpi_df.pivot_table(index='datetime', columns='cmdb_id', values='norm_value')
        pivoted = pivoted.resample('1T').mean().ffill().bfill()
        
        if pivoted.empty:
            continue

        # Sort components by peak value in diagnosis window
        diag_window_pivoted = pivoted[(pivoted.index >= start_dt) & (pivoted.index <= end_dt)]
        if diag_window_pivoted.empty:
            continue
            
        peak_values = diag_window_pivoted.max() if kpi not in drop_anomaly_kpis else diag_window_pivoted.min()
        sorted_components = peak_values.sort_values(ascending=False).index

        metric_anomalies = []
        
        for component in sorted_components:
            series = pivoted[component].dropna()
            if len(series) < (stable_window_size + anomaly_window_size + recovery_window_size):
                continue

            is_drop = kpi in drop_anomaly_kpis
            
            # --- Internal function to find events ---
            def find_events_in_series_fa56326f(ts, diag_start, diag_end):
                events = []
                covered_indices = set()
                
                for _ in range(2): # Find at most 2 events
                    search_series = ts[(ts.index >= diag_start) & (ts.index <= diag_end)]
                    
                    # Find next peak/valley not already covered
                    peak_idx, peak_val = None, None
                    sorted_ts = search_series.sort_values(ascending=not is_drop)
                    
                    for idx, val in sorted_ts.items():
                        if idx not in covered_indices:
                            peak_idx, peak_val = idx, val
                            break
                    
                    if peak_idx is None:
                        break

                    # Search backwards for anomaly start
                    start_idx = None
                    for i in range(ts.index.get_loc(peak_idx), ts.index.get_loc(peak_idx) - anomaly_window_size, -1):
                        if i <= 0: break
                        
                        # Check for sharp change
                        current_val, prev_val = ts.iloc[i], ts.iloc[i-1]
                        if abs(current_val - prev_val) > 0.1: # Basic check for any change
                            start_idx = ts.index[i]
                            break
                    
                    if start_idx is None or start_idx < diag_start:
                        covered_indices.add(peak_idx)
                        continue

                    # Check for stable phase before start
                    stable_end_loc = ts.index.get_loc(start_idx)
                    stable_start_loc = stable_end_loc - stable_window_size
                    if stable_start_loc < 0:
                        covered_indices.add(peak_idx)
                        continue
                    
                    stable_series = ts.iloc[stable_start_loc:stable_end_loc]
                    stable_mean = stable_series.mean()
                    stable_std = stable_series.std()

                    # Check for recovery phase after peak
                    recovery_start_loc = ts.index.get_loc(peak_idx) + 1
                    recovery_end_loc = recovery_start_loc + recovery_window_size
                    if recovery_end_loc > len(ts):
                        covered_indices.add(peak_idx)
                        continue
                    
                    recovery_series = ts.iloc[recovery_start_loc:recovery_end_loc]
                    recovery_std = recovery_series.std()

                    # Validate anomaly pattern
                    is_stable = stable_std < (abs(stable_mean) * stable_std_thresh_factor + 1) # Add 1 to avoid issues with mean=0
                    is_recovered = recovery_std < (abs(recovery_series.mean()) * stable_std_thresh_factor + 1)
                    
                    ramp_up_val = abs(ts.loc[start_idx] - ts.iloc[stable_end_loc - 1])
                    is_sudden = ramp_up_val > (ramp_up_thresh_factor * stable_std + 0.1) # Add 0.1 to avoid std=0 issue

                    if is_stable and is_recovered and is_sudden:
                        delta = abs(peak_val - stable_mean)
                        events.append({
                            'timestamp': start_idx,
                            'cmdb_id': component,
                            'kpi': kpi,
                            'delta': delta
                        })
                        
                        # Mark window as covered
                        event_indices = ts.loc[start_idx:peak_idx].index
                        for idx in event_indices:
                            covered_indices.add(idx)
                return events
            # --- End internal function ---

            found_events = find_events_in_series_fa56326f(series, start_dt, end_dt)
            metric_anomalies.extend(found_events)

        if not metric_anomalies:
            continue

        # Denoise per metric
        max_delta = max(e['delta'] for e in metric_anomalies)
        denoised_anomalies = [e for e in metric_anomalies if e['delta'] >= noise_reduction_w * max_delta]

        # Cluster
        if not denoised_anomalies:
            continue
        
        denoised_anomalies.sort(key=lambda x: x['timestamp'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for i in range(1, len(denoised_anomalies)):
                time_diff = (denoised_anomalies[i]['timestamp'] - current_cluster[-1]['timestamp']).total_seconds() / 60
                if time_diff <= cluster_minutes_k:
                    current_cluster.append(denoised_anomalies[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [denoised_anomalies[i]]
            clusters.append(current_cluster)

        if 1 <= len(clusters) <= max_clusters_n:
            clusters.sort(key=lambda c: max(e['delta'] for e in c), reverse=True)
            # Keep anomalies from top 2 clusters
            for cluster in clusters[:2]:
                all_kpi_anomalies.extend(cluster)

    if not all_kpi_anomalies:
        return []

    # Final noise reduction
    overall_max_delta = max(e['delta'] for e in all_kpi_anomalies)
    final_anomalies = [e for e in all_kpi_anomalies if e['delta'] >= final_noise_x * overall_max_delta]

    # Format output
    anomaly_events = []
    for event in final_anomalies:
        anomaly_type = "low" if event['kpi'] in drop_anomaly_kpis else "high"
        anomaly_events.append({
            "data_source": "Metric",
            "timestamp": event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "cmdb_id": event['cmdb_id'],
            "description": f"'{event['kpi']}' is abnormally {anomaly_type}, delta={event['delta']:.2f}"
        })

    # Sort by delta and remove duplicates
    unique_events = [dict(t) for t in {tuple(d.items()) for d in anomaly_events}]
    unique_events.sort(key=lambda x: float(x['description'].split('=')[-1]), reverse=True)
    
    return unique_events[:10]
if __name__ == '__main__':
    date = "2021_03_07"
    start_time = "2021-03-07 12:30:00"
    end_time = "2021-03-07 13:00:00"
    
    anomaly_events = detect_metric_anomalies_from_files_51fa07e4(date, start_time, end_time)
    
    print(anomaly_events)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def detect_metric_anomalies_by_shape_8025a14a(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_07'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-07 12:30:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-07 13:00:00'"],
    stable_window_size: Annotated[int, "Number of data points for the stable phase."] = 5,
    recovery_window_size: Annotated[int, "Number of data points for the recovery phase."] = 5,
    anomaly_window_min_len: Annotated[int, "Minimum number of data points for the anomaly phase."] = 1,
    anomaly_window_max_len: Annotated[int, "Maximum number of data points for the anomaly phase."] = 8,
    stable_std_thresh: Annotated[float, "Standard deviation threshold for a phase to be considered stable (on a 0-100 normalized scale)."] = 6.0,
    ramp_thresh: Annotated[float, "Threshold for the sudden change at the beginning of an anomaly (on a 0-100 normalized scale)."] = 10.0,
    noise_reduction_w: Annotated[float, "Factor for metric-level noise reduction (w * delta_max)."] = 0.6,
    clustering_time_k_minutes: Annotated[int, "Time window in minutes for clustering anomalies."] = 3,
    max_clusters_nc: Annotated[int, "Maximum number of clusters for a metric to be considered a potential root cause."] = 3,
    final_filter_x: Annotated[float, "Factor for final noise reduction across all metrics (x * global_max_delta)."] = 0.2
) -> List[Dict[str, Any]]:
    """
    Analyzes metric data to detect anomalies based on a shape-based pattern (stable-anomaly-recovery).

    This tool processes time-series data from 'metric_container.csv' for a given date and time window. It normalizes
    the data for specified KPIs, identifies components with the most extreme values, and then searches for anomaly
    patterns. An anomaly is defined by a stable period, followed by a sharp spike or dip, and then a return to a
    stable state. The tool filters out noise at both the individual metric level and globally, and uses clustering
    to assess the significance of a metric's anomalies. The final output is a sorted list of the most severe
    anomaly events, which can be used as hypotheses for root cause analysis.

    Parameters:
    - date_str: Date string for the data directory, e.g., '2021_03_07'.
    - start_time_str: Start of the diagnosis window, e.g., '2021-03-07 12:30:00'.
    - end_time_str: End of the diagnosis window, e.g., '2021-03-07 13:00:00'.
    - stable_window_size: Number of data points defining the stable phase before an anomaly.
    - recovery_window_size: Number of data points defining the recovery phase after an anomaly.
    - anomaly_window_min_len: Minimum duration (in data points) of an anomaly phase.
    - anomaly_window_max_len: Maximum duration (in data points) of an anomaly phase.
    - stable_std_thresh: Maximum standard deviation for a period to be considered stable.
    - ramp_thresh: Minimum change required to be considered a sudden spike/dip.
    - noise_reduction_w: Filtering threshold for anomalies within the same metric.
    - clustering_time_k_minutes: Time window for grouping anomalies into clusters.
    - max_clusters_nc: If a metric has more than this many anomaly clusters, it's considered noisy and ignored.
    - final_filter_x: Final global filtering threshold for all detected anomalies.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event. Returns an empty list if no significant anomalies are found.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    
    KPI_DIRECTIONS = {
        'OSLinux-CPU_CPU_CPUCpuUtil': 'up',
        'OSLinux-CPU_CPU_CPUUserTime': 'up',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc': 'up',
        'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc': 'up',
        'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem': 'down',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT': 'up',
        'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum': 'up',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite': 'up',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity': 'up',
        'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity': 'up',
        'JVM-Operating System_7779_JVM_JVM_CPULoad': 'up',
        'JVM-Operating System_7778_JVM_JVM_CPULoad': 'up',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed': 'up',
        'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed': 'up',
    }

    tz = pytz.timezone('Asia/Shanghai')
    start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    extended_start_time = start_time - timedelta(minutes=15)
    extended_end_time = end_time + timedelta(minutes=15)

    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        return []
    
    df = pd.read_csv(metric_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)
    df_extended = df[(df['datetime'] >= extended_start_time) & (df['datetime'] <= extended_end_time)].copy()

    all_final_anomalies = []

    for kpi, direction in KPI_DIRECTIONS.items():
        df_kpi = df_extended[df_extended['kpi_name'] == kpi].copy()
        if df_kpi.empty:
            continue

        min_val, max_val = df_kpi['value'].min(), df_kpi['value'].max()
        if max_val == min_val:
            continue
        df_kpi['norm_val'] = 100 * (df_kpi['value'] - min_val) / (max_val - min_val)

        df_kpi_diag = df_kpi[(df_kpi['datetime'] >= start_time) & (df_kpi['datetime'] <= end_time)]
        if df_kpi_diag.empty:
            continue

        if direction == 'up':
            component_extrema = df_kpi_diag.loc[df_kpi_diag.groupby('cmdb_id')['norm_val'].idxmax()]
            sorted_components = component_extrema.sort_values('norm_val', ascending=False)['cmdb_id'].tolist()
        else: # 'down'
            component_extrema = df_kpi_diag.loc[df_kpi_diag.groupby('cmdb_id')['norm_val'].idxmin()]
            sorted_components = component_extrema.sort_values('norm_val', ascending=True)['cmdb_id'].tolist()

        kpi_events = []
        for comp in sorted_components:
            comp_series = df_kpi[df_kpi['cmdb_id'] == comp].sort_values('datetime').reset_index(drop=True)
            if len(comp_series) < stable_window_size + anomaly_window_min_len + recovery_window_size:
                continue

            diag_indices = comp_series[(comp_series['datetime'] >= start_time) & (comp_series['datetime'] <= end_time)].index
            if len(diag_indices) == 0:
                continue

            if direction == 'up':
                sorted_extrema = comp_series.loc[diag_indices].sort_values('norm_val', ascending=False)
            else:
                sorted_extrema = comp_series.loc[diag_indices].sort_values('norm_val', ascending=True)

            processed_indices = set()
            found_events_for_comp = 0
            for extremum_idx, _ in sorted_extrema.iterrows():
                if extremum_idx in processed_indices:
                    continue
                if found_events_for_comp >= 2:
                    break

                for L in range(anomaly_window_min_len, anomaly_window_max_len + 1):
                    for offset in range(L):
                        anomaly_start_idx = extremum_idx - offset
                        anomaly_end_idx = anomaly_start_idx + L - 1

                        if not (anomaly_start_idx in diag_indices and anomaly_end_idx in diag_indices):
                            continue
                        
                        stable_start_idx = anomaly_start_idx - stable_window_size
                        recovery_end_idx = anomaly_end_idx + recovery_window_size

                        if stable_start_idx < 0 or recovery_end_idx >= len(comp_series):
                            continue

                        stable_vals = comp_series.loc[stable_start_idx:anomaly_start_idx - 1, 'norm_val']
                        anomaly_vals = comp_series.loc[anomaly_start_idx:anomaly_end_idx, 'norm_val']
                        recovery_vals = comp_series.loc[anomaly_end_idx + 1:recovery_end_idx, 'norm_val']

                        if len(stable_vals) < stable_window_size or len(recovery_vals) < recovery_window_size:
                            continue

                        stable_std = stable_vals.std()
                        recovery_std = recovery_vals.std()

                        if stable_std < stable_std_thresh and recovery_std < stable_std_thresh:
                            ramp_val = comp_series.loc[anomaly_start_idx, 'norm_val'] - comp_series.loc[anomaly_start_idx - 1, 'norm_val']
                            
                            is_valid_ramp = (direction == 'up' and ramp_val > ramp_thresh) or \
                                            (direction == 'down' and ramp_val < -ramp_thresh)
                            
                            if is_valid_ramp:
                                stable_avg = stable_vals.mean()
                                if direction == 'up':
                                    delta = anomaly_vals.max() - stable_avg
                                    desc_qualifier = "abnormally high"
                                else:
                                    delta = stable_avg - anomaly_vals.min()
                                    desc_qualifier = "abnormally low"

                                if delta > 0:
                                    event = {
                                        'data_source': 'Metric',
                                        'timestamp': comp_series.loc[anomaly_start_idx, 'datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                                        'cmdb_id': comp,
                                        'description': f"'{kpi}' is {desc_qualifier}, delta={delta:.2f}",
                                        'delta': delta,
                                        'start_time_obj': comp_series.loc[anomaly_start_idx, 'datetime']
                                    }
                                    kpi_events.append(event)
                                    processed_indices.update(range(anomaly_start_idx, anomaly_end_idx + 1))
                                    found_events_for_comp += 1
                                    break 
                    if extremum_idx in processed_indices:
                        break
        
        if not kpi_events:
            continue

        max_delta_kpi = max(event['delta'] for event in kpi_events)
        denoised_events = [e for e in kpi_events if e['delta'] >= noise_reduction_w * max_delta_kpi]

        if not denoised_events:
            continue

        denoised_events.sort(key=lambda x: x['start_time_obj'])
        clusters = []
        if denoised_events:
            current_cluster = [denoised_events[0]]
            clusters.append(current_cluster)
            for event in denoised_events[1:]:
                if (event['start_time_obj'] - current_cluster[0]['start_time_obj']) <= timedelta(minutes=clustering_time_k_minutes):
                    current_cluster.append(event)
                else:
                    current_cluster = [event]
                    clusters.append(current_cluster)
        
        if 0 < len(clusters) <= max_clusters_nc:
            for event in denoised_events:
                del event['start_time_obj'] 
                all_final_anomalies.append(event)

    if not all_final_anomalies:
        return []

    max_delta_global = max(event['delta'] for event in all_final_anomalies)
    final_anomalies = [e for e in all_final_anomalies if e['delta'] >= final_filter_x * max_delta_global]
    
    for event in final_anomalies:
        del event['delta']

    final_anomalies = [dict(t) for t in {tuple(d.items()) for d in final_anomalies}]
    
    return final_anomalies
if __name__ == '__main__':
    anomaly_events = detect_metric_anomalies_by_shape_8025a14a(
        date_str='2021_03_07',
        start_time_str='2021-03-07 12:30:00',
        end_time_str='2021-03-07 13:00:00',
        stable_std_thresh=8.0,      # Loosened from 6.0
        ramp_thresh=8.0,            # Loosened from 10.0
        noise_reduction_w=0.4,      # Loosened from 0.6
        max_clusters_nc=4,          # Loosened from 3
        final_filter_x=0.1          # Loosened from 0.2
    )
    print(anomaly_events)




import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_by_shape_86fb5e5a(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-04 19:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-04 19:30:00'"],
    stable_stage_len: Annotated[int, "Number of data points for the stable stage."] = 5,
    anomaly_stage_min_len: Annotated[int, "Minimum number of data points for the anomaly stage."] = 1,
    anomaly_stage_max_len: Annotated[int, "Maximum number of data points for the anomaly stage."] = 8,
    recovery_stage_len: Annotated[int, "Number of data points for the recovery stage."] = 5,
    stable_std_thresh: Annotated[float, "Standard deviation threshold for a stage to be considered stable."] = 10.0,
    ramp_thresh: Annotated[float, "Threshold for the ramp-up value to be considered significant."] = 8.0,
    noise_reduction_w: Annotated[float, "Weight for metric-level noise reduction (w * delta_max)."] = 0.3,
    cluster_time_k: Annotated[int, "Time window in minutes for clustering anomalies."] = 3,
    max_clusters_nc: Annotated[int, "Maximum number of clusters allowed for a metric to be a potential root cause."] = 5,
    final_filter_x: Annotated[float, "Weight for final noise reduction (x * overall_max_delta)."] = 0.05
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Analyzes metric data from metric_container.csv to detect anomalies based on a shape-based pattern 
    (stable -> anomalous -> recovery). It normalizes metrics within an extended time window, identifies 
    significant spikes or drops for a predefined set of KPIs, clusters them by time, and filters out 
    noise to identify the top anomaly events.

    Parameters:
    - date_str: The date of the telemetry data to analyze, e.g., '2021_03_04'.
    - start_time_str: The start of the time window for anomaly diagnosis, e.g., '2021-03-04 19:00:00'.
    - end_time_str: The end of the time window for anomaly diagnosis, e.g., '2021-03-04 19:30:00'.
    - stable_stage_len: Number of data points defining the stable phase before an anomaly.
    - anomaly_stage_min_len: Minimum duration in data points for the anomalous phase.
    - anomaly_stage_max_len: Maximum duration in data points for the anomalous phase.
    - recovery_stage_len: Number of data points defining the recovery phase after an anomaly.
    - stable_std_thresh: Normalized standard deviation threshold to identify a stable phase.
    - ramp_thresh: Normalized threshold for the change from stable to anomalous phase.
    - noise_reduction_w: Factor to filter out minor anomalies within a single metric's context.
    - cluster_time_k: Time window (minutes) to group anomalies into a single incident cluster.
    - max_clusters_nc: If a metric has more than this many clusters, it's considered noisy and ignored.
    - final_filter_x: Factor to filter out minor anomalies across all metrics.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    tz = pytz.timezone('Asia/Shanghai')

    kpi_anomaly_direction = {
        'OSLinux-CPU_CPU_CPUCpuUtil': 'spike', 'OSLinux-CPU_CPU_CPUUserTime': 'spike',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc': 'spike', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc': 'spike',
        'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem': 'drop',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT': 'spike', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum': 'spike',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite': 'spike',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity': 'spike', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity': 'spike',
        'JVM-Operating System_7779_JVM_JVM_CPULoad': 'spike', 'JVM-Operating System_7778_JVM_JVM_CPULoad': 'spike',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed': 'spike', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed': 'spike'
    }
    target_kpis = list(kpi_anomaly_direction.keys())

    try:
        start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
        extended_start_time = start_time - timedelta(minutes=15)
        extended_end_time = end_time + timedelta(minutes=15)
    except ValueError as e:
        raise ValueError(f"Invalid time format: {e}. Please use 'YYYY-mm-dd HH:MM:SS'.")

    metric_file_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_file_path):
        raise FileNotFoundError(f"Metric file not found at {metric_file_path}")

    df = pd.read_csv(metric_file_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(tz)
    
    df_extended = df[(df['datetime'] >= extended_start_time) & (df['datetime'] <= extended_end_time)]
    df_extended = df_extended[df_extended['kpi_name'].isin(target_kpis)]

    if df_extended.empty:
        return []

    all_anomaly_events = []

    for kpi_name in target_kpis:
        anomaly_type = kpi_anomaly_direction[kpi_name]
        df_kpi = df_extended[df_extended['kpi_name'] == kpi_name]
        if df_kpi.empty:
            continue

        df_pivot = df_kpi.pivot_table(index='datetime', columns='cmdb_id', values='value').resample('1min').mean()
        df_pivot = df_pivot.ffill().bfill()

        if df_pivot.empty:
            continue

        global_min = df_pivot.min().min()
        global_max = df_pivot.max().max()

        if global_max == global_min:
            continue
        
        df_norm = 100 * (df_pivot - global_min) / (global_max - global_min)

        component_peaks = df_norm[start_time:end_time].max().sort_values(ascending=False)
        sorted_components = component_peaks.index

        kpi_events = []
        for cmdb_id in sorted_components:
            series = df_norm[cmdb_id].dropna()
            if series.empty:
                continue
            
            detected_event_ranges = []
            
            for i in range(len(series)):
                if not (start_time <= series.index[i] <= end_time):
                    continue

                is_overlapping = any(start <= series.index[i] < end for start, end in detected_event_ranges)
                if is_overlapping:
                    continue

                stable_start_idx = i - stable_stage_len
                if stable_start_idx < 0: continue
                stable_window = series.iloc[stable_start_idx:i]
                if len(stable_window) < stable_stage_len: continue
                
                stable_avg = stable_window.mean()
                stable_std = stable_window.std()
                if np.isnan(stable_std) or stable_std >= stable_std_thresh:
                    continue

                for anom_len in range(anomaly_stage_min_len, anomaly_stage_max_len + 1):
                    anom_end_idx = i + anom_len
                    if anom_end_idx > len(series) or series.index[anom_end_idx - 1] > end_time:
                        break

                    anomaly_window = series.iloc[i:anom_end_idx]
                    
                    recovery_start_idx = anom_end_idx
                    recovery_end_idx = recovery_start_idx + recovery_stage_len
                    if recovery_end_idx > len(series): continue
                    
                    recovery_window = series.iloc[recovery_start_idx:recovery_end_idx]
                    if len(recovery_window) < recovery_stage_len: continue
                    
                    recovery_std = recovery_window.std()
                    if np.isnan(recovery_std) or recovery_std >= stable_std_thresh:
                        continue

                    delta = 0
                    if anomaly_type == 'spike':
                        peak_val = anomaly_window.max()
                        if peak_val - stable_avg > ramp_thresh:
                            delta = peak_val - stable_avg
                    else: # drop
                        valley_val = anomaly_window.min()
                        if stable_avg - valley_val > ramp_thresh:
                            delta = stable_avg - valley_val
                    
                    if delta > 0:
                        event_time = series.index[i]
                        kpi_events.append({
                            'timestamp_dt': event_time,
                            'cmdb_id': cmdb_id,
                            'kpi_name': kpi_name,
                            'delta': delta,
                            'anomaly_type': anomaly_type
                        })
                        detected_event_ranges.append((event_time, series.index[anom_end_idx - 1] + timedelta(seconds=1)))
                        break
            
            if len(kpi_events) >= 2:
                break

        if not kpi_events:
            continue

        max_delta_kpi = max(event['delta'] for event in kpi_events)
        denoised_events = [e for e in kpi_events if e['delta'] >= noise_reduction_w * max_delta_kpi]

        if not denoised_events:
            continue

        denoised_events.sort(key=lambda x: x['timestamp_dt'])
        clusters = []
        if denoised_events:
            current_cluster = [denoised_events[0]]
            clusters.append(current_cluster)
            for i in range(1, len(denoised_events)):
                time_diff = (denoised_events[i]['timestamp_dt'] - current_cluster[0]['timestamp_dt']).total_seconds() / 60
                if time_diff <= cluster_time_k:
                    current_cluster.append(denoised_events[i])
                else:
                    current_cluster = [denoised_events[i]]
                    clusters.append(current_cluster)
        
        if 1 <= len(clusters) <= max_clusters_nc:
            all_anomaly_events.extend(denoised_events)

    if not all_anomaly_events:
        return []

    max_overall_delta = max(event['delta'] for event in all_anomaly_events)
    final_events = [e for e in all_anomaly_events if e['delta'] >= final_filter_x * max_overall_delta]

    final_events.sort(key=lambda x: x['delta'], reverse=True)
    
    anomaly_events_output = []
    for event in final_events[:10]: # Increased output limit
        description_verb = "abnormally high" if event['anomaly_type'] == 'spike' else "abnormally low"
        anomaly_events_output.append({
            "data_source": "Metric",
            "timestamp": event['timestamp_dt'].strftime('%Y-%m-%d %H:%M:%S'),
            "cmdb_id": event['cmdb_id'],
            "description": f"Metric '{event['kpi_name']}' on component '{event['cmdb_id']}' is {description_verb}, with a normalized delta of {event['delta']:.2f}."
        })

    return anomaly_events_output
if __name__ == '__main__':
    date_param = "2021_03_04"
    start_time_param = "2021-03-04 19:00:00"
    end_time_param = "2021-03-04 19:30:00"
    
    anomaly_events = find_metric_anomalies_by_shape_86fb5e5a(
        date_str=date_param,
        start_time_str=start_time_param,
        end_time_str=end_time_param
    )
    
    print(anomaly_events)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_by_shape_5b5af2fd(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_06'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-06 23:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-06 23:30:00'"],
    stable_std_thresh: Annotated[float, "Standard deviation threshold for defining a stable phase."] = 15.0,
    ramp_thresh: Annotated[float, "Threshold for the ramp-up/down value change from stable to anomaly phase."] = 5.0,
    noise_reduction_w: Annotated[float, "Weight for metric-level noise reduction. Events with delta < w * max_delta are removed."] = 0.1,
    max_clusters_nc: Annotated[int, "Maximum number of anomaly clusters allowed for a metric to be considered a potential root cause."] = 7,
    final_filter_x: Annotated[float, "Weight for final noise filtering. Events with delta < x * overall_max_delta are removed."] = 0.01,
    cluster_minutes_k: Annotated[int, "Time window in minutes for clustering anomalies."] = 3,
    stable_window_size: Annotated[int, "Number of data points for the stable phase window."] = 5,
    recovery_window_size: Annotated[int, "Number of data points for the recovery phase window."] = 5,
    anomaly_window_min: Annotated[int, "Minimum number of data points for the anomaly phase window."] = 1,
    anomaly_window_max: Annotated[int, "Maximum number of data points for the anomaly phase window."] = 8
) -> List[Dict[str, Any]]:
    """
    This tool detects metric anomalies by analyzing the shape of time-series data. It identifies patterns consisting of a stable phase, a sudden spike or drop, and a recovery phase.

    The process involves:
    1. Normalizing metric data to a 0-100 scale within an extended time window.
    2. For each component and metric, searching for the specific anomaly shape (stable-anomaly-recovery).
    3. Calculating a 'delta' value (peak/valley - stable average) for each found anomaly.
    4. Applying a metric-level noise reduction based on the maximum delta for that metric.
    5. Clustering anomalies by time and filtering out metrics with too many scattered clusters.
    6. Performing a final global filtering based on the overall maximum delta across all metrics.
    7. Returning a sorted list of the most significant anomaly events.

    Parameters:
    - date_str: The date of the telemetry data to analyze.
    - start_time_str: The start of the diagnosis time window.
    - end_time_str: The end of the diagnosis time window.
    - stable_std_thresh: Threshold for standard deviation to consider a phase stable.
    - ramp_thresh: Minimum change required to be considered a ramp up/down.
    - noise_reduction_w: Factor to filter out minor anomalies within a single metric.
    - max_clusters_nc: If a metric has more than this many anomaly clusters, it's considered noisy and discarded.
    - final_filter_x: Factor to filter out minor anomalies across all metrics.
    - cluster_minutes_k: Time window for grouping anomalies into clusters.
    - stable_window_size: Size of the stable phase before the anomaly.
    - recovery_window_size: Size of the recovery phase after the anomaly.
    - anomaly_window_min: Minimum duration of the anomaly phase.
    - anomaly_window_max: Maximum duration of the anomaly phase.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    
    TARGET_KPIS = {
        'OSLinux-CPU_CPU_CPUCpuUtil': 'spike', 'OSLinux-CPU_CPU_CPUUserTime': 'spike',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc': 'spike', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc': 'spike',
        'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem': 'drop',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT': 'spike', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum': 'spike',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite': 'spike',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity': 'spike', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity': 'spike',
        'JVM-Operating System_7779_JVM_JVM_CPULoad': 'spike', 'JVM-Operating System_7778_JVM_JVM_CPULoad': 'spike',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed': 'spike', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed': 'spike'
    }

    tz = pytz.timezone('Asia/Shanghai')
    
    try:
        start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-mm-dd HH:MM:SS'")

    diag_start_ts = int(start_time.timestamp())
    diag_end_ts = int(end_time.timestamp())

    extended_start_time = start_time - timedelta(minutes=15)
    extended_end_time = end_time + timedelta(minutes=15)
    extended_start_ts = int(extended_start_time.timestamp())
    extended_end_ts = int(extended_end_time.timestamp())

    metric_file = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_file):
        return []

    df = pd.read_csv(metric_file)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df = df[(df['timestamp'] >= extended_start_ts) & (df['timestamp'] <= extended_end_ts)]
    df = df[df['kpi_name'].isin(TARGET_KPIS.keys())]

    if df.empty:
        return []

    all_anomaly_events = []

    for kpi, anomaly_type in TARGET_KPIS.items():
        kpi_df = df[df['kpi_name'] == kpi].copy()
        if kpi_df.empty:
            continue

        min_val, max_val = kpi_df['value'].min(), kpi_df['value'].max()
        if max_val == min_val:
            continue
        
        kpi_df['normalized_value'] = 100 * (kpi_df['value'] - min_val) / (max_val - min_val)

        components = kpi_df['cmdb_id'].unique()
        component_scores = {}
        for comp in components:
            comp_df = kpi_df[(kpi_df['cmdb_id'] == comp) & (kpi_df['timestamp'] >= diag_start_ts) & (kpi_df['timestamp'] <= diag_end_ts)]
            if not comp_df.empty:
                if anomaly_type == 'spike':
                    component_scores[comp] = comp_df['normalized_value'].max()
                else: # drop
                    component_scores[comp] = comp_df['normalized_value'].min()
        
        sorted_components = sorted(component_scores.keys(), key=lambda c: component_scores[c], reverse=(anomaly_type == 'spike'))

        kpi_anomalies = []
        for comp in sorted_components:
            comp_series = kpi_df[kpi_df['cmdb_id'] == comp].sort_values('timestamp').reset_index()
            if len(comp_series) < stable_window_size + anomaly_window_min + recovery_window_size:
                continue

            used_indices = set()
            comp_events_count = 0

            for i in range(stable_window_size, len(comp_series) - (anomaly_window_max + recovery_window_size)):
                if i in used_indices or comp_events_count >= 2:
                    break

                stable_window = comp_series.iloc[i - stable_window_size : i]
                std_stable = stable_window['normalized_value'].std()

                if std_stable < stable_std_thresh:
                    ramp_start_val = comp_series.iloc[i]['normalized_value']
                    stable_end_val = comp_series.iloc[i-1]['normalized_value']
                    
                    is_ramp = (anomaly_type == 'spike' and ramp_start_val - stable_end_val > ramp_thresh) or \
                              (anomaly_type == 'drop' and stable_end_val - ramp_start_val > ramp_thresh)

                    if is_ramp:
                        for anomaly_len in range(anomaly_window_min, anomaly_window_max + 1):
                            anomaly_end_idx = i + anomaly_len
                            if anomaly_end_idx + recovery_window_size > len(comp_series):
                                continue

                            anomaly_window = comp_series.iloc[i:anomaly_end_idx]
                            if not (anomaly_window['timestamp'].min() >= diag_start_ts and anomaly_window['timestamp'].max() <= diag_end_ts):
                                continue

                            recovery_window = comp_series.iloc[anomaly_end_idx : anomaly_end_idx + recovery_window_size]
                            if len(recovery_window) < recovery_window_size:
                                continue
                            
                            std_recovery = recovery_window['normalized_value'].std()

                            if std_recovery < stable_std_thresh:
                                avg_stable = stable_window['normalized_value'].mean()
                                if anomaly_type == 'spike':
                                    peak_val = anomaly_window['normalized_value'].max()
                                    delta = peak_val - avg_stable
                                else: # drop
                                    valley_val = anomaly_window['normalized_value'].min()
                                    delta = avg_stable - valley_val
                                
                                if delta > 0:
                                    anomaly_start_ts_event = comp_series.iloc[i]['timestamp']
                                    dt_object = datetime.fromtimestamp(anomaly_start_ts_event, tz=pytz.utc).astimezone(tz)
                                    
                                    kpi_anomalies.append({
                                        'data_source': 'Metric',
                                        'timestamp': dt_object.strftime('%Y-%m-%d %H:%M:%S'),
                                        'cmdb_id': comp,
                                        'description': f"Metric '{kpi}' on component '{comp}' showed an anomalous {anomaly_type} with delta={delta:.2f}.",
                                        'kpi_name': kpi,
                                        'delta': delta,
                                        'start_ts': anomaly_start_ts_event
                                    })
                                    comp_events_count += 1
                                    for used_idx in range(i, anomaly_end_idx + recovery_window_size):
                                        used_indices.add(used_idx)
                                    break # Found anomaly for this start point, move to next
        
        if not kpi_anomalies:
            continue

        max_delta_kpi = max(event['delta'] for event in kpi_anomalies)
        denoised_anomalies = [event for event in kpi_anomalies if event['delta'] >= noise_reduction_w * max_delta_kpi]

        if not denoised_anomalies:
            continue
            
        denoised_anomalies.sort(key=lambda x: x['start_ts'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for i in range(1, len(denoised_anomalies)):
                if (denoised_anomalies[i]['start_ts'] - current_cluster[0]['start_ts']) / 60 <= cluster_minutes_k:
                    current_cluster.append(denoised_anomalies[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [denoised_anomalies[i]]
            clusters.append(current_cluster)

        if 1 <= len(clusters) <= max_clusters_nc:
            denoised_anomalies.sort(key=lambda x: x['delta'], reverse=True)
            top_deltas = sorted([event['delta'] for event in denoised_anomalies], reverse=True)[:2]
            
            root_cause_clusters = []
            for cluster in clusters:
                for event in cluster:
                    if event['delta'] in top_deltas:
                        root_cause_clusters.append(cluster)
                        break
            
            for cluster in root_cause_clusters:
                all_anomaly_events.extend(cluster)

    if not all_anomaly_events:
        return []

    # Remove duplicates
    unique_events = []
    seen = set()
    for event in all_anomaly_events:
        identifier = (event['timestamp'], event['cmdb_id'], event['kpi_name'])
        if identifier not in seen:
            unique_events.append(event)
            seen.add(identifier)
    
    if not unique_events:
        return []

    overall_max_delta = max(event['delta'] for event in unique_events)
    final_anomalies = [event for event in unique_events if event['delta'] >= final_filter_x * overall_max_delta]
    
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)

    # Clean up for final output
    for event in final_anomalies:
        del event['kpi_name']
        del event['delta']
        del event['start_ts']

    return final_anomalies
if __name__ == '__main__':
    date_str = "2021_03_06"
    start_time_str = "2021-03-06 23:00:00"
    end_time_str = "2021-03-06 23:30:00"
    
    anomaly_events = find_metric_anomalies_by_shape_5b5af2fd(
        date_str=date_str,
        start_time_str=start_time_str,
        end_time_str=end_time_str
    )
    
    print(anomaly_events)
    if anomaly_events:
        pd.DataFrame(anomaly_events).to_csv("anomaly_events.csv", index=False)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def analyze_app_kpi_anomalies_b0d84adb(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_06'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-06 23:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-06 23:30:00'"],
    mrt_abs_thresh: Annotated[float, "Absolute threshold for mean response time (ms)."] = 500.0,
    mrt_rel_increase_thresh: Annotated[float, "Relative increase threshold for mean response time (e.g., 0.5 for 50%)."] = 0.5,
    sr_drop_thresh_pp: Annotated[float, "Success rate drop threshold in percentage points."] = 5.0,
    cnt_rel_drop_thresh: Annotated[float, "Relative drop threshold for request count (e.g., 0.3 for 30%)."] = 0.3,
    rr_rel_drop_thresh: Annotated[float, "Relative drop threshold for response rate (e.g., 0.3 for 30%)."] = 0.3,
    baseline_hours: Annotated[int, "Number of hours before the start time to use for baseline calculation."] = 6
) -> List[Dict[str, Any]]:
    """
    Analyzes application-level KPIs (rr, sr, cnt, mrt) from metric_app.csv to detect anomalies against a dynamic baseline.

    This tool performs the following steps:
    1.  Reads metric_app.csv for the specified date.
    2.  Establishes a baseline for each service ('tc') by averaging KPIs over a preceding time window (default 6 hours).
    3.  Scans the diagnosis window for data points that deviate significantly from the baseline according to specified rules.
    4.  Detects anomalies such as high response times, success rate drops, and request count/rate drops.
    5.  Produces a list of structured anomaly events. The service name ('tc') is used as the identifier in place of 'cmdb_id'.

    Parameters:
    - date_str: The date of the telemetry data.
    - start_time_str: The start of the diagnosis time window.
    - end_time_str: The end of the diagnosis time window.
    - mrt_abs_thresh: Absolute MRT threshold in ms.
    - mrt_rel_increase_thresh: Relative MRT increase threshold (e.g., 0.5 for 50%).
    - sr_drop_thresh_pp: Success rate drop threshold in percentage points.
    - cnt_rel_drop_thresh: Relative drop threshold for request count.
    - rr_rel_drop_thresh: Relative drop threshold for response rate.
    - baseline_hours: Duration in hours for calculating the baseline.

    Returns:
    - A list of dictionaries, each representing a detected application KPI anomaly in the standard format.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    tz = pytz.timezone('Asia/Shanghai')
    app_metric_file = os.path.join(data_source_path, date_str, 'metric', 'metric_app.csv')

    if not os.path.exists(app_metric_file):
        return []

    try:
        analysis_start_dt = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        analysis_end_dt = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-mm-dd HH:MM:SS'")

    analysis_start_ts = int(analysis_start_dt.timestamp())
    analysis_end_ts = int(analysis_end_dt.timestamp())

    baseline_start_dt = analysis_start_dt - timedelta(hours=baseline_hours)
    baseline_start_ts = int(baseline_start_dt.timestamp())

    df = pd.read_csv(app_metric_file)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    baseline_df = df[(df['timestamp'] >= baseline_start_ts) & (df['timestamp'] < analysis_start_ts)]
    analysis_df = df[(df['timestamp'] >= analysis_start_ts) & (df['timestamp'] <= analysis_end_ts)]

    if analysis_df.empty:
        return []

    if baseline_df.empty:
        baselines = {}
    else:
        baseline_agg = baseline_df.groupby('tc')[['rr', 'sr', 'cnt', 'mrt']].mean()
        baselines = baseline_agg.to_dict('index')

    detailed_events = []
    for _, row in analysis_df.iterrows():
        service_tc = row['tc']
        if service_tc not in baselines:
            continue

        baseline_metrics = baselines[service_tc]
        current_ts = row['timestamp']
        dt_object = datetime.fromtimestamp(current_ts, tz=pytz.utc).astimezone(tz)
        
        # Check MRT
        mrt_baseline = baseline_metrics.get('mrt', 0)
        current_mrt = row['mrt']
        if current_mrt > mrt_abs_thresh or (mrt_baseline > 0 and current_mrt > mrt_baseline * (1 + mrt_rel_increase_thresh)):
            severity = (current_mrt - mrt_baseline) / mrt_baseline if mrt_baseline > 0 else float('inf')
            desc = f"Service '{service_tc}' MRT spiked to {current_mrt:.2f}ms (baseline: {mrt_baseline:.2f}ms)."
            detailed_events.append({'timestamp': dt_object, 'cmdb_id': service_tc, 'description': desc, 'severity': severity})

        # Check SR
        sr_baseline = baseline_metrics.get('sr', 100)
        current_sr = row['sr']
        if current_sr < sr_baseline - sr_drop_thresh_pp:
            severity = (sr_baseline - current_sr) / sr_baseline if sr_baseline > 0 else 0
            desc = f"Service '{service_tc}' SR dropped to {current_sr:.2f}% (baseline: {sr_baseline:.2f}%)."
            detailed_events.append({'timestamp': dt_object, 'cmdb_id': service_tc, 'description': desc, 'severity': severity})

        # Check CNT
        cnt_baseline = baseline_metrics.get('cnt', 0)
        current_cnt = row['cnt']
        if cnt_baseline > 0 and current_cnt < cnt_baseline * (1 - cnt_rel_drop_thresh):
            severity = (cnt_baseline - current_cnt) / cnt_baseline
            desc = f"Service '{service_tc}' request count dropped to {current_cnt} (baseline: {cnt_baseline:.2f})."
            detailed_events.append({'timestamp': dt_object, 'cmdb_id': service_tc, 'description': desc, 'severity': severity})

        # Check RR
        rr_baseline = baseline_metrics.get('rr', 100)
        current_rr = row['rr']
        if rr_baseline > 0 and current_rr < rr_baseline * (1 - rr_rel_drop_thresh):
            severity = (rr_baseline - current_rr) / rr_baseline
            desc = f"Service '{service_tc}' response rate dropped to {current_rr:.2f}% (baseline: {rr_baseline:.2f}%)."
            detailed_events.append({'timestamp': dt_object, 'cmdb_id': service_tc, 'description': desc, 'severity': severity})

    if not detailed_events:
        return []

    # Sort by severity and then timestamp
    detailed_events.sort(key=lambda x: (-x['severity'], x['timestamp']))

    # Format for standard output and deduplicate
    anomaly_events = []
    seen = set()
    for event in detailed_events:
        identifier = (event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), event['cmdb_id'], event['description'])
        if identifier not in seen:
            anomaly_events.append({
                "data_source": "Metric",
                "timestamp": event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "cmdb_id": event['cmdb_id'],
                "description": event['description']
            })
            seen.add(identifier)
            
    return anomaly_events
if __name__ == '__main__':
    date_str = "2021_03_06"
    start_time_str = "2021-03-06 23:00:00"
    end_time_str = "2021-03-06 23:30:00"
    
    anomaly_events = analyze_app_kpi_anomalies_b0d84adb(
        date_str=date_str,
        start_time_str=start_time_str,
        end_time_str=end_time_str
    )
    
    print(anomaly_events)
    if anomaly_events:
        pd.DataFrame(anomaly_events).to_csv("anomaly_events.csv", index=False)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_by_shape_a949f08f(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_09'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-09 16:30:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-09 17:00:00'"],
    stable_stage_len: Annotated[int, "Number of data points for the stable/recovery stage."] = 5,
    anomaly_stage_min_len: Annotated[int, "Minimum number of data points for the anomaly stage."] = 1,
    anomaly_stage_max_len: Annotated[int, "Maximum number of data points for the anomaly stage."] = 8,
    stable_std_thresh: Annotated[float, "Standard deviation threshold to identify a stable phase."] = 10.0,
    ramp_thresh: Annotated[float, "Threshold for the ramp-up/down from stable to anomaly phase."] = 7.0,
    noise_reduction_w: Annotated[float, "Weight for metric-level noise reduction. Events with delta < w * max_delta_for_metric are removed."] = 0.3,
    cluster_minutes_k: Annotated[int, "Time window in minutes to group anomalies into a cluster."] = 3,
    max_clusters_nc: Annotated[int, "Maximum number of anomaly clusters allowed for a metric to be considered a potential root cause."] = 5,
    final_filter_x: Annotated[float, "Weight for final noise reduction. Events with delta < x * max_overall_delta are removed."] = 0.05,
    top_n_events: Annotated[int, "Number of top anomaly events to return."] = 10
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Analyzes metric time-series data to detect anomalies based on their shape (stable -> anomaly -> recovery).
    It normalizes metrics, identifies significant spikes or dips, and filters out noise to report the most critical anomaly events.

    Parameters:
    - date_str: The date for which to analyze data.
    - start_time_str: The start of the diagnosis time window.
    - end_time_str: The end of the diagnosis time window.
    - stable_stage_len: Length of the stable phase before/after an anomaly.
    - anomaly_stage_min_len: Minimum duration of the anomalous phase.
    - anomaly_stage_max_len: Maximum duration of the anomalous phase.
    - stable_std_thresh: Maximum standard deviation for a phase to be considered stable.
    - ramp_thresh: Minimum change required to transition from stable to anomalous phase.
    - noise_reduction_w: Factor to filter out minor anomalies within a single metric.
    - cluster_minutes_k: Time window for clustering anomalies.
    - max_clusters_nc: Maximum number of clusters for a metric to be considered significant.
    - final_filter_x: Factor to filter out minor anomalies across all metrics.
    - top_n_events: The maximum number of anomaly events to return, sorted by severity.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event.
    """
    
    # Constants and Setup
    data_source_path = './coding/dataset/Bank/telemetry'
    tz = pytz.timezone('Asia/Shanghai')
    
    target_kpis = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    try:
        start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-mm-dd HH:MM:SS'")

    # Data Loading
    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        return []
        
    df = pd.read_csv(metric_path)
    df = df[df['kpi_name'].isin(target_kpis)]
    if df.empty:
        return []

    # Timezone conversion and filtering
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(tz)
    
    extended_start_time = start_time - timedelta(minutes=15)
    extended_end_time = end_time + timedelta(minutes=15)
    
    df_extended = df[(df['datetime'] >= extended_start_time) & (df['datetime'] <= extended_end_time)]
    if df_extended.empty:
        return []

    all_anomalies = []

    # Main loop per KPI
    for kpi in df_extended['kpi_name'].unique():
        df_kpi = df_extended[df_extended['kpi_name'] == kpi].copy()
        
        # Normalization
        min_val, max_val = df_kpi['value'].min(), df_kpi['value'].max()
        if max_val == min_val:
            continue
        df_kpi['normalized_value'] = 100 * (df_kpi['value'] - min_val) / (max_val - min_val)

        # Pivot to get time series for each component
        pivot_df = df_kpi.pivot_table(index='datetime', columns='cmdb_id', values='normalized_value')
        pivot_df = pivot_df.resample('1min').mean().ffill().bfill()
        
        # Filter pivot table to extended window
        pivot_df = pivot_df[(pivot_df.index >= extended_start_time) & (pivot_df.index <= extended_end_time)]
        if pivot_df.empty:
            continue

        # Sort components by peak value in diagnosis window
        diag_window_df = pivot_df[(pivot_df.index >= start_time) & (pivot_df.index <= end_time)]
        if diag_window_df.empty:
            continue
        
        peak_values = diag_window_df.max().sort_values(ascending=False)
        sorted_components = peak_values.index

        kpi_anomalies = []
        
        # Anomaly detection loop per component
        for cmdb_id in sorted_components:
            ts = pivot_df[cmdb_id].dropna()
            if len(ts) < (stable_stage_len + anomaly_stage_min_len + stable_stage_len):
                continue

            component_anomalies_count = 0
            # Iterate through possible anomaly windows
            for anomaly_len in range(anomaly_stage_max_len, anomaly_stage_min_len - 1, -1):
                if component_anomalies_count >= 2: break
                
                # The anomaly window must be strictly within the diagnosis window
                diag_ts = ts[(ts.index >= start_time) & (ts.index <= end_time)]
                
                for i in range(len(diag_ts) - anomaly_len + 1):
                    if component_anomalies_count >= 2: break
                    
                    anomaly_window_start_idx_in_diag = i
                    
                    anomaly_start_time_ts = diag_ts.index[anomaly_window_start_idx_in_diag]
                    
                    # Find the index in the full extended timeseries
                    try:
                        anomaly_start_idx_full = ts.index.get_loc(anomaly_start_time_ts)
                    except KeyError:
                        continue
                    
                    anomaly_end_idx_full = anomaly_start_idx_full + anomaly_len - 1

                    # Define stable and recovery windows
                    stable_start_idx = max(0, anomaly_start_idx_full - stable_stage_len)
                    stable_end_idx = anomaly_start_idx_full
                    
                    recovery_start_idx = anomaly_end_idx_full + 1
                    recovery_end_idx = min(len(ts), recovery_start_idx + stable_stage_len)

                    if (stable_end_idx - stable_start_idx < stable_stage_len or
                        recovery_end_idx - recovery_start_idx < stable_stage_len):
                        continue

                    stable_phase = ts.iloc[stable_start_idx:stable_end_idx]
                    anomaly_phase = ts.iloc[anomaly_start_idx_full:recovery_start_idx]
                    recovery_phase = ts.iloc[recovery_start_idx:recovery_end_idx]

                    # Check conditions
                    is_stable = stable_phase.std() < stable_std_thresh
                    is_recovered = recovery_phase.std() < stable_std_thresh
                    
                    if not (is_stable and is_recovered):
                        continue

                    ramp_value = abs(anomaly_phase.iloc[0] - stable_phase.iloc[-1])
                    if ramp_value < ramp_thresh:
                        continue
                        
                    stable_avg = stable_phase.mean()
                    anomaly_peak = anomaly_phase.max()
                    anomaly_valley = anomaly_phase.min()
                    
                    # Determine if it's a spike or a drop
                    if abs(anomaly_peak - stable_avg) > abs(anomaly_valley - stable_avg):
                        delta = anomaly_peak - stable_avg
                        description_verb = "abnormally high"
                    else:
                        delta = stable_avg - anomaly_valley
                        description_verb = "abnormally low"

                    if delta <= 0:
                        continue

                    # Avoid re-detecting the same event
                    is_new_event = True
                    for existing_event in kpi_anomalies:
                        if existing_event['cmdb_id'] == cmdb_id:
                            time_diff = abs((anomaly_start_time_ts - existing_event['timestamp_obj']).total_seconds() / 60)
                            if time_diff < (anomaly_stage_max_len + stable_stage_len):
                                is_new_event = False
                                break
                    
                    if is_new_event:
                        kpi_anomalies.append({
                            'timestamp_obj': anomaly_start_time_ts,
                            'cmdb_id': cmdb_id,
                            'kpi_name': kpi,
                            'delta': delta,
                            'description': f"'{kpi}' on '{cmdb_id}' is {description_verb}, delta={delta:.2f}"
                        })
                        component_anomalies_count += 1

        # Noise reduction for the current KPI
        if not kpi_anomalies:
            continue
            
        max_delta_kpi = max(a['delta'] for a in kpi_anomalies)
        denoised_anomalies = [a for a in kpi_anomalies if a['delta'] >= noise_reduction_w * max_delta_kpi]
        
        if not denoised_anomalies:
            continue

        # Clustering
        denoised_anomalies.sort(key=lambda x: x['timestamp_obj'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for i in range(1, len(denoised_anomalies)):
                time_diff = (denoised_anomalies[i]['timestamp_obj'] - current_cluster[-1]['timestamp_obj']).total_seconds() / 60
                if time_diff <= cluster_minutes_k:
                    current_cluster.append(denoised_anomalies[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [denoised_anomalies[i]]
            clusters.append(current_cluster)

        if len(clusters) > max_clusters_nc:
            continue 
        
        denoised_anomalies.sort(key=lambda x: x['delta'], reverse=True)
        top_event_clusters = set()
        if len(denoised_anomalies) > 0:
            top_event_ts = denoised_anomalies[0]['timestamp_obj']
            for cluster in clusters:
                if any(event['timestamp_obj'] == top_event_ts for event in cluster):
                    top_event_clusters.add(tuple(c['timestamp_obj'] for c in cluster))
                    break
        if len(denoised_anomalies) > 1:
            second_top_event_ts = denoised_anomalies[1]['timestamp_obj']
            for cluster in clusters:
                if any(event['timestamp_obj'] == second_top_event_ts for event in cluster):
                    top_event_clusters.add(tuple(c['timestamp_obj'] for c in cluster))
                    break
        
        final_kpi_anomalies = []
        for cluster in clusters:
            if tuple(c['timestamp_obj'] for c in cluster) in top_event_clusters:
                final_kpi_anomalies.extend(cluster)

        all_anomalies.extend(final_kpi_anomalies)

    # Final filtering across all KPIs
    if not all_anomalies:
        return []

    max_overall_delta = max(a['delta'] for a in all_anomalies)
    final_anomalies = [a for a in all_anomalies if a['delta'] >= final_filter_x * max_overall_delta]

    # Sort, format, and return top N events
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)
    
    output_events = []
    for anomaly in final_anomalies[:top_n_events]:
        output_events.append({
            "data_source": "Metric",
            "timestamp": anomaly['timestamp_obj'].strftime('%Y-%m-%d %H:%M:%S'),
            "cmdb_id": anomaly['cmdb_id'],
            "description": anomaly['description'],
        })
        
    return output_events
if __name__ == '__main__':
    date = "2021_03_09"
    start_time = "2021-03-09 16:30:00"
    end_time = "2021-03-09 17:00:00"
    
    anomaly_events = find_metric_anomalies_by_shape_a949f08f(date, start_time, end_time)
    
    print(anomaly_events)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_by_shape_0020b3c4(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_09'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-09 20:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-09 20:30:00'"],
    stable_std_thresh: Annotated[float, "Maximum standard deviation for a phase to be considered stable."] = 10.0,
    ramp_thresh: Annotated[float, "Minimum ramp value to be considered a significant change."] = 7.0,
    noise_reduction_w: Annotated[float, "Weight for noise reduction filter (w * max_delta)."] = 0.3,
    max_clusters_nc: Annotated[int, "Maximum number of anomaly clusters allowed for a metric to be a root cause candidate."] = 5,
    final_filter_x: Annotated[float, "Weight for the final global noise filter (x * global_max_delta)."] = 0.05,
    top_n_events: Annotated[int, "The number of top anomaly events to return."] = 10,
    extended_window_minutes: Annotated[int, "Minutes to extend the window for context analysis."] = 15,
    stable_phase_len: Annotated[int, "Number of data points for the stable phase."] = 5,
    anomaly_phase_max_len: Annotated[int, "Maximum number of data points for the anomaly phase."] = 8,
    cluster_time_k_minutes: Annotated[int, "Time window in minutes to cluster anomalies."] = 3
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Analyzes time-series metric data to find shape-based anomalies (spike or drop). It normalizes metrics,
    identifies patterns with stable, anomalous, and recovery phases, filters noise using clustering and delta-based thresholds,
    and returns the most significant anomaly events.

    Parameters:
    - date_str: The date of the telemetry data to analyze.
    - start_time_str: The start of the diagnosis time window.
    - end_time_str: The end of the diagnosis time window.
    - stable_std_thresh: Threshold for stable phase standard deviation.
    - ramp_thresh: Threshold for the ramp between stable and anomaly phases.
    - noise_reduction_w: Weight for metric-level noise reduction.
    - max_clusters_nc: Maximum allowed anomaly clusters for a metric.
    - final_filter_x: Weight for global noise reduction.
    - top_n_events: Number of top events to return.
    - extended_window_minutes: Minutes to extend the time window for better context.
    - stable_phase_len: Length of the stable phase.
    - anomaly_phase_max_len: Maximum length of the anomaly phase.
    - cluster_time_k_minutes: Time window for clustering anomalies.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    
    KPI_LIST = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    tz = pytz.timezone('Asia/Shanghai')
    try:
        diag_start_dt = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        diag_end_dt = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-mm-dd HH:MM:SS'.")

    diag_start_ts = int(diag_start_dt.timestamp())
    diag_end_ts = int(diag_end_dt.timestamp())

    extended_start_dt = diag_start_dt - timedelta(minutes=extended_window_minutes)
    extended_end_dt = diag_end_dt + timedelta(minutes=extended_window_minutes)
    extended_start_ts = int(extended_start_dt.timestamp())
    extended_end_ts = int(extended_end_dt.timestamp())

    metric_file = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_file):
        return []

    df = pd.read_csv(metric_file)
    df = df[df['kpi_name'].isin(KPI_LIST)]
    df = df[(df['timestamp'] >= extended_start_ts) & (df['timestamp'] <= extended_end_ts)]

    if df.empty:
        return []

    all_anomaly_events = []

    for kpi_name in df['kpi_name'].unique():
        kpi_df = df[df['kpi_name'] == kpi_name].copy()
        
        min_val, max_val = kpi_df['value'].min(), kpi_df['value'].max()
        if max_val > min_val:
            kpi_df['norm_value'] = 100 * (kpi_df['value'] - min_val) / (max_val - min_val)
        else:
            kpi_df['norm_value'] = 0.0

        diag_kpi_df = kpi_df[(kpi_df['timestamp'] >= diag_start_ts) & (kpi_df['timestamp'] <= diag_end_ts)]
        if diag_kpi_df.empty:
            continue

        peak_values = diag_kpi_df.groupby('cmdb_id')['norm_value'].max()
        sorted_components = peak_values.sort_values(ascending=False).index.tolist()

        metric_anomalies = []
        for cmdb_id in sorted_components:
            comp_df = kpi_df[kpi_df['cmdb_id'] == cmdb_id].sort_values('timestamp').reset_index(drop=True)
            if len(comp_df) < (2 * stable_phase_len + 1):
                continue

            comp_anomalies = []
            
            diag_indices = comp_df[(comp_df['timestamp'] >= diag_start_ts) & (comp_df['timestamp'] <= diag_end_ts)].index
            if len(diag_indices) == 0:
                continue

            sorted_diag_points = comp_df.loc[diag_indices].sort_values('norm_value', ascending=False)
            
            covered_indices = set()

            for idx, point in sorted_diag_points.iterrows():
                if idx in covered_indices:
                    continue
                
                # Search for anomaly shape around this point
                found_shape = False
                for anom_len in range(1, anomaly_phase_max_len + 1):
                    if found_shape: break
                    for peak_offset in range(anom_len):
                        if found_shape: break
                        
                        start_idx = idx - peak_offset
                        end_idx = start_idx + anom_len - 1

                        if not (start_idx in diag_indices and end_idx in diag_indices):
                            continue

                        stable_start_idx = start_idx - stable_phase_len
                        stable_end_idx = start_idx - 1
                        
                        recov_start_idx = end_idx + 1
                        recov_end_idx = end_idx + stable_phase_len

                        if stable_start_idx < 0 or recov_end_idx >= len(comp_df):
                            continue

                        stable_vals = comp_df.loc[stable_start_idx:stable_end_idx, 'norm_value']
                        anom_vals = comp_df.loc[start_idx:end_idx, 'norm_value']
                        recov_vals = comp_df.loc[recov_start_idx:recov_end_idx, 'norm_value']

                        if len(stable_vals) < stable_phase_len or len(recov_vals) < stable_phase_len:
                            continue

                        is_spike = (stable_vals.std() < stable_std_thresh and
                                    recov_vals.std() < stable_std_thresh and
                                    (anom_vals.iloc[0] - stable_vals.iloc[-1]) > ramp_thresh)
                        
                        is_drop = (stable_vals.std() < stable_std_thresh and
                                   recov_vals.std() < stable_std_thresh and
                                   (stable_vals.iloc[-1] - anom_vals.iloc[0]) > ramp_thresh)

                        if is_spike or is_drop:
                            anomaly_type = 'high' if is_spike else 'low'
                            stable_mean = stable_vals.mean()
                            delta = (anom_vals.max() - stable_mean) if is_spike else (stable_mean - anom_vals.min())
                            
                            anomaly_ts = comp_df.loc[start_idx, 'timestamp']
                            
                            event = {
                                'data_source': 'Metric',
                                'timestamp_sec': anomaly_ts,
                                'cmdb_id': cmdb_id,
                                'kpi_name': kpi_name,
                                'description': f"Metric '{kpi_name}' on component '{cmdb_id}' is abnormally {anomaly_type}, delta={delta:.2f}",
                                'delta': delta
                            }
                            comp_anomalies.append(event)
                            
                            for i in range(stable_start_idx, recov_end_idx + 1):
                                covered_indices.add(i)
                            found_shape = True
            
            comp_anomalies.sort(key=lambda x: x['delta'], reverse=True)
            metric_anomalies.extend(comp_anomalies[:2])

        if not metric_anomalies:
            continue

        max_delta = max(e['delta'] for e in metric_anomalies)
        denoised_anomalies = [e for e in metric_anomalies if e['delta'] >= noise_reduction_w * max_delta]

        if not denoised_anomalies:
            continue

        denoised_anomalies.sort(key=lambda x: x['timestamp_sec'])
        
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for i in range(1, len(denoised_anomalies)):
                if (denoised_anomalies[i]['timestamp_sec'] - current_cluster[-1]['timestamp_sec']) <= cluster_time_k_minutes * 60:
                    current_cluster.append(denoised_anomalies[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [denoised_anomalies[i]]
            clusters.append(current_cluster)

        if 1 <= len(clusters) <= max_clusters_nc:
            clusters.sort(key=lambda c: max(e['delta'] for e in c), reverse=True)
            for event in clusters[0]:
                all_anomaly_events.append(event)
            if len(clusters) > 1:
                 for event in clusters[1]:
                    all_anomaly_events.append(event)

    if not all_anomaly_events:
        return []

    global_max_delta = max(e['delta'] for e in all_anomaly_events)
    final_anomalies = [e for e in all_anomaly_events if e['delta'] >= final_filter_x * global_max_delta]
    
    # Deduplicate
    final_anomalies_unique = []
    seen = set()
    for event in final_anomalies:
        key = (event['timestamp_sec'], event['cmdb_id'], event['kpi_name'])
        if key not in seen:
            final_anomalies_unique.append(event)
            seen.add(key)

    final_anomalies_unique.sort(key=lambda x: x['delta'], reverse=True)
    
    output_events = []
    for event in final_anomalies_unique[:top_n_events]:
        ts_str = pd.to_datetime(event['timestamp_sec'], unit='s', utc=True).tz_convert(tz).strftime('%Y-%m-%d %H:%M:%S')
        output_events.append({
            'data_source': event['data_source'],
            'timestamp': ts_str,
            'cmdb_id': event['cmdb_id'],
            'description': event['description']
        })

    return output_events
if __name__ == '__main__':
    date_str = '2021_03_09'
    start_time_str = '2021-03-09 20:00:00'
    end_time_str = '2021-03-09 20:30:00'
    
    anomaly_events = find_metric_anomalies_by_shape_0020b3c4(date_str, start_time_str, end_time_str)
    
    print(anomaly_events)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_by_shape_1dd91b92(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-04 13:30:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-04 14:00:00'"],
    stable_std_thresh: Annotated[float, "Standard deviation threshold to identify a stable phase."] = 10.0,
    ramp_thresh: Annotated[float, "Threshold for the ramp-up value from stable to anomaly phase."] = 8.0,
    noise_reduction_w: Annotated[float, "Weight for per-metric noise reduction based on max delta."] = 0.3,
    cluster_time_k: Annotated[int, "Time window in minutes for clustering anomalies."] = 3,
    max_clusters_nc: Annotated[int, "Maximum number of clusters allowed for a metric to be considered a root cause."] = 5,
    final_filter_x: Annotated[float, "Weight for final global noise reduction based on max delta."] = 0.05,
    top_n: Annotated[int, "The maximum number of anomaly events to return."] = 10
) -> List[Dict[str, Any]]:
    """
    This tool analyzes metric time-series data to detect anomalies based on their shape.
    It identifies patterns with stable, anomalous, and recovery phases, filters noise,
    and clusters events to highlight potential root causes.

    Parameters:
    - date_str: The date for which to analyze data, e.g., '2021_03_04'.
    - start_time_str: The start of the diagnosis window, e.g., '2021-03-04 13:30:00'.
    - end_time_str: The end of the diagnosis window, e.g., '2021-03-04 14:00:00'.
    - stable_std_thresh: Max standard deviation for a phase to be considered stable.
    - ramp_thresh: Minimum delta between the start of an anomaly and the preceding stable average.
    - noise_reduction_w: Factor to compute the per-metric delta threshold for noise reduction (w * max_delta).
    - cluster_time_k: Time in minutes to group anomalies into a single cluster.
    - max_clusters_nc: If a metric has more than this many clusters, it's considered noisy and discarded.
    - final_filter_x: Factor to compute the global delta threshold for final filtering (x * max_delta).
    - top_n: The number of top anomalies to return, sorted by severity (delta).

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event.
    """
    
    # 1. Setup Timezone and Time Windows
    try:
        tz = pytz.timezone('Asia/Shanghai')
        start_dt = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_dt = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS'.")

    start_utc = start_dt.astimezone(pytz.utc)
    end_utc = end_dt.astimezone(pytz.utc)
    
    extended_start_utc = start_utc - timedelta(minutes=15)
    extended_end_utc = end_utc + timedelta(minutes=15)

    # 2. Load and Prepare Data
    data_source_path = './coding/dataset/Bank/telemetry'
    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Metric data file not found at {metric_path}")

    df = pd.read_csv(metric_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    
    target_kpis = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    df_filtered = df[(df['datetime'] >= extended_start_utc) & (df['datetime'] <= extended_end_utc) & (df['kpi_name'].isin(target_kpis))].copy()

    if df_filtered.empty:
        return []

    # Resample to 1-minute intervals to ensure consistent time steps
    df_resampled = df_filtered.set_index('datetime').groupby(['cmdb_id', 'kpi_name']).resample('1min').mean().drop(columns=['timestamp'], errors='ignore').reset_index()

    all_valid_anomalies = []

    # 3. Process each KPI
    for kpi in df_resampled['kpi_name'].unique():
        kpi_df = df_resampled[df_resampled['kpi_name'] == kpi].copy()
        
        # Normalize data for the current KPI
        min_val, max_val = kpi_df['value'].min(), kpi_df['value'].max()
        if max_val > min_val:
            kpi_df['norm_value'] = 100 * (kpi_df['value'] - min_val) / (max_val - min_val)
        else:
            kpi_df['norm_value'] = 0

        kpi_pivot = kpi_df.pivot(index='datetime', columns='cmdb_id', values='norm_value')
        kpi_pivot = kpi_pivot.interpolate(method='time').fillna(0)
        
        # Sort components by peak value in the diagnosis window
        diagnosis_window_data = kpi_pivot.loc[start_utc:end_utc]
        if diagnosis_window_data.empty:
            continue
            
        peak_values = diagnosis_window_data.max().sort_values(ascending=False)
        sorted_components = peak_values.index

        metric_anomalies = []

        # 4. Find anomaly events for each component
        for component in sorted_components:
            ts = kpi_pivot[component]
            if ts.empty:
                continue

            potential_events = []
            diagnosis_indices = ts.index.get_indexer_for(diagnosis_window_data.index)
            
            for i in diagnosis_indices:
                if i < 5: continue # Need at least 5 points for a stable phase
                
                for anomaly_len in range(1, 9):
                    # Define phase indices
                    anomaly_start_idx, anomaly_end_idx = i, i + anomaly_len - 1
                    stable_start_idx, stable_end_idx = i - 5, i - 1
                    recovery_start_idx, recovery_end_idx = anomaly_end_idx + 1, anomaly_end_idx + 5

                    # Check if phases are within bounds and diagnosis window constraints
                    if anomaly_end_idx >= len(ts) or recovery_end_idx >= len(ts) or ts.index[anomaly_end_idx] > end_utc:
                        continue
                    
                    # Extract phase data
                    stable_slice = ts.iloc[stable_start_idx:stable_end_idx + 1]
                    anomaly_slice = ts.iloc[anomaly_start_idx:anomaly_end_idx + 1]
                    recovery_slice = ts.iloc[recovery_start_idx:recovery_end_idx + 1]

                    if len(stable_slice) < 5 or len(recovery_slice) < 5:
                        continue

                    # Check anomaly conditions
                    std_stable = stable_slice.std()
                    std_recovery = recovery_slice.std()
                    avg_stable = stable_slice.mean()
                    ramp_value = ts.iloc[i] - avg_stable
                    
                    if std_stable < stable_std_thresh and std_recovery < stable_std_thresh and ramp_value > ramp_thresh:
                        peak_anomaly = anomaly_slice.max()
                        delta = peak_anomaly - avg_stable
                        if delta > 0:
                            potential_events.append({
                                'start_time': ts.index[i],
                                'delta': delta,
                                'cmdb_id': component,
                                'kpi_name': kpi,
                                'indices': set(range(i, anomaly_end_idx + 1))
                            })
            
            # Greedy selection of best non-overlapping events
            potential_events.sort(key=lambda x: x['delta'], reverse=True)
            covered_indices = set()
            component_anomalies = []
            for event in potential_events:
                if not event['indices'].intersection(covered_indices):
                    component_anomalies.append(event)
                    covered_indices.update(event['indices'])
                    if len(component_anomalies) >= 2:
                        break
            metric_anomalies.extend(component_anomalies)

        if not metric_anomalies:
            continue

        # 5. Per-Metric Noise Reduction
        max_delta_metric = max(event['delta'] for event in metric_anomalies)
        delta_thresh_metric = noise_reduction_w * max_delta_metric
        denoised_anomalies = [event for event in metric_anomalies if event['delta'] >= delta_thresh_metric]

        if not denoised_anomalies:
            continue
            
        # 6. Clustering and Root Cause Potential
        denoised_anomalies.sort(key=lambda x: x['start_time'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            clusters.append(current_cluster)
            for event in denoised_anomalies[1:]:
                if event['start_time'] - current_cluster[-1]['start_time'] <= timedelta(minutes=cluster_time_k):
                    current_cluster.append(event)
                else:
                    current_cluster = [event]
                    clusters.append(current_cluster)
        
        if 1 <= len(clusters) <= max_clusters_nc:
            all_valid_anomalies.extend(denoised_anomalies)

    # 7. Final Filtering and Formatting
    if not all_valid_anomalies:
        return []

    max_delta_global = max(event['delta'] for event in all_valid_anomalies)
    final_filter_thresh = final_filter_x * max_delta_global
    
    final_anomalies = [event for event in all_valid_anomalies if event['delta'] >= final_filter_thresh]
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)
    
    anomaly_events = []
    for event in final_anomalies[:top_n]:
        anomaly_events.append({
            'data_source': 'Metric',
            'timestamp': event['start_time'].astimezone(tz).strftime('%Y-%m-%d %H:%M:%S'),
            'cmdb_id': event['cmdb_id'],
            'description': f"Metric '{event['kpi_name']}' is abnormally high, delta={event['delta']:.2f}"
        })

    return anomaly_events
if __name__ == '__main__':
    anomaly_events = find_metric_anomalies_by_shape_1dd91b92(
        date_str='2021_03_04',
        start_time_str='2021-03-04 13:30:00',
        end_time_str='2021-03-04 14:00:00'
    )
    print(anomaly_events)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from typing import Annotated, List, Dict, Any, Optional
def find_metric_anomalies_by_shape_d49d4823(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    start_time_str: Annotated[str, "Start time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-04 21:00:00'"],
    end_time_str: Annotated[str, "End time of the diagnosis window in 'YYYY-mm-dd HH:MM:SS' format, e.g., '2021-03-04 21:30:00'"],
    stable_stage_len: Annotated[int, "Number of data points for the stable stage before an anomaly."] = 5,
    anomaly_stage_max_len: Annotated[int, "Maximum number of data points for the anomaly stage."] = 8,
    recovery_stage_len: Annotated[int, "Number of data points for the recovery stage after an anomaly."] = 5,
    stable_std_thresh: Annotated[float, "Standard deviation threshold for a stage to be considered stable."] = 12.0,
    ramp_thresh: Annotated[float, "Threshold for the value change to be considered an anomaly ramp."] = 7.0,
    noise_reduction_w: Annotated[float, "Weight for noise reduction. Events with delta < w * max_delta are removed."] = 0.2,
    cluster_time_k_minutes: Annotated[int, "Time window in minutes to cluster anomalies together."] = 3,
    max_clusters_nc: Annotated[int, "Maximum number of anomaly clusters allowed for a metric to be considered a root cause candidate."] = 5,
    final_filter_x: Annotated[float, "Final filter weight. Events with delta < x * overall_max_delta are removed."] = 0.05,
    top_n: Annotated[int, "Number of top anomaly events to return."] = 15
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Analyzes metric time-series data to find shape-based anomalies (spikes or drops) within a specified time window.
    It normalizes metric values, identifies patterns of 'stable -> anomaly -> recovery', and filters out noise to report the most significant events.

    Working Principle:
    1.  Extends the diagnosis window by 15 minutes on both sides for context.
    2.  Loads container metrics for the specified date.
    3.  For each target KPI, normalizes its values across all components to a 0-100 scale.
    4.  For each component, it scans the time-series for anomaly shapes (both spikes and drops). An anomaly shape consists of a stable phase, a sharp ramp, an anomalous phase, and a recovery phase.
    5.  Filters detected anomalies for the metric based on their delta (change from stable to peak).
    6.  Clusters the remaining anomalies by time. If too many clusters are found, the metric is considered noisy and its anomalies are discarded.
    7.  Finally, aggregates all valid anomalies from all metrics, performs a final filtering based on the overall maximum delta, and returns the top N events.

    Parameters:
    - date_str: The date of the telemetry data to analyze.
    - start_time_str: The start of the diagnosis time window.
    - end_time_str: The end of the diagnosis time window.
    - stable_stage_len: Length of the stable phase.
    - anomaly_stage_max_len: Maximum length of the anomaly phase.
    - recovery_stage_len: Length of the recovery phase.
    - stable_std_thresh: STD threshold to define a stable phase.
    - ramp_thresh: Minimum ramp value to detect an anomaly.
    - noise_reduction_w: Denoising factor for metric-level anomalies.
    - cluster_time_k_minutes: Time window for clustering anomalies.
    - max_clusters_nc: Maximum number of clusters for a metric to be considered significant.
    - final_filter_x: Denoising factor for the final list of all anomalies.
    - top_n: The maximum number of anomalies to return.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity (delta).
    """
    
    data_source_path = './coding/dataset/Bank/telemetry'
    tz = pytz.timezone('Asia/Shanghai')
    
    try:
        start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        raise ValueError("Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS'.")

    extended_start_time = start_time - timedelta(minutes=15)
    extended_end_time = end_time + timedelta(minutes=15)
    
    target_kpis = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        return []
        
    df = pd.read_csv(metric_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(tz)
    
    df_extended = df[(df['datetime'] >= extended_start_time) & (df['datetime'] <= extended_end_time)]
    df_extended = df_extended[df_extended['kpi_name'].isin(target_kpis)].copy()

    if df_extended.empty:
        return []

    all_valid_anomalies = []

    for kpi in df_extended['kpi_name'].unique():
        kpi_anomalies = []
        df_kpi = df_extended[df_extended['kpi_name'] == kpi].copy()

        min_val, max_val = df_kpi['value'].min(), df_kpi['value'].max()
        df_kpi['value_norm'] = 100 * (df_kpi['value'] - min_val) / (max_val - min_val) if max_val > min_val else 0.0

        components = df_kpi['cmdb_id'].unique()
        
        component_peaks = {c: df_kpi[(df_kpi['cmdb_id'] == c) & (df_kpi['datetime'].between(start_time, end_time))]['value_norm'].max() for c in components}
        component_valleys = {c: df_kpi[(df_kpi['cmdb_id'] == c) & (df_kpi['datetime'].between(start_time, end_time))]['value_norm'].min() for c in components}
        
        sorted_components_spike = sorted([c for c, p in component_peaks.items() if p is not np.nan], key=component_peaks.get, reverse=True)
        sorted_components_drop = sorted([c for c, v in component_valleys.items() if v is not np.nan], key=component_valleys.get)

        for anomaly_type, sorted_components in [('spike', sorted_components_spike), ('drop', sorted_components_drop)]:
            for comp in sorted_components:
                comp_events = 0
                df_comp = df_kpi[df_kpi['cmdb_id'] == comp].sort_values('datetime').reset_index(drop=True)
                
                if len(df_comp) < stable_stage_len + 1 + recovery_stage_len: continue

                i = stable_stage_len
                while i < len(df_comp) - recovery_stage_len:
                    if comp_events >= 2: break
                    
                    if not (start_time <= df_comp.loc[i, 'datetime'] <= end_time):
                        i += 1
                        continue

                    stable_window = df_comp.iloc[i - stable_stage_len : i]
                    if stable_window.empty or stable_window['value_norm'].std(ddof=0) >= stable_std_thresh:
                        i += 1
                        continue
                    
                    avg_stable_value = stable_window['value_norm'].mean()
                    ramp_val = df_comp.loc[i, 'value_norm'] - df_comp.loc[i - 1, 'value_norm']
                    
                    is_ramp = (anomaly_type == 'spike' and ramp_val > ramp_thresh) or \
                              (anomaly_type == 'drop' and -ramp_val > ramp_thresh)
                    
                    if not is_ramp:
                        i += 1
                        continue

                    found_anomaly = False
                    for k in range(1, anomaly_stage_max_len + 1):
                        anomaly_end_idx = i + k
                        if anomaly_end_idx + recovery_stage_len > len(df_comp): break
                        if not (df_comp.loc[anomaly_end_idx - 1, 'datetime'] <= end_time): break

                        recovery_window = df_comp.iloc[anomaly_end_idx : anomaly_end_idx + recovery_stage_len]
                        if recovery_window['value_norm'].std(ddof=0) < stable_std_thresh:
                            anomaly_window = df_comp.iloc[i:anomaly_end_idx]
                            delta = (anomaly_window['value_norm'].max() - avg_stable_value) if anomaly_type == 'spike' else (avg_stable_value - anomaly_window['value_norm'].min())
                            
                            if delta > 0:
                                kpi_anomalies.append({'timestamp': df_comp.loc[i, 'datetime'], 'cmdb_id': comp, 'kpi_name': kpi, 'delta': delta, 'type': anomaly_type})
                                comp_events += 1
                                i = anomaly_end_idx + recovery_stage_len
                                found_anomaly = True
                                break
                    if not found_anomaly:
                        i += 1

        if not kpi_anomalies: continue

        max_delta_kpi = max(a['delta'] for a in kpi_anomalies)
        denoised_anomalies = [a for a in kpi_anomalies if a['delta'] >= noise_reduction_w * max_delta_kpi]
        if not denoised_anomalies: continue

        denoised_anomalies.sort(key=lambda x: x['timestamp'])
        clusters = []
        if denoised_anomalies:
            current_cluster = [denoised_anomalies[0]]
            for anomaly in denoised_anomalies[1:]:
                if (anomaly['timestamp'] - current_cluster[-1]['timestamp']).total_seconds() / 60 <= cluster_time_k_minutes:
                    current_cluster.append(anomaly)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [anomaly]
            clusters.append(current_cluster)
        
        if 1 <= len(clusters) <= max_clusters_nc:
            all_valid_anomalies.extend(denoised_anomalies)

    if not all_valid_anomalies: return []

    overall_max_delta = max(a['delta'] for a in all_valid_anomalies)
    final_anomalies = [a for a in all_valid_anomalies if a['delta'] >= final_filter_x * overall_max_delta]
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)
    
    anomaly_events = []
    seen_keys = set()
    for anomaly in final_anomalies:
        key = (anomaly['cmdb_id'], anomaly['kpi_name'], anomaly['type'])
        if key not in seen_keys:
            description = f"Metric '{anomaly['kpi_name']}' on component '{anomaly['cmdb_id']}' showed an abnormal {anomaly['type']} with a change delta of {anomaly['delta']:.2f}."
            anomaly_events.append({'data_source': 'Metric', 'timestamp': anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'cmdb_id': anomaly['cmdb_id'], 'description': description})
            seen_keys.add(key)

    return anomaly_events[:top_n]
if __name__ == '__main__':
    date_str = "2021_03_04"
    start_time_str = "2021-03-04 21:00:00"
    end_time_str = "2021-03-04 21:30:00"
    
    anomaly_events = find_metric_anomalies_by_shape_d49d4823(
        date_str=date_str,
        start_time_str=start_time_str,
        end_time_str=end_time_str
    )
    
    if anomaly_events:
        print(anomaly_events)
    else:
        print([])



import subprocess
import sys
import os
from datetime import datetime
from typing import Annotated, List, Dict, Any, Optional
def detect_metric_anomalies_3b637087(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    start_time_str: Annotated[str, "Start time in 'YYYY-MM-DD HH:MM:SS' format (UTC+8), e.g., '2021-03-04 18:00:00'"],
    end_time_str: Annotated[str, "End time in 'YYYY-MM-DD HH:MM:SS' format (UTC+8), e.g., '2021-03-04 18:30:00'"],
    stable_window_size: Annotated[int, "Number of data points for the stable/recovery phase."] = 5,
    max_anomaly_window_size: Annotated[int, "Maximum number of data points for the anomaly phase."] = 8,
    stable_std_thresh_factor: Annotated[float, "Factor to determine the standard deviation threshold for stable phases."] = 0.8,
    slope_thresh_factor: Annotated[float, "Factor to determine the threshold for a sudden spike/drop."] = 1.2,
    noise_reduction_factor: Annotated[float, "Threshold factor to filter out noise events within a metric (w)."] = 0.4,
    cluster_window_minutes: Annotated[int, "Time window in minutes to group anomalies into a cluster (k)."] = 3,
    max_clusters: Annotated[int, "Maximum number of clusters allowed for a metric to be considered a potential root cause."] = 4,
    final_filter_factor: Annotated[float, "Final threshold factor to filter out noise across all metrics (x)."] = 0.1
) -> List[Dict[str, Any]]:
    """
    Analyzes metric data to detect and diagnose anomalies based on a pattern-matching approach.

    This tool processes time-series metric data for specified Key Performance Indicators (KPIs). It operates in the following stages:
    1.  Data Loading and Preprocessing: Loads metric data for a given date, filters it for a specified time window (extended by 15 minutes on both ends for context), and normalizes each KPI's values to a 0-100 scale.
    2.  Pattern-Based Anomaly Detection: For each component and KPI, it searches for a three-stage anomaly pattern: a stable phase, a sharp anomalous spike or drop, and a recovery phase.
    3.  Noise Reduction: Within each KPI, it filters out less significant anomalies based on the magnitude of the deviation (delta).
    4.  Clustering and Root Cause Analysis: It groups the remaining anomalies by their start times. KPIs with a small number of tight clusters are considered more likely to be root causes. Anomalies from KPIs with too many scattered clusters are discarded.
    5.  Final Filtering and Output: All identified anomalies from potential root-cause KPIs are aggregated, and a final noise filter is applied. The result is a ranked list of the most critical anomaly events.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event, sorted by severity.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    
    target_kpis = [
        'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
        'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
        'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
        'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
        'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
        'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
        'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
    ]

    tz = pytz.timezone('Asia/Shanghai')
    start_dt = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    end_dt = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    diag_start_ts = int(start_dt.timestamp())
    diag_end_ts = int(end_dt.timestamp())

    extended_start_ts = diag_start_ts - 15 * 60
    extended_end_ts = diag_end_ts + 15 * 60

    metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Metric file not found at {metric_path}")

    df = pd.read_csv(metric_path)
    df_filtered = df[(df['timestamp'] >= extended_start_ts) & (df['timestamp'] <= extended_end_ts) & (df['kpi_name'].isin(target_kpis))].copy()

    if df_filtered.empty:
        return []

    all_valid_anomalies = []

    for kpi in df_filtered['kpi_name'].unique():
        kpi_df = df_filtered[df_filtered['kpi_name'] == kpi].copy()
        
        min_val, max_val = kpi_df['value'].min(), kpi_df['value'].max()
        if max_val - min_val < 1e-6:
            continue
        kpi_df['norm_val'] = 100 * (kpi_df['value'] - min_val) / (max_val - min_val)

        pivot_df = kpi_df.pivot_table(index='timestamp', columns='cmdb_id', values='norm_val').sort_index()
        
        diag_pivot_df = pivot_df[(pivot_df.index >= diag_start_ts) & (pivot_df.index <= diag_end_ts)]
        if diag_pivot_df.empty:
            continue

        component_peaks = diag_pivot_df.max().sort_values(ascending=False)
        
        metric_anomalies = []
        for cmdb_id in component_peaks.index:
            series = pivot_df[cmdb_id].dropna()
            if len(series) < (2 * stable_window_size + 1):
                continue

            diag_series = series[(series.index >= diag_start_ts) & (series.index <= diag_end_ts)]
            sorted_diag_points = diag_series.sort_values(ascending=False)
            
            processed_indices = set()
            
            for ts, val in sorted_diag_points.items():
                if ts in processed_indices or len(metric_anomalies) >= 2:
                    break

                best_pattern = None
                
                for anomaly_len in range(1, max_anomaly_window_size + 1):
                    for offset in range(anomaly_len):
                        try:
                            anomaly_start_idx = series.index.get_loc(ts) - offset
                        except KeyError:
                            continue
                        anomaly_end_idx = anomaly_start_idx + anomaly_len - 1

                        if anomaly_start_idx < 0 or anomaly_end_idx >= len(series.index):
                            continue
                        
                        anomaly_start_ts = series.index[anomaly_start_idx]
                        anomaly_end_ts = series.index[anomaly_end_idx]

                        if not (diag_start_ts <= anomaly_start_ts and anomaly_end_ts <= diag_end_ts):
                            continue

                        stable_end_idx = anomaly_start_idx - 1
                        stable_start_idx = stable_end_idx - stable_window_size + 1
                        
                        recovery_start_idx = anomaly_end_idx + 1
                        recovery_end_idx = recovery_start_idx + stable_window_size - 1

                        if stable_start_idx < 0 or recovery_end_idx >= len(series.index):
                            continue

                        stable_vals = series.iloc[stable_start_idx:stable_start_idx + stable_window_size]
                        anomaly_vals = series.iloc[anomaly_start_idx:anomaly_start_idx + anomaly_len]
                        recovery_vals = series.iloc[recovery_start_idx:recovery_start_idx + stable_window_size]

                        stable_mean = stable_vals.mean()
                        stable_std = stable_vals.std()
                        recovery_std = recovery_vals.std()
                        
                        overall_std = series.std()
                        if overall_std < 1e-6: continue
                        
                        is_stable = stable_std < stable_std_thresh_factor * overall_std
                        is_recovered = recovery_std < stable_std_thresh_factor * overall_std
                        
                        ramp_up_val = anomaly_vals.iloc[0]
                        slope = ramp_up_val - stable_vals.iloc[-1]
                        
                        is_spike = slope > slope_thresh_factor * stable_std if stable_std > 0 else slope > 5
                        
                        if is_stable and is_recovered and is_spike:
                            delta = anomaly_vals.max() - stable_mean
                            if best_pattern is None or delta > best_pattern['delta']:
                                best_pattern = {
                                    'start_ts': anomaly_start_ts,
                                    'delta': delta,
                                    'cmdb_id': cmdb_id,
                                    'kpi': kpi,
                                    'indices': set(series.index[stable_start_idx:recovery_end_idx+1])
                                }
                
                if best_pattern:
                    metric_anomalies.append(best_pattern)
                    processed_indices.update(best_pattern['indices'])

        if not metric_anomalies:
            continue

        max_delta = max(a['delta'] for a in metric_anomalies)
        denoised_anomalies = [a for a in metric_anomalies if a['delta'] >= noise_reduction_factor * max_delta]
        
        denoised_anomalies.sort(key=lambda x: x['start_ts'])
        
        if not denoised_anomalies:
            continue
            
        clusters = []
        current_cluster = [denoised_anomalies[0]]
        for i in range(1, len(denoised_anomalies)):
            time_diff = (denoised_anomalies[i]['start_ts'] - current_cluster[-1]['start_ts']) / 60
            if time_diff <= cluster_window_minutes:
                current_cluster.append(denoised_anomalies[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [denoised_anomalies[i]]
        clusters.append(current_cluster)

        if 1 <= len(clusters) <= max_clusters:
            clusters.sort(key=lambda c: max(a['delta'] for a in c), reverse=True)
            top_clusters = clusters[:2]
            for cluster in top_clusters:
                all_valid_anomalies.extend(cluster)

    if not all_valid_anomalies:
        return []

    max_overall_delta = max(a['delta'] for a in all_valid_anomalies)
    final_anomalies = [a for a in all_valid_anomalies if a['delta'] >= final_filter_factor * max_overall_delta]
    final_anomalies.sort(key=lambda x: x['delta'], reverse=True)

    anomaly_events = []
    for anomaly in final_anomalies[:20]:
        ts_utc = datetime.fromtimestamp(anomaly['start_ts'], tz=pytz.utc)
        ts_shanghai = ts_utc.astimezone(tz)
        
        anomaly_events.append({
            "data_source": "Metric",
            "timestamp": ts_shanghai.strftime('%Y-%m-%d %H:%M:%S'),
            "cmdb_id": anomaly['cmdb_id'],
            "description": f"KPI '{anomaly['kpi']}' is abnormally high, delta={anomaly['delta']:.2f}"
        })
        
    return anomaly_events
if __name__ == '__main__':
    date_str = "2021_03_04"
    start_time_str = "2021-03-04 18:00:00"
    end_time_str = "2021-03-04 18:30:00"
    
    anomaly_events = detect_metric_anomalies_3b637087(date_str, start_time_str, end_time_str)
    
    print(anomaly_events)


