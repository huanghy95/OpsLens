from .utils import get_generated_functions_excluding_called
def get_logCoder_generated_functions():
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
        print(f"Error in get_logCoder_generated_functions: {e}")
        print(traceback.format_exc())
        return []



import pandas as pd
import pytz
from datetime import datetime
import re
import os
from typing import Annotated, List
def find_log_evidence_in_anomaly_window_498ac326(
    date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_07'"],
    start_time_str: Annotated[str, "Start time for the overall analysis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-07 12:30:00'"],
    end_time_str: Annotated[str, "End time for the overall analysis window in 'YYYY-MM-DD HH:MM:SS' format, e.g., '2021-03-07 13:00:00'"],
    anomaly_window_start_str: Annotated[str, "Start time of the specific metric anomaly window to focus on, e.g., '2021-03-07 12:41:00'"],
    anomaly_window_end_str: Annotated[str, "End time of the specific metric anomaly window to focus on, e.g., '2021-03-07 12:46:00'"],
    target_cmdb_ids: Annotated[List[str], "List of component IDs to investigate, e.g., ['Tomcat04', 'MG01']"],
    keywords: Annotated[List[str], "List of keywords to search for in logs, e.g., ['OOM', 'Exception']"]
) -> List[dict]:
    """
    Analyzes service logs to find evidence related to metric anomalies within a specific time window.

    This tool loads the log file for a given date, filters it by a specified time range and a list of
    target components. It then scans the log messages for a set of keywords (e.g., 'OOM', 'GC', 'Exception')
    specifically within a narrower time window corresponding to known metric anomalies. For logs related
    to garbage collection (GC), it attempts to extract the pause duration.

    Parameters:
    - date_str: The date of the logs to analyze, in 'YYYY_MM_DD' format.
    - start_time_str: The start of the broader time window to load logs from.
    - end_time_str: The end of the broader time window to load logs from.
    - anomaly_window_start_str: The start of the specific window to search for log evidence.
    - anomaly_window_end_str: The end of the specific window to search for log evidence.
    - target_cmdb_ids: A list of component IDs to focus the analysis on.
    - keywords: A list of strings to search for within the log messages.

    Returns:
    - A list of dictionaries, where each dictionary represents a significant log event found within the
      anomaly window. Returns an empty list if no such events are found.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    log_file_path = os.path.join(data_source_path, date_str, 'log', 'log_service.csv')

    if not os.path.exists(log_file_path):
        # print(f"Error: Log file not found at {log_file_path}")
        return []

    df = pd.read_csv(log_file_path)
    
    # Set up timezone
    tz = pytz.timezone('Asia/Shanghai')

    # Convert timestamp to datetime objects in the correct timezone
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)

    # Define time windows for filtering
    start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    anomaly_start_time = tz.localize(datetime.strptime(anomaly_window_start_str, '%Y-%m-%d %H:%M:%S'))
    anomaly_end_time = tz.localize(datetime.strptime(anomaly_window_end_str, '%Y-%m-%d %H:%M:%S'))

    # Filter by the overall time window and target components
    filtered_df = df[
        (df['datetime'] >= start_time) & 
        (df['datetime'] <= end_time) &
        (df['cmdb_id'].isin(target_cmdb_ids))
    ]

    # Further filter to the specific anomaly window
    anomaly_df = filtered_df[
        (filtered_df['datetime'] >= anomaly_start_time) &
        (filtered_df['datetime'] <= anomaly_end_time)
    ].copy()

    anomaly_events = []
    
    # Use a set to avoid adding duplicate log entries if they match multiple keywords
    processed_log_ids = set()

    for _, row in anomaly_df.iterrows():
        if row['log_id'] in processed_log_ids:
            continue

        log_value = str(row['value'])
        found_keyword = None

        for keyword in keywords:
            if re.search(keyword, log_value, re.IGNORECASE):
                found_keyword = keyword
                break
        
        if found_keyword:
            description = f"Keyword '{found_keyword}' found. "
            
            # Special handling for GC logs to extract pause duration
            if 'GC' in found_keyword.upper():
                gc_pause_match = re.search(r"real=([\d.]+) secs", log_value)
                if gc_pause_match:
                    pause_duration = gc_pause_match.group(1)
                    description += f"Extracted GC pause: {pause_duration}s. "

            description += f"Log value: '{log_value}'"

            event = {
                'data_source': 'Log',
                'timestamp': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'cmdb_id': row['cmdb_id'],
                'description': description
            }
            anomaly_events.append(event)
            processed_log_ids.add(row['log_id'])

    return anomaly_events
if __name__ == '__main__':
    # Parameters from the task description
    date = '2021_03_07'
    start_t = '2021-03-07 12:30:00'
    end_t = '2021-03-07 13:00:00'
    anomaly_start_t = '2021-03-07 12:41:00'
    anomaly_end_t = '2021-03-07 12:46:00'
    components = ['Tomcat04', 'MG01', 'MG02', 'IG01', 'IG02', 'Mysql02']
    search_keywords = [
        'OOM', 'OutOfMemoryError', 'OutOfMemory', 'GC', 'Full GC', 'long GC', 
        'GC pause', 'Exception', 'timeout', 'connection refused', 'socket', 
        'connection reset', 'java.lang.OutOfMemoryError'
    ]

    # Execute the analysis
    log_anomalies = find_log_evidence_in_anomaly_window_498ac326(
        date_str=date,
        start_time_str=start_t,
        end_time_str=end_t,
        anomaly_window_start_str=anomaly_start_t,
        anomaly_window_end_str=anomaly_end_t,
        target_cmdb_ids=components,
        keywords=search_keywords
    )

    # Print the results
    print(log_anomalies)
    if log_anomalies:
        pd.DataFrame(log_anomalies).to_csv('anomaly_events.csv', index=False)



import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
import re
from typing import Annotated, List, Dict, Any
def find_root_cause_from_logs_7c3dab74(
    start_time_str: Annotated[str, "Start of the time window for analysis, e.g., '2021-03-04 13:30:00'"],
    end_time_str: Annotated[str, "End of the time window for analysis, e.g., '2021-03-04 14:00:00'"],
    date_str: Annotated[str, "Date string for the data directory, e.g., '2021_03_04'"],
    metric_anomalies: Annotated[List[Dict[str, Any]], "List of metric anomalies to investigate, each a dict with 'timestamp' and 'cmdb_id'"],
    target_components: Annotated[List[str], "List of component IDs to focus on. Supports wildcards like 'Tomcat*'"],
    keywords: Annotated[List[str], "List of keywords to scan for in the logs"],
    snippet_window_minutes: Annotated[int, "Time window in minutes (+/-) around a metric anomaly to search for logs"] = 1
) -> List[Dict[str, Any]]:
    """
    Tool description:
    Scans service logs within a specified time window for specific keywords to find evidence related to pre-identified metric anomalies.
    It filters logs by time and component, searches for keywords, and then correlates the findings with the provided metric anomaly timestamps,
    extracting log snippets from a small window around each anomaly.

    Parameters:
    - start_time_str: The start of the overall time window to load logs from.
    - end_time_str: The end of the overall time window to load logs from.
    - date_str: The date for which to load the log data, corresponding to the directory name.
    - metric_anomalies: A list of dictionaries, where each dictionary represents a metric anomaly and must contain a 'timestamp' (e.g., '2021-03-04 13:46:00') and 'cmdb_id'.
    - target_components: A list of component names to investigate. Wildcards like 'Tomcat*' are supported for prefix matching.
    - keywords: A list of strings to search for within the log messages. The search is case-insensitive.
    - snippet_window_minutes: The number of minutes before and after each metric anomaly timestamp to look for relevant logs.

    Returns:
    - A list of anomaly_event dictionaries, each representing a significant log event found near a metric anomaly.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    log_path = os.path.join(data_source_path, date_str, 'log', 'log_service.csv')

    if not os.path.exists(log_path):
        # Return empty list if data file is not found, as per error handling guidelines
        return []

    # Setup timezone
    tz = pytz.timezone('Asia/Shanghai')

    # Convert window strings to datetime objects and then to UTC timestamps
    start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
    start_ts_utc = int(start_time.timestamp())
    end_ts_utc = int(end_time.timestamp())

    # Load and pre-process logs
    df_log = pd.read_csv(log_path)
    df_log = df_log[(df_log['timestamp'] >= start_ts_utc) & (df_log['timestamp'] <= end_ts_utc)]

    # Filter by target components
    component_filters = []
    for comp in target_components:
        if comp.endswith('*'):
            component_filters.append(df_log['cmdb_id'].str.startswith(comp[:-1]))
        else:
            component_filters.append(df_log['cmdb_id'] == comp)
    
    if not component_filters:
        return []
        
    df_filtered_log = df_log[pd.concat(component_filters, axis=1).any(axis=1)].copy()

    # Search for keywords
    keyword_pattern = '|'.join(keywords)
    df_keyword_hits = df_filtered_log[df_filtered_log['value'].str.contains(keyword_pattern, case=False, na=False, regex=True)].copy()

    if df_keyword_hits.empty:
        return []

    # Convert log timestamps to datetime objects for easier comparison
    df_keyword_hits['datetime'] = pd.to_datetime(df_keyword_hits['timestamp'], unit='s', utc=True).dt.tz_convert(tz)

    anomaly_events = []
    
    # Parse metric anomaly timestamps
    parsed_metric_anomalies = []
    for anom in metric_anomalies:
        try:
            ts = tz.localize(datetime.strptime(anom['timestamp'], '%Y-%m-%d %H:%M:%S'))
            parsed_metric_anomalies.append({'timestamp': ts, 'cmdb_id': anom['cmdb_id']})
        except (ValueError, KeyError):
            continue

    # Correlate logs with metric anomalies
    for anom in parsed_metric_anomalies:
        metric_ts = anom['timestamp']
        metric_cmdb_id = anom['cmdb_id']
        
        snippet_start = metric_ts - timedelta(minutes=snippet_window_minutes)
        snippet_end = metric_ts + timedelta(minutes=snippet_window_minutes)

        relevant_logs = df_keyword_hits[
            (df_keyword_hits['datetime'] >= snippet_start) &
            (df_keyword_hits['datetime'] <= snippet_end) &
            (df_keyword_hits['cmdb_id'] == metric_cmdb_id)
        ]

        for _, log_row in relevant_logs.iterrows():
            description = f"Log event found for {log_row['cmdb_id']} near metric anomaly at {metric_ts.strftime('%H:%M:%S')}. Content: {log_row['value']}"
            
            if 'gc' in str(log_row['log_name']).lower() or 'gc' in str(log_row['value']).lower():
                match = re.search(r'real=([\d.]+) secs', log_row['value'])
                if match:
                    pause_duration = float(match.group(1))
                    description = f"GC event on {log_row['cmdb_id']} with pause duration {pause_duration:.2f}s. Found near metric anomaly at {metric_ts.strftime('%H:%M:%S')}. Log: {log_row['value']}"
                else:
                    description = f"GC event on {log_row['cmdb_id']}. Found near metric anomaly at {metric_ts.strftime('%H:%M:%S')}. Log: {log_row['value']}"

            anomaly_events.append({
                "data_source": "Log",
                "timestamp": log_row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                "cmdb_id": log_row['cmdb_id'],
                "description": description
            })

    unique_events = []
    seen = set()
    for event in anomaly_events:
        event_tuple = (event['timestamp'], event['cmdb_id'], event['description'])
        if event_tuple not in seen:
            unique_events.append(event)
            seen.add(event_tuple)
            
    return unique_events
if __name__ == '__main__':
    start_time = "2021-03-04 13:30:00"
    end_time = "2021-03-04 14:00:00"
    date = "2021_03_04"
    
    metric_anomalies_input = [
        {'timestamp': '2021-03-04 13:46:00', 'cmdb_id': 'MG02'},
        {'timestamp': '2021-03-04 13:37:00', 'cmdb_id': 'MG02'},
        {'timestamp': '2021-03-04 13:52:00', 'cmdb_id': 'MG02'},
        {'timestamp': '2021-03-04 13:56:00', 'cmdb_id': 'IG02'},
        {'timestamp': '2021-03-04 13:36:00', 'cmdb_id': 'MG01'},
        {'timestamp': '2021-03-04 13:43:00', 'cmdb_id': 'IG01'},
        {'timestamp': '2021-03-04 13:57:00', 'cmdb_id': 'IG01'},
        {'timestamp': '2021-03-04 13:48:00', 'cmdb_id': 'MG02'}
    ]
    
    components_to_scan = ['MG02', 'MG01', 'IG01', 'IG02', 'Tomcat*', 'Mysql02']
    
    keywords_to_scan = ['ERROR', 'Exception', 'OutOfMemory', 'OOM', 'GC', 'CMS Initial Mark', 'Connection reset', 'timeout', 'disk full']

    log_anomaly_events = find_root_cause_from_logs_7c3dab74(
        start_time_str=start_time,
        end_time_str=end_time,
        date_str=date,
        metric_anomalies=metric_anomalies_input,
        target_components=components_to_scan,
        keywords=keywords_to_scan,
        snippet_window_minutes=1
    )

    if log_anomaly_events:
        df_output = pd.DataFrame(log_anomaly_events)
        print("anomaly_events = ")
        print(df_output.to_string())
        df_output.to_csv("anomaly_events.csv", index=False)
    else:
        print("anomaly_events = []")



import pandas as pd
import os
import pytz
from datetime import datetime
import re
from typing import Annotated, List, Dict, Any
def search_log_patterns_457b80b8(
    start_time_str: Annotated[str, "The start time for the analysis window in 'YYYY-MM-DD HH:MM:SS' format (UTC+8), e.g., '2021-03-04 17:45:00'"],
    end_time_str: Annotated[str, "The end time for the analysis window in 'YYYY-MM-DD HH:MM:SS' format (UTC+8), e.g., '2021-03-04 18:45:00'"],
    date_str: Annotated[str, "The date string for locating the data directory in 'YYYY_MM_DD' format, e.g., '2021_03_04'"],
    target_cmdb_ids: Annotated[List[str], "A list of specific component IDs (cmdb_id) to filter the logs for, e.g., ['Redis02', 'MG01']"],
    patterns: Annotated[List[str], "A list of keywords or regex patterns to search for within the log values, e.g., ['OOM', 'error']"]
) -> List[Dict[str, Any]]:
    """
    Tool description:
    This tool ingests log data from a specified date, filters it by a given time window and a list of component IDs (cmdb_id),
    and searches for specific keywords or patterns within the log messages. It then generates a list of anomaly events
    for each log entry that matches any of the provided patterns.

    Parameters:
    - start_time_str: The start of the time window for log analysis, in 'YYYY-MM-DD HH:MM:SS' format (UTC+8).
    - end_time_str: The end of the time window for log analysis, in 'YYYY-MM-DD HH:MM:SS' format (UTC+8).
    - date_str: The date used to construct the file path to the log data.
    - target_cmdb_ids: The list of components to be included in the analysis.
    - patterns: The list of string patterns to search for in the log 'value' column.

    Returns:
    - A list of dictionaries, where each dictionary represents a detected anomaly event.
      Returns an empty list if no matching log entries are found.
    """
    data_source_path = './coding/dataset/Bank/telemetry'
    log_file_path = os.path.join(data_source_path, date_str, 'log', 'log_service.csv')

    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found at the specified path: {log_file_path}")

    df = pd.read_csv(log_file_path)

    # Define the UTC+8 timezone
    tz = pytz.timezone('Asia/Shanghai')

    # Convert start and end time strings to timezone-aware datetime objects
    start_time = tz.localize(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
    end_time = tz.localize(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))

    # Convert timezone-aware datetime objects to UTC timestamps in seconds
    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    # Filter by timestamp
    df_filtered = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]

    # Filter by cmdb_id
    df_filtered = df_filtered[df_filtered['cmdb_id'].isin(target_cmdb_ids)]

    if df_filtered.empty:
        return []

    # Combine patterns into a single regex with case-insensitivity
    regex_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    
    anomaly_events = []

    for _, row in df_filtered.iterrows():
        log_value = str(row['value'])
        match = regex_pattern.search(log_value)
        if match:
            # Convert UTC timestamp to UTC+8 datetime string
            utc_dt = datetime.fromtimestamp(row['timestamp'], tz=pytz.utc)
            shanghai_dt = utc_dt.astimezone(tz)
            timestamp_str = shanghai_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Create a snippet of the log for the description
            snippet_length = 200
            log_snippet = (log_value[:snippet_length] + '...') if len(log_value) > snippet_length else log_value

            event = {
                "data_source": "Log",
                "timestamp": timestamp_str,
                "cmdb_id": row['cmdb_id'],
                "description": f"Matched pattern '{match.group(0)}' found in log. Snippet: {log_snippet}"
            }
            anomaly_events.append(event)

    return anomaly_events
if __name__ == '__main__':
    # Parameters based on the task description
    start_time = '2021-03-04 17:45:00'
    end_time = '2021-03-04 18:45:00'
    date = '2021_03_04'
    cmdb_ids = ['Redis02', 'MG01', 'Tomcat03', 'IG02', 'MG02', 'IG01']
    search_patterns = [
        'OOM', 'OutOfMemory', 'GC', 'Full GC', 'heap', 'Connection refused',
        'timeout', 'socket', 'no route', 'disk full', 'IO error', 'exception', 'error'
    ]

    # Execute the function and print the results
    detected_anomalies = search_log_patterns_457b80b8(
        start_time_str=start_time,
        end_time_str=end_time,
        date_str=date,
        target_cmdb_ids=cmdb_ids,
        patterns=search_patterns
    )
    
    print(detected_anomalies)


