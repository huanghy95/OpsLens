import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import json
import pandas as pd
from functools import lru_cache
import warnings
from tqdm import tqdm
import os

class DataConverter:
    def __init__(self, base_path, record_path):
        self.base_path = base_path
        self.record_path = record_path
        self.tz = pytz.timezone('Asia/Shanghai')
        
        # File paths - will be set dynamically in process_single_record based on date
        self.trace_path = None
        self.metric_container_path = None
        self.metric_app_path = None
        self.metric_service_path = None
        self.metric_middleware_path = None
        
        # Data cache
        self.trace_df = None
        self.metric_container_df = None
        self.metric_app_df = None
        self.metric_service_df = None
        self.metric_middleware_df = None
        
        # Fault time windows
        self.time_windows = []
        
        # Cached grouped data
        self._metric_container_groups = None
        self._metric_service_groups = None
        
        # Cache for time window processing results
        self._time_window_cache = {}
        
        # Key KPI indicators list, based on common faults and indicator mapping in telecom scenarios
        self.key_kpis = {
            'metric_container': [
                'container_cpu_used',
                'container_mem_used'
            ],
            'metric_service': [
                'Proc_User_Used_Pct',
                'Login_Per_Sec',
                'On_Off_State'
            ],
            'metric_app': [
                'succee_rate',
                'avg_time'
            ]
        }
        
    def set_data_paths(self, date_str):
        """Set data file paths based on date"""
        date_path = os.path.join(self.base_path, date_str)
        if not os.path.exists(date_path):
            raise ValueError(f"Cannot find data directory for date {date_str}: {date_path}")
            
        self.trace_path = os.path.join(date_path, "trace/trace_span.csv")
        self.metric_container_path = os.path.join(date_path, "metric/metric_container.csv")
        self.metric_app_path = os.path.join(date_path, "metric/metric_app.csv")
        self.metric_service_path = os.path.join(date_path, "metric/metric_service.csv")
        self.metric_middleware_path = os.path.join(date_path, "metric/metric_middleware.csv")
        
        # Check if files exist
        for path in [self.trace_path, self.metric_container_path, self.metric_app_path, 
                    self.metric_service_path, self.metric_middleware_path]:
            if not os.path.exists(path):
                raise ValueError(f"Cannot find data file: {path}")
                
        print(f"\nUsing data directory: {date_path}")
        print(f"Trace file: {self.trace_path}")
        print(f"Metric file: {self.metric_container_path}")

    def filter_by_time_windows(self, df, time_col='timestamp'):
        """Efficiently filter data within time windows"""
        if len(self.time_windows) == 0:
            return df
            
        mask = pd.Series(False, index=df.index)
        for start, end in self.time_windows:
            # Convert timezone-aware datetime to naive datetime for comparison
            window_mask = (df[time_col].dt.tz_localize(None) >= start) & (df[time_col].dt.tz_localize(None) <= end)
            mask = mask | window_mask
            
        return df[mask]
        
    def load_data(self):
        """Load all data files, only load data within time windows"""
        print("\n=== Start loading data ===")
        
        # Load trace data
        print("Loading trace data...")
        self.trace_df = pd.read_csv(self.trace_path, low_memory=False)
        # Convert UTC timestamp to local time (UTC+8)
        self.trace_df['start_time'] = pd.to_datetime(self.trace_df['startTime'], unit='ms', utc=True)
        self.trace_df['start_time'] = self.trace_df['start_time'].dt.tz_convert('Asia/Shanghai')
        self.trace_df['end_time'] = self.trace_df['start_time'] + pd.to_timedelta(self.trace_df['elapsedTime'], unit='ms')
        
        # Filter data within time windows
        mask = pd.Series(False, index=self.trace_df.index)
        for start, end in self.time_windows:
            # Convert timezone-aware datetime to naive datetime for comparison
            window_mask = ((self.trace_df['start_time'].dt.tz_localize(None) >= start) & 
                         (self.trace_df['start_time'].dt.tz_localize(None) <= end)) | \
                        ((self.trace_df['end_time'].dt.tz_localize(None) >= start) & 
                         (self.trace_df['end_time'].dt.tz_localize(None) <= end))
            mask = mask | window_mask
        self.trace_df = self.trace_df[mask]
        print(f"Trace data loaded, total {len(self.trace_df)} records")
        
        # Load metric_container data
        print("\nLoading metric_container data...")
        self.metric_container_df = pd.read_csv(self.metric_container_path, low_memory=False)
        # First filter key indicators
        self.metric_container_df = self.metric_container_df[
            self.metric_container_df['name'].isin(self.key_kpis['metric_container'])
        ]
        # Then convert timestamp and filter by time window
        self.metric_container_df['timestamp'] = pd.to_datetime(self.metric_container_df['timestamp'], unit='ms', utc=True)
        self.metric_container_df['timestamp'] = self.metric_container_df['timestamp'].dt.tz_convert('Asia/Shanghai')
        self.metric_container_df = self.filter_by_time_windows(self.metric_container_df)
        print(f"Metric_container data loaded, total {len(self.metric_container_df)} records")
        
        # Load metric_app data
        print("\nLoading metric_app data...")
        self.metric_app_df = pd.read_csv(self.metric_app_path, low_memory=False)
        print("Metric_app data columns:", self.metric_app_df.columns.tolist())
        
        # First convert timestamp
        self.metric_app_df['timestamp'] = pd.to_datetime(self.metric_app_df['startTime'], unit='ms', utc=True)
        self.metric_app_df['timestamp'] = self.metric_app_df['timestamp'].dt.tz_convert('Asia/Shanghai')
        
        # Then filter by time window
        self.metric_app_df = self.filter_by_time_windows(self.metric_app_df)
        
        # Finally filter key indicators, but keep timestamp column
        key_metrics = self.key_kpis['metric_app'] + ['timestamp']
        self.metric_app_df = self.metric_app_df[key_metrics]
        
        print(f"Metric_app data loaded, total {len(self.metric_app_df)} records")
        
        # Load metric_service data
        print("\nLoading metric_service data...")
        self.metric_service_df = pd.read_csv(self.metric_service_path, low_memory=False)
        # First filter key indicators
        self.metric_service_df = self.metric_service_df[
            self.metric_service_df['name'].isin(self.key_kpis['metric_service'])
        ]
        # Then convert timestamp and filter by time window
        self.metric_service_df['timestamp'] = pd.to_datetime(self.metric_service_df['timestamp'], unit='ms', utc=True)
        self.metric_service_df['timestamp'] = self.metric_service_df['timestamp'].dt.tz_convert('Asia/Shanghai')
        self.metric_service_df = self.filter_by_time_windows(self.metric_service_df)
        print(f"Metric_service data loaded, total {len(self.metric_service_df)} records")
        
        # Preprocessing: group by cmdb_id to speed up subsequent operations
        print("\nPreprocessing data...")
        self._metric_container_groups = dict(list(self.metric_container_df.groupby('cmdb_id')))
        self._metric_service_groups = dict(list(self.metric_service_df.groupby('cmdb_id')))
        print("Data preprocessing completed")
        
    def get_metrics_for_time_window(self, start_time, end_time, cmdb_id, ds_name):
        """Get indicator data for the specified time window, optimized with cache"""
        # Generate cache key, convert datetime to timestamp
        cache_key = (
            start_time.timestamp(),
            end_time.timestamp(),
            str(cmdb_id),
            str(ds_name) if ds_name else ''
        )
        if cache_key in self._time_window_cache:
            return self._time_window_cache[cache_key]
        
        result = {}
        
        # Process container indicators
        if cmdb_id in self._metric_container_groups:
            container_metrics = self._metric_container_groups[cmdb_id]
            span_metrics = container_metrics[
                (container_metrics['timestamp'] >= start_time) &
                (container_metrics['timestamp'] <= end_time)
            ]
            if not span_metrics.empty:
                for metric_name, group in span_metrics.groupby('name'):
                    result[f'metric@{metric_name}'] = group['value'].mean()
        
        # Process service indicators - use dsName to match
        if ds_name:
            matching_services = self.metric_service_df[self.metric_service_df['cmdb_id'] == ds_name]
            if not matching_services.empty:
                span_metrics = matching_services[
                    (matching_services['timestamp'] >= start_time) &
                    (matching_services['timestamp'] <= end_time)
                ]
                if not span_metrics.empty:
                    for metric_name, group in span_metrics.groupby('name'):
                        result[f'metric@{metric_name}'] = group['value'].mean()
        
        # Process app indicators
        span_app_metrics = self.metric_app_df[
            (self.metric_app_df['timestamp'] >= start_time) &
            (self.metric_app_df['timestamp'] <= end_time)
        ]
        if not span_app_metrics.empty:
            for metric in self.key_kpis['metric_app']:
                if metric in span_app_metrics.columns:
                    result[f'metric@{metric}'] = span_app_metrics[metric].mean()
        
        # Cache result
        self._time_window_cache[cache_key] = result
        return result

    def batch_merge_metrics(self):
        """Batch process indicator data"""
        print("\nBatch merging indicator data...")
        
        # Create result DataFrame
        result_df = pd.DataFrame(index=self.trace_df.index)
        
        # Process each span
        for idx, span in tqdm(self.trace_df.iterrows(), total=len(self.trace_df), desc="Merging indicator data"):
            start_time = span['start_time']
            end_time = span['end_time']
            cmdb_id = span['cmdb_id']
            ds_name = span['dsName']
            
            # Get indicator data for this time window (using cache)
            metrics = self.get_metrics_for_time_window(start_time, end_time, cmdb_id, ds_name)
            
            # Add indicator data to result DataFrame
            for metric_name, value in metrics.items():
                result_df.at[idx, metric_name] = value
        
        # Fill missing values with 0
        result_df = result_df.fillna(0)
        return result_df
    
    def process_single_record(self, record, record_idx):
        """Process a single fault record"""
        print(f"\n=== Processing fault record {record_idx + 1} ===")
        print(f"Fault level: {record['level']}")
        print(f"Fault reason: {record['reason']}")
        print(f"Fault component: {record['component']}")
        print(f"Fault time: {record['datetime']}")
        
        # Set time window
        fault_time = pd.to_datetime(record['datetime'])
        window_start = fault_time - timedelta(
            minutes=fault_time.minute % 30,
            seconds=fault_time.second,
            microseconds=fault_time.microsecond
        )
        window_end = window_start + timedelta(minutes=30)
        self.time_windows = [(window_start, window_end)]
        print(f"Processing time window: {window_start} to {window_end}")
        
        # Set data path based on fault time
        date_str = fault_time.strftime('%Y_%m_%d')
        try:
            self.set_data_paths(date_str)
        except ValueError as e:
            print(f"Error: {str(e)}")
            return None
        
        # Load data
        self.load_data()
        
        print("\n=== Start merging data ===")
        # Create result DataFrame
        result_df = self.trace_df.copy()
        total_spans = len(result_df)
        
        if total_spans == 0:
            print(f"Warning: Fault record {record_idx + 1} has no relevant data in the time window")
            return None
        
        print(f"Found {total_spans} relevant span data")
        
        # Batch merge indicator data
        metric_data = self.batch_merge_metrics()
        for col in metric_data.columns:
            result_df[col] = metric_data[col]
        print(f"Indicator data merged, {len(metric_data.columns)} indicator columns added")
        
        print("\nProcessing final data...")
        # Add other necessary columns
        result_df['id'] = range(1, len(result_df) + 1)
        result_df['trace_rowid'] = result_df['id']
        result_df['span_id'] = result_df['id']  # Add span_id column
        result_df['start_time_unix'] = result_df['start_time'].dt.tz_localize(None).astype(np.int64) // 10**9
        result_df['end_time_unix'] = result_df['end_time'].dt.tz_localize(None).astype(np.int64) // 10**9
        
        # Map cmdb_id to name
        result_df['name'] = result_df['cmdb_id']
        
        # Rename pid to parent_id
        result_df = result_df.rename(columns={'pid': 'parent_id'})
        
        # Add events, status_code, and status_message columns
        result_df['events'] = '[]'
        result_df['status_code'] = result_df['success'].map({'True': 0, 'False': 1}).fillna(0)
        result_df['status_message'] = ''
        
        # Select and reorder columns
        columns = ['id', 'trace_rowid', 'span_id', 'parent_id', 'name', 'start_time', 'end_time',
                  'events', 'status_code', 'status_message', 'start_time_unix', 'end_time_unix']
        
        # Add all indicator columns
        metric_columns = [col for col in result_df.columns if col.startswith('metric@')]
        columns.extend(metric_columns)
        
        # Add elapsed time column
        columns.extend(['elapsedTime'])
        
        # Reorder
        result_df = result_df[columns]
        
        # Create output directory
        output_dir = "YOURPATH/Telecom/datafinal"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate file name: use record row number, fault level, component, and time as file name
        file_name = f"{record_idx}_{record['level']}_{record['component']}_{fault_time.strftime('%Y%m%d_%H%M%S')}_datafinal.csv"
        output_filename = os.path.join(output_dir, file_name)
        print(f"\nSaving result to {output_filename}...")
        result_df.to_csv(output_filename, index=False)
        print(f"Saved {len(result_df)} records to {output_filename}")
        
        print("\n=== Data conversion completed ===")
        print(f"Total processed spans: {total_spans}")
        print(f"Final data columns: {len(columns)}")
        print(f"Indicator columns: {len(metric_columns)}")
        
        return result_df

    def convert(self):
        """Execute data conversion"""
        print("\n=== Start data conversion ===")
        
        # Read all fault records
        records_df = pd.read_csv(self.record_path)
        records_df['datetime'] = pd.to_datetime(records_df['datetime'])
        total_records = len(records_df)
        print(f"Total {total_records} fault records read")
        
        # Process fault records one by one
        processed_records = 0
        successful_records = 0
        failed_records = 0
        
        for idx, record in records_df.iterrows():
            processed_records += 1
            print(f"\n=== Progress: {processed_records}/{total_records} ({processed_records/total_records*100:.1f}%) ===")
            
            try:
                result = self.process_single_record(record, idx)
                if result is not None:
                    successful_records += 1
                else:
                    failed_records += 1
            except Exception as e:
                print(f"Error occurred while processing fault record {idx + 1}: {str(e)}")
                failed_records += 1
            
            # Output current statistics
            print(f"\nCurrent statistics:")
            print(f"Total records: {total_records}")
            print(f"Processed: {processed_records}")
            print(f"Successfully processed: {successful_records}")
            print(f"Failed: {failed_records}")
            print(f"Remaining: {total_records - processed_records}")
            
        print("\n=== All fault records processed ===")
        print(f"Total records: {total_records}")
        print(f"Successfully processed: {successful_records}")
        print(f"Failed: {failed_records}")
        print(f"Success rate: {successful_records/total_records*100:.1f}%")

if __name__ == "__main__":
    # Usage example
    import os

    # Set paths
    record_path = "YOURPATH/Telecom/record.csv"
    base_path_root = "YOURPATH/Telecom/telemetry/"
    
    # Create converter instance and execute conversion
    converter = DataConverter(
        base_path=base_path_root,
        record_path=record_path
    )
    converter.convert()
