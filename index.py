import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil 
import platform
from functools import partial
import warnings
import threading
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize the performance analyzer with dataset"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.results = {}
        self.load_data()
        self.print_system_specs()
    
    def load_data(self):
        """Load the CSV file and prepare the trip_duration column"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_file_path)
        print(f"Dataset loaded successfully: {len(self.df)} rows")
        print(f"Trip duration stats:\n{self.df['trip_duration'].describe()}")

    def print_system_specs(self):
        """Print system specifications"""
        print("\n" + "="*50)
        print("SYSTEM SPECIFICATIONS")
        print("="*50)
        print(f"Processor: {platform.processor()}")
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.architecture()[0]}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        print(f"Python Version: {platform.python_version()}")
        print("="*50)
    
    def monitor_cpu_usage(self, duration, interval=0.1):
        """Monitor CPU usage during execution"""
        cpu_percentages = []
        memory_percentages = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percentages.append(psutil.cpu_percent(interval=None))
            memory_percentages.append(psutil.virtual_memory().percent)
            time.sleep(interval)
        
        return {
            'avg_cpu': np.mean(cpu_percentages),
            'max_cpu': np.max(cpu_percentages),
            'min_cpu': np.min(cpu_percentages),
            'avg_memory': np.mean(memory_percentages),
            'max_memory': np.max(memory_percentages)
        }
    
    def run_with_monitoring(self, func, *args, **kwargs):
        """Run a function while monitoring system resources"""
        psutil.cpu_percent() 
        time.sleep(0.1) 
        
        monitoring_active = threading.Event()
        monitoring_active.set()
        cpu_data = []
        memory_data = []
        
        def monitor():
            while monitoring_active.is_set():
                cpu_data.append(psutil.cpu_percent(interval=None))
                memory_data.append(psutil.virtual_memory().percent)
                time.sleep(0.05) 
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        monitoring_active.clear()
        monitor_thread.join()
        
        execution_time = end_time - start_time
        cpu_stats = {
            'avg_cpu': np.mean(cpu_data) if cpu_data else 0,
            'max_cpu': np.max(cpu_data) if cpu_data else 0,
            'min_cpu': np.min(cpu_data) if cpu_data else 0,
            'avg_memory': np.mean(memory_data) if memory_data else 0,
            'max_memory': np.max(memory_data) if memory_data else 0
        }
        
        return result, execution_time, cpu_stats
    
    def sequential_sort(self, data):
        """Sequential sorting"""
        return sorted(data)
    
    def sequential_filter(self, data, threshold=1000):
        """Sequential filtering"""
        return [x for x in data if x > threshold]
    
    def sequential_process(self, data, threshold=1000):
        """Combined sequential sorting and filtering"""
        filtered_data = self.sequential_filter(data, threshold)
        sorted_data = self.sequential_sort(filtered_data)
        return sorted_data
    
    def thread_sort_chunk(self, chunk):
        """Sort a chunk of data in a thread"""
        return sorted(chunk)
    
    def thread_filter_chunk(self, chunk, threshold):
        """Filter a chunk of data in a thread"""
        return [x for x in chunk if x > threshold]
    
    def threading_process(self, data, threshold=1000, n_threads=4):
        """Process data using threading"""
        chunk_size = len(data) // n_threads
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            filter_func = partial(self.thread_filter_chunk, threshold=threshold)
            filtered_chunks = list(executor.map(filter_func, chunks))
        
        filtered_data = []
        for chunk in filtered_chunks:
            filtered_data.extend(chunk)
        
        if filtered_data:
            chunk_size = len(filtered_data) // n_threads
            sort_chunks = [filtered_data[i:i + chunk_size] for i in range(0, len(filtered_data), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                sorted_chunks = list(executor.map(self.thread_sort_chunk, sort_chunks))
            
            result = []
            for chunk in sorted_chunks:
                result.extend(chunk)
            return sorted(result)  
        
        return []
    
    def mp_sort_chunk(self, chunk):
        """Sort a chunk of data in a process"""
        return sorted(chunk)
    
    def mp_filter_chunk(self, args):
        """Filter a chunk of data in a process"""
        chunk, threshold = args
        return [x for x in chunk if x > threshold]
    
    def multiprocessing_process(self, data, threshold=1000, n_processes=None):
        """Process data using multiprocessing"""
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        chunk_size = len(data) // n_processes
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            filter_args = [(chunk, threshold) for chunk in chunks]
            filtered_chunks = list(executor.map(self.mp_filter_chunk, filter_args))
        
        filtered_data = []
        for chunk in filtered_chunks:
            filtered_data.extend(chunk)
        
        if filtered_data:
            chunk_size = len(filtered_data) // n_processes
            sort_chunks = [filtered_data[i:i + chunk_size] for i in range(0, len(filtered_data), chunk_size)]
            
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                sorted_chunks = list(executor.map(self.mp_sort_chunk, sort_chunks))
            
            result = []
            for chunk in sorted_chunks:
                result.extend(chunk)
            return sorted(result) 
        
        return []
    
    def run_performance_test(self, data_percentages=[25, 50, 75, 100], threshold=1000, n_runs=3):
        """Run performance tests for different data sizes and approaches with CPU monitoring"""
        print(f"\nRunning performance tests with threshold > {threshold}")
        print(f"Data percentages: {data_percentages}%")
        print(f"Number of runs per test: {n_runs}")
        
        trip_data = self.df['trip_duration'].tolist()
        results = {
            'data_size': [],
            'approach': [],
            'time_taken': [],
            'data_points': [],
            'results_count': [],
            'avg_cpu_usage': [],
            'max_cpu_usage': [],
            'min_cpu_usage': [],
            'avg_memory_usage': [],
            'max_memory_usage': []
        }
        
        for percentage in data_percentages:
            data_size = int(len(trip_data) * percentage / 100)
            current_data = trip_data[:data_size]
            
            print(f"\nTesting with {percentage}% of data ({data_size:,} records)")
            
            for approach_name, approach_func in [
                ('Sequential', lambda d: self.sequential_process(d, threshold)),
                ('Threading', lambda d: self.threading_process(d, threshold)),
                ('Multiprocessing', lambda d: self.multiprocessing_process(d, threshold))
            ]:
                times = []
                result_counts = []
                cpu_stats_list = []
                
                for run in range(n_runs):
                    result, execution_time, cpu_stats = self.run_with_monitoring(
                        approach_func, current_data.copy()
                    )
                    
                    times.append(execution_time)
                    result_counts.append(len(result))
                    cpu_stats_list.append(cpu_stats)
                    
                    print(f"  {approach_name} - Run {run+1}: {execution_time:.4f}s "
                          f"({len(result):,} results) - "
                          f"CPU: {cpu_stats['avg_cpu']:.1f}% avg, "
                          f"{cpu_stats['max_cpu']:.1f}% max")
                
                avg_time = np.mean(times)
                avg_results = np.mean(result_counts)
                avg_cpu_stats = {
                    'avg_cpu': np.mean([stats['avg_cpu'] for stats in cpu_stats_list]),
                    'max_cpu': np.mean([stats['max_cpu'] for stats in cpu_stats_list]),
                    'min_cpu': np.mean([stats['min_cpu'] for stats in cpu_stats_list]),
                    'avg_memory': np.mean([stats['avg_memory'] for stats in cpu_stats_list]),
                    'max_memory': np.mean([stats['max_memory'] for stats in cpu_stats_list])
                }
                
                results['data_size'].append(percentage)
                results['approach'].append(approach_name)
                results['time_taken'].append(avg_time)
                results['data_points'].append(data_size)
                results['results_count'].append(avg_results)
                results['avg_cpu_usage'].append(avg_cpu_stats['avg_cpu'])
                results['max_cpu_usage'].append(avg_cpu_stats['max_cpu'])
                results['min_cpu_usage'].append(avg_cpu_stats['min_cpu'])
                results['avg_memory_usage'].append(avg_cpu_stats['avg_memory'])
                results['max_memory_usage'].append(avg_cpu_stats['max_memory'])
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def print_detailed_results(self):
        """Print detailed results including CPU usage"""
        if self.results is None or self.results.empty:
            print("No results to display. Run performance tests first.")
            return
        
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE RESULTS")
        print("="*80)
        
        for percentage in sorted(self.results['data_size'].unique()):
            print(f"\n--- {percentage}% of data ---")
            subset = self.results[self.results['data_size'] == percentage]
            
            for _, row in subset.iterrows():
                print(f"{row['approach']:<15}: "
                      f"Time: {row['time_taken']:.4f}s, "
                      f"Results: {int(row['results_count']):,}, "
                      f"CPU Avg: {row['avg_cpu_usage']:.1f}%, "
                      f"CPU Max: {row['max_cpu_usage']:.1f}%, "
                      f"Memory Avg: {row['avg_memory_usage']:.1f}%")
    

def main():
    """Main function to run the performance analysis"""
    csv_file_path = "train.csv" 
    
    print("Performance Analysis: Sequential vs Threading vs Multiprocessing")
    print("=" * 70)
    
    analyzer = PerformanceAnalyzer(csv_file_path)
    
    results = analyzer.run_performance_test(
        data_percentages=[25, 50, 75, 100],
        threshold=1000,  
        n_runs=1 
    )
    
    print("\nPerformance Test Results:")
    analyzer.print_detailed_results()


if __name__ == "__main__":
    main()