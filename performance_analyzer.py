#!/usr/bin/env python3
"""
PixTrick ML-Driven Performance Analyzer
Intelligent benchmarking, trend analysis, and performance optimization
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import argparse

# Import our performance tracking system
from performance_data_schema import (
    PerformanceDatabase, BenchmarkCollector, BenchmarkResult,
    ImageMetadata, FilterConfig, PerformanceMetrics, SystemInfo
)

try:
    import pixtrick_engine as engine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("⚠️  PixTrick engine not available. Run: cd src/engine && maturin develop")

class ImageGenerator:
    """Generate test images with different characteristics"""
    
    @staticmethod
    def create_test_image(width: int, height: int, pattern: str = "gradient") -> 'engine.PixImage':
        """Create test images with different complexity patterns"""
        if not ENGINE_AVAILABLE:
            raise RuntimeError("Engine not available")
        
        img = engine.PixImage(width, height)
        
        if pattern == "gradient":
            # Simple gradient - low complexity
            for y in range(height):
                for x in range(width):
                    intensity = int(255 * (x + y) / (width + height))
                    img.set_pixel(x, y, intensity, intensity, intensity, 255)
        
        elif pattern == "checkerboard":
            # Checkerboard - medium complexity
            for y in range(height):
                for x in range(width):
                    if (x // 10 + y // 10) % 2:
                        img.set_pixel(x, y, 255, 255, 255, 255)
                    else:
                        img.set_pixel(x, y, 0, 0, 0, 255)
        
        elif pattern == "noise":
            # Random noise - high complexity
            import random
            for y in range(height):
                for x in range(width):
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    img.set_pixel(x, y, r, g, b, 255)
        
        elif pattern == "photo_sim":
            # Simulate photo-like content
            for y in range(height):
                for x in range(width):
                    # Create some photo-like regions
                    center_x, center_y = width // 2, height // 2
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    
                    # Sky-like region
                    if y < height * 0.3:
                        r, g, b = 135, 206, 235  # Sky blue
                    # Ground-like region
                    elif y > height * 0.7:
                        r, g, b = 34, 139, 34    # Forest green
                    # Subject in middle
                    elif dist < min(width, height) * 0.15:
                        r, g, b = 255, 218, 185  # Skin tone
                    else:
                        # Blend regions
                        factor = y / height
                        r = int(135 + factor * (34 - 135))
                        g = int(206 + factor * (139 - 206))
                        b = int(235 + factor * (34 - 235))
                    
                    img.set_pixel(x, y, r, g, b, 255)
        
        return img
    
    @staticmethod
    def calculate_complexity_score(img: 'engine.PixImage') -> float:
        """Calculate a complexity score for the image (0.0 = simple, 1.0 = complex)"""
        # Simple heuristic: sample pixels and measure variance
        sample_size = min(1000, img.width * img.height // 100)
        pixels = []
        
        step_x = max(1, img.width // int(sample_size ** 0.5))
        step_y = max(1, img.height // int(sample_size ** 0.5))
        
        for y in range(0, img.height, step_y):
            for x in range(0, img.width, step_x):
                r, g, b, a = img.get_pixel(x, y)
                # Convert to grayscale for simplicity
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                pixels.append(gray)
        
        if len(pixels) < 2:
            return 0.0
        
        # Calculate variance and normalize
        variance = np.var(pixels)
        # Normalize to 0-1 range (255^2 is max variance for 8-bit)
        complexity = min(1.0, variance / (255 ** 2 * 0.25))
        
        return complexity

class PerformanceBenchmarker:
    """Advanced benchmarking with ML-ready data collection"""
    
    def __init__(self):
        self.db = PerformanceDatabase()
        self.collector = BenchmarkCollector(self.db)
        self.image_generator = ImageGenerator()
    
    def benchmark_single_filter(self, 
                               filter_type: str, 
                               params: Dict[str, float],
                               image: 'engine.PixImage',
                               n_runs: int = 5) -> PerformanceMetrics:
        """Benchmark a single filter with detailed metrics"""
        
        times = []
        memory_usage = []
        
        # Create node graph
        graph = engine.NodeGraph()
        graph.set_source_image(image)
        
        # Create filter node
        node = engine.FilterNode("benchmark", filter_type)
        for key, value in params.items():
            node.set_parameter(key, value)
        graph.add_node(node)
        
        # Warm up
        for _ in range(2):
            try:
                graph.clear_cache()
                result = graph.process()
            except Exception:
                pass
        
        # Benchmark runs
        for run in range(n_runs):
            graph.clear_cache()
            
            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure time
            start_time = time.perf_counter()
            try:
                result = graph.process()
                success = True
            except Exception as e:
                print(f"❌ Filter {filter_type} failed: {e}")
                success = False
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            if success:
                times.append((end_time - start_time) * 1000)  # Convert to ms
                memory_usage.append(memory_after - memory_before)
        
        if not times:
            raise RuntimeError(f"All benchmark runs failed for {filter_type}")
        
        # Calculate statistics
        avg_time = np.mean(times)
        memory_peak = max(memory_usage) if memory_usage else 0.0
        memory_avg = np.mean(memory_usage) if memory_usage else 0.0
        
        return PerformanceMetrics(
            execution_time_ms=avg_time,
            memory_peak_mb=memory_peak,
            memory_allocated_mb=memory_avg,
            cpu_usage_percent=0.0,  # Would need separate monitoring
            cache_hits=0,  # Would need instrumentation
            cache_misses=0,
            filter_times_ms={filter_type: avg_time}
        )
    
    def benchmark_filter_chain(self,
                              filters: List[Tuple[str, Dict[str, float]]],
                              image: 'engine.PixImage',
                              n_runs: int = 3) -> Tuple[PerformanceMetrics, List[FilterConfig]]:
        """Benchmark a chain of filters"""
        
        # Create filter chain
        filter_configs = []
        graph = engine.NodeGraph()
        graph.set_source_image(image)
        
        for i, (filter_type, params) in enumerate(filters):
            node_id = f"filter_{i}_{filter_type}"
            node = engine.FilterNode(node_id, filter_type)
            
            for key, value in params.items():
                node.set_parameter(key, value)
            
            graph.add_node(node)
            
            filter_configs.append(FilterConfig(
                filter_type=filter_type,
                parameters=params,
                enabled=True,
                order_index=i
            ))
        
        # Benchmark the complete chain
        times = []
        memory_usage = []
        individual_times = {}
        
        for run in range(n_runs):
            graph.clear_cache()
            
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            try:
                result = graph.process()
                success = True
            except Exception as e:
                print(f"❌ Filter chain failed: {e}")
                success = False
            end_time = time.perf_counter()
            
            memory_after = process.memory_info().rss / 1024 / 1024
            
            if success:
                total_time = (end_time - start_time) * 1000
                times.append(total_time)
                memory_usage.append(memory_after - memory_before)
                
                # Estimate individual filter times (rough approximation)
                time_per_filter = total_time / len(filters)
                for filter_type, _ in filters:
                    if filter_type not in individual_times:
                        individual_times[filter_type] = []
                    individual_times[filter_type].append(time_per_filter)
        
        if not times:
            raise RuntimeError("All benchmark runs failed for filter chain")
        
        # Calculate final individual times
        final_individual_times = {}
        for filter_type, filter_times in individual_times.items():
            final_individual_times[filter_type] = np.mean(filter_times)
        
        metrics = PerformanceMetrics(
            execution_time_ms=np.mean(times),
            memory_peak_mb=max(memory_usage) if memory_usage else 0.0,
            memory_allocated_mb=np.mean(memory_usage) if memory_usage else 0.0,
            cpu_usage_percent=0.0,
            cache_hits=0,
            cache_misses=0,
            filter_times_ms=final_individual_times
        )
        
        return metrics, filter_configs
    
    def comprehensive_benchmark_suite(self, save_results: bool = True):
        """Run comprehensive benchmarks across different scenarios"""
        
        if not ENGINE_AVAILABLE:
            print("❌ Cannot run benchmarks without engine")
            return
        
        print("🚀 Starting Comprehensive Performance Benchmark Suite")
        print("=" * 60)
        
        # Test configurations
        image_sizes = [
            (200, 200, "Small"),
            (800, 600, "Medium"),
            (1920, 1080, "Large"),
            (4000, 3000, "XLarge")
        ]
        
        image_patterns = ["gradient", "checkerboard", "noise", "photo_sim"]
        
        single_filters = [
            ("brightness", {"amount": 20.0}),
            ("contrast", {"amount": 15.0}),
            ("saturation", {"amount": 130.0}),
            ("hue_shift", {"degrees": 45.0}),
            ("box_blur", {"radius": 2.0}),
            ("gaussian_blur", {"radius": 1.5}),
            ("invert", {}),
            ("grayscale", {}),
        ]
        
        filter_chains = [
            [("brightness", {"amount": 10.0}), ("contrast", {"amount": 5.0})],
            [("saturation", {"amount": 120.0}), ("hue_shift", {"degrees": 30.0})],
            [("brightness", {"amount": 15.0}), ("contrast", {"amount": 10.0}), ("saturation", {"amount": 110.0})],
            [("box_blur", {"radius": 1.0}), ("brightness", {"amount": 5.0})],
        ]
        
        total_tests = len(image_sizes) * len(image_patterns) * (len(single_filters) + len(filter_chains))
        test_count = 0
        
        results_summary = []
        
        for width, height, size_name in image_sizes:
            print(f"\n📏 Testing {size_name} Images ({width}x{height})")
            
            for pattern in image_patterns:
                print(f"  🎨 Pattern: {pattern}")
                
                try:
                    # Generate test image
                    img = self.image_generator.create_test_image(width, height, pattern)
                    complexity = self.image_generator.calculate_complexity_score(img)
                    
                    image_meta = ImageMetadata(
                        width=width,
                        height=height,
                        channels=4,
                        file_size_mb=None,  # Generated image
                        format="Generated",
                        complexity_score=complexity
                    )
                    
                    # Test single filters
                    for filter_type, params in single_filters:
                        test_count += 1
                        print(f"    🔧 [{test_count}/{total_tests}] {filter_type}")
                        
                        try:
                            metrics = self.benchmark_single_filter(filter_type, params, img)
                            
                            filter_config = [FilterConfig(
                                filter_type=filter_type,
                                parameters=params,
                                enabled=True,
                                order_index=0
                            )]
                            
                            if save_results:
                                result = self.collector.create_benchmark_result(
                                    image_meta, filter_config, metrics,
                                    test_type="comprehensive_suite",
                                    tags=["single_filter", size_name.lower(), pattern],
                                    notes=f"Single filter benchmark: {filter_type} on {pattern} {size_name}"
                                )
                                self.collector.store_benchmark(result)
                            
                            results_summary.append({
                                'size': size_name,
                                'pattern': pattern,
                                'filter': filter_type,
                                'time_ms': metrics.execution_time_ms,
                                'memory_mb': metrics.memory_peak_mb,
                                'megapixels': image_meta.megapixels(),
                                'complexity': complexity
                            })
                            
                        except Exception as e:
                            print(f"      ❌ Failed: {e}")
                    
                    # Test filter chains
                    for chain in filter_chains:
                        test_count += 1
                        chain_name = " → ".join([f[0] for f in chain])
                        print(f"    ⛓️  [{test_count}/{total_tests}] {chain_name}")
                        
                        try:
                            metrics, filter_configs = self.benchmark_filter_chain(chain, img)
                            
                            if save_results:
                                result = self.collector.create_benchmark_result(
                                    image_meta, filter_configs, metrics,
                                    test_type="comprehensive_suite",
                                    tags=["filter_chain", size_name.lower(), pattern],
                                    notes=f"Filter chain: {chain_name} on {pattern} {size_name}"
                                )
                                self.collector.store_benchmark(result)
                            
                            results_summary.append({
                                'size': size_name,
                                'pattern': pattern,
                                'filter': chain_name,
                                'time_ms': metrics.execution_time_ms,
                                'memory_mb': metrics.memory_peak_mb,
                                'megapixels': image_meta.megapixels(),
                                'complexity': complexity
                            })
                            
                        except Exception as e:
                            print(f"      ❌ Failed: {e}")
                
                except Exception as e:
                    print(f"    ❌ Failed to create {pattern} image: {e}")
        
        print(f"\n✅ Completed {test_count} benchmark tests!")
        return results_summary

class PerformanceAnalyzer:
    """ML-driven analysis of performance data"""
    
    def __init__(self):
        self.db = PerformanceDatabase()
    
    def analyze_performance_trends(self, days: int = 30):
        """Analyze performance trends over time"""
        print(f"\n📈 Performance Trends (Last {days} days)")
        print("-" * 40)
        
        trends = self.db.get_performance_trends(days)
        
        if not trends['dates']:
            print("No data available for trend analysis")
            return
        
        print(f"Date Range: {trends['dates'][0]} to {trends['dates'][-1]}")
        print(f"Average Performance: {np.mean(trends['avg_times']):.1f}ms")
        print(f"Best Performance: {min(trends['min_times']):.1f}ms")
        print(f"Worst Performance: {max(trends['max_times']):.1f}ms")
        
        # Simple trend detection
        if len(trends['avg_times']) >= 2:
            recent_avg = np.mean(trends['avg_times'][-7:])  # Last week
            older_avg = np.mean(trends['avg_times'][:-7])   # Before that
            
            if recent_avg < older_avg:
                improvement = ((older_avg - recent_avg) / older_avg) * 100
                print(f"🚀 Performance IMPROVED by {improvement:.1f}% recently")
            else:
                regression = ((recent_avg - older_avg) / older_avg) * 100
                print(f"⚠️  Performance REGRESSED by {regression:.1f}% recently")
    
    def analyze_by_image_characteristics(self):
        """Analyze performance patterns by image characteristics"""
        print("\n🖼️  Performance by Image Characteristics")
        print("-" * 45)
        
        results = self.db.query_results()
        if not results:
            print("No data available")
            return
        
        df = pd.DataFrame(results)
        
        # Performance by image size
        print("\n📏 Performance by Image Size:")
        size_groups = df.groupby(pd.cut(df['img_megapixels'], bins=[0, 1, 4, 10, float('inf')], 
                                       labels=['<1MP', '1-4MP', '4-10MP', '>10MP']))
        
        for size_range, group in size_groups:
            if len(group) > 0:
                avg_time = group['execution_time_ms'].mean()
                count = len(group)
                print(f"  {size_range}: {avg_time:.1f}ms avg ({count} samples)")
        
        # Performance by complexity
        if 'img_complexity_score' in df.columns and df['img_complexity_score'].notna().any():
            print("\n🎨 Performance by Image Complexity:")
            complexity_groups = df.groupby(pd.cut(df['img_complexity_score'], bins=3, 
                                                 labels=['Low', 'Medium', 'High']))
            
            for complexity, group in complexity_groups:
                if len(group) > 0:
                    avg_time = group['execution_time_ms'].mean()
                    count = len(group)
                    print(f"  {complexity}: {avg_time:.1f}ms avg ({count} samples)")
    
    def analyze_filter_performance(self):
        """Analyze individual filter performance"""
        print("\n🔧 Filter Performance Analysis")
        print("-" * 35)
        
        results = self.db.query_results()
        if not results:
            print("No data available")
            return
        
        # Extract filter information from filter_chain JSON
        filter_stats = {}
        
        for result in results:
            try:
                import json
                filter_chain = json.loads(result['filter_chain'])
                
                for filter_config in filter_chain:
                    filter_type = filter_config['filter_type']
                    if filter_type not in filter_stats:
                        filter_stats[filter_type] = []
                    
                    # Use individual filter time if available, otherwise estimate
                    if result['filter_times_ms']:
                        filter_times = json.loads(result['filter_times_ms'])
                        if filter_type in filter_times:
                            filter_stats[filter_type].append(filter_times[filter_type])
                        else:
                            # Estimate from total time
                            total_filters = len(filter_chain)
                            estimated_time = result['execution_time_ms'] / total_filters
                            filter_stats[filter_type].append(estimated_time)
                    else:
                        # Estimate from total time
                        total_filters = len(filter_chain)
                        estimated_time = result['execution_time_ms'] / total_filters
                        filter_stats[filter_type].append(estimated_time)
            
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Display filter statistics
        print("Filter Performance (Average Times):")
        for filter_type, times in sorted(filter_stats.items()):
            avg_time = np.mean(times)
            count = len(times)
            std_time = np.std(times)
            print(f"  {filter_type:15}: {avg_time:6.1f}ms ± {std_time:4.1f}ms ({count:3d} samples)")
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        print("🏆 PixTrick Performance Analysis Report")
        print("=" * 50)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get basic statistics
        results = self.db.query_results()
        if not results:
            print("\n❌ No benchmark data available!")
            print("Run: python performance_analyzer.py --benchmark")
            return
        
        print(f"\n📊 Dataset Overview:")
        print(f"  Total Benchmark Results: {len(results)}")
        
        df = pd.DataFrame(results)
        print(f"  Date Range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
        print(f"  Test Types: {', '.join(df['test_type'].unique())}")
        
        # Performance overview
        print(f"\n⚡ Performance Overview:")
        print(f"  Average Execution Time: {df['execution_time_ms'].mean():.1f}ms")
        print(f"  Fastest Execution: {df['execution_time_ms'].min():.1f}ms")
        print(f"  Slowest Execution: {df['execution_time_ms'].max():.1f}ms")
        print(f"  Average Memory Usage: {df['memory_peak_mb'].mean():.1f}MB")
        
        # Run detailed analyses
        self.analyze_performance_trends()
        self.analyze_by_image_characteristics()
        self.analyze_filter_performance()
        
        print(f"\n✅ Report complete! Database: {self.db.db_path}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="PixTrick Performance Analyzer")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run comprehensive benchmark suite")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing performance data")
    parser.add_argument("--report", action="store_true",
                       help="Generate performance report")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (faster, less comprehensive)")
    
    args = parser.parse_args()
    
    if not any([args.benchmark, args.analyze, args.report]):
        # Default behavior - run everything
        args.benchmark = True
        args.analyze = True
        args.report = True
    
    benchmarker = PerformanceBenchmarker()
    analyzer = PerformanceAnalyzer()
    
    try:
        if args.benchmark:
            print("🚀 Running Benchmark Suite...")
            benchmarker.comprehensive_benchmark_suite(save_results=True)
        
        if args.analyze or args.report:
            print("\n🔍 Analyzing Performance Data...")
            analyzer.generate_performance_report()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()