"""
Test script for PixTrick Rust engine
Fast functional tests with optional performance tracking
"""

import sys
import time
from pathlib import Path
import traceback

# Import performance tracking 
try:
    from performance_data_schema import (
        PerformanceDatabase, BenchmarkCollector, 
        ImageMetadata, FilterConfig, PerformanceMetrics
    )
    PERFORMANCE_TRACKING = True
except ImportError:
    PERFORMANCE_TRACKING = False

def test_basic_functionality():
    """Test basic image loading and processing"""
    print("=== Testing PixTrick Engine ===\n")
    
    try:
        import pixtrick_engine as engine
        print("✓ Engine imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import engine: {e}")
        print("Run: cd src/engine && maturin develop")
        return False
    
    # Test 1: Create a blank image
    print("\n1. Testing image creation...")
    try:
        img = engine.PixImage(100, 100)
        print(f"  Created image: {img.info()}")
        print(f"  Dimensions: {img.width}x{img.height}, Channels: {img.channels}")
    except Exception as e:
        print(f"  Failed to create image: {e}")
        return False
    
    # Test 2: Pixel operations
    print("\n2. Testing pixel operations...")
    try:
        # Set a red pixel
        img.set_pixel(10, 10, 255, 0, 0, 255)
        r, g, b, a = img.get_pixel(10, 10)
        print(f"✓ Set pixel (10,10) to red: ({r}, {g}, {b}, {a})")
        
        # Test bounds checking
        try:
            img.get_pixel(200, 200)  # Should fail
            print("  Bounds checking failed")
        except Exception:
            print("  Bounds checking works")
    except Exception as e:
        print(f"  Pixel operations failed: {e}")
        return False
    
    return True

def test_filter_system_with_tracking():
    """Test the complete filter system with performance tracking"""
    print("\n3. Testing filter system with performance tracking...")
    
    try:
        import pixtrick_engine as engine
        
        # Initialize performance tracking if available
        collector = None
        if PERFORMANCE_TRACKING:
            db = PerformanceDatabase()
            collector = BenchmarkCollector(db)
            print("    Performance tracking enabled")
        
        # Create a test image with some pattern
        print("  Creating test pattern image...")
        img = engine.PixImage(200, 200)
        
        # Create a simple gradient pattern
        for y in range(200):
            for x in range(200):
                intensity = int(255 * ((x + y) / (200 + 200)))
                img.set_pixel(x, y, intensity, intensity // 2, 255 - intensity, 255)
        
        print("✓ Created gradient test image")
        
        # Test filters with performance tracking
        filters_to_test = [
            ("brightness", {"amount": 20.0}),
            ("contrast", {"amount": 15.0}),
            ("saturation", {"amount": 150.0}),
            ("box_blur", {"radius": 2.0}),
            ("invert", {}),
            ("grayscale", {}),
        ]
        
        # Create node graph
        graph = engine.NodeGraph()
        graph.set_source_image(img)
        
        performance_results = []
        
        # Test each filter individually
        for filter_type, params in filters_to_test:
            try:
                # Clear previous nodes
                for node_id in graph.get_node_ids():
                    graph.remove_node(node_id)
                
                # Add single filter
                node = engine.FilterNode(f"test_{filter_type}", filter_type)
                for key, value in params.items():
                    node.set_parameter(key, value)
                
                graph.add_node(node)
                
                # Measure performance
                start_time = time.perf_counter()
                result = graph.process()
                end_time = time.perf_counter()
                
                processing_time = (end_time - start_time) * 1000
                print(f"{filter_type}: {processing_time:.2f}ms - {result.info()}")
                
                # Store performance data if tracking enabled
                if collector:
                    image_meta = ImageMetadata(
                        width=img.width,
                        height=img.height,
                        channels=img.channels,
                        file_size_mb=None,
                        format="Generated",
                        complexity_score=0.5  # Medium complexity gradient
                    )
                    
                    filter_config = [FilterConfig(
                        filter_type=filter_type,
                        parameters=params,
                        enabled=True,
                        order_index=0
                    )]
                    
                    performance_metrics = PerformanceMetrics(
                        execution_time_ms=processing_time,
                        memory_peak_mb=0.0,  # Would need separate monitoring
                        memory_allocated_mb=0.0,
                        cpu_usage_percent=0.0,
                        cache_hits=0,
                        cache_misses=0,
                        filter_times_ms={filter_type: processing_time}
                    )
                    
                    benchmark_result = collector.create_benchmark_result(
                        image_meta, filter_config, performance_metrics,
                        test_type="functional_test",
                        tags=["quick_test", filter_type],
                        notes=f"Functional test of {filter_type} filter"
                    )
                    
                    collector.store_benchmark(benchmark_result)
                
                performance_results.append((filter_type, processing_time))
                
            except Exception as e:
                print(f"{filter_type} failed: {e}")
                return False
        
        # Test filter chain
        print("\n  Testing filter chain...")
        
        # Clear all nodes
        for node_id in graph.get_node_ids():
            graph.remove_node(node_id)
        
        # Add a chain of filters
        chain_filters = [
            ("brightness", {"amount": 10.0}),
            ("contrast", {"amount": 8.0}),
            ("saturation", {"amount": 120.0})
        ]
        
        filter_configs = []
        for i, (filter_type, params) in enumerate(chain_filters):
            node = engine.FilterNode(f"chain_{i}_{filter_type}", filter_type)
            for key, value in params.items():
                node.set_parameter(key, value)
            graph.add_node(node)
            
            filter_configs.append(FilterConfig(
                filter_type=filter_type,
                parameters=params,
                enabled=True,
                order_index=i
            ))
        
        # Process chain
        start_time = time.perf_counter()
        chain_result = graph.process()
        end_time = time.perf_counter()
        
        chain_time = (end_time - start_time) * 1000
        print(f"Filter chain: {chain_time:.2f}ms - {chain_result.info()}")
        
        # Store chain performance if tracking enabled
        if collector:
            chain_performance = PerformanceMetrics(
                execution_time_ms=chain_time,
                memory_peak_mb=0.0,
                memory_allocated_mb=0.0,
                cpu_usage_percent=0.0,
                cache_hits=0,
                cache_misses=0,
                filter_times_ms={f[0]: chain_time / len(chain_filters) for f, _ in chain_filters}
            )
            
            chain_benchmark = collector.create_benchmark_result(
                image_meta, filter_configs, chain_performance,
                test_type="functional_test",
                tags=["quick_test", "filter_chain"],
                notes="Functional test of filter chain"
            )
            
            collector.store_benchmark(chain_benchmark)
        
        # Performance summary
        if performance_results:
            print(f"\n  Performance Summary:")
            total_time = sum(time for _, time in performance_results)
            print(f"    Total individual filter time: {total_time:.1f}ms")
            print(f"    Chain processing time: {chain_time:.1f}ms")
            print(f"    Chain efficiency: {(total_time/chain_time)*100:.1f}% (lower is better)")
            
            if PERFORMANCE_TRACKING:
                print(f"    Results stored in performance database")
        
        return True
        
    except Exception as e:
        print(f"✗ Filter system test failed: {e}")
        traceback.print_exc()
        return False

def test_node_graph_features():
    """Test advanced node graph features"""
    print("\n4. Testing advanced node graph features...")
    
    try:
        import pixtrick_engine as engine
        
        # Create test image
        img = engine.PixImage(100, 100)
        
        # Create a complex filter chain
        graph = engine.NodeGraph()
        graph.set_source_image(img)
        
        # Add multiple filters
        filters = [
            ("brightness_1", "brightness", {"amount": 10.0}),
            ("contrast_1", "contrast", {"amount": 5.0}),
            ("saturation_1", "saturation", {"amount": 120.0}),
            ("blur_1", "box_blur", {"radius": 1.0}),
        ]
        
        for node_id, filter_type, params in filters:
            node = engine.FilterNode(node_id, filter_type)
            for key, value in params.items():
                node.set_parameter(key, value)
            graph.add_node(node)
        
        print(f"Added {len(filters)} filters to graph")
        
        # Test execution order
        original_order = graph.get_execution_order()
        print(f"Original order: {original_order}")
        
        # Test reordering
        new_order = original_order[::-1]  # Reverse order
        graph.set_execution_order(new_order)
        reordered = graph.get_execution_order()
        print(f"Reordered to: {reordered}")
        
        # Test processing with different orders
        start_time = time.time()
        result1 = graph.process()
        time1 = time.time() - start_time
        
        # Process again (should use cache)
        start_time = time.time()
        result2 = graph.process()
        time2 = time.time() - start_time
        
        print(f"First processing: {time1*1000:.2f}ms")
        print(f"Cached processing: {time2*1000:.2f}ms")
        if time2 > 0:
            print(f"Cache speedup: {time1/time2:.1f}x")
        else:
            print("Cache working (instant processing)")
        
        # Test node enable/disable
        node = graph.get_node("brightness_1")
        node.toggle_enabled()
        graph.update_node(node)
        print("Toggled node enabled/disabled")
        
        # Test parameter introspection
        available_filters = engine.NodeGraph.get_available_filters()
        print(f"Available filters: {', '.join(available_filters)}")
        
        return True
        
    except Exception as e:
        print(f"Node graph features test failed: {e}")
        traceback.print_exc()
        return False

def test_image_loading():
    """Test loading a real image file"""
    print("\n5. Testing image I/O...")
    
    try:
        import pixtrick_engine as engine
        
        # Try to create a test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a more interesting test image
            print("  Creating colorful test image...")
            width, height = 300, 200
            
            # Create a gradient with color bands
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    # Create horizontal color bands
                    hue_band = (y // 20) % 6
                    intensity = int(255 * (x / width))
                    
                    if hue_band == 0:    # Red
                        img_array[y, x] = [intensity, 0, 0]
                    elif hue_band == 1:  # Green
                        img_array[y, x] = [0, intensity, 0]
                    elif hue_band == 2:  # Blue
                        img_array[y, x] = [0, 0, intensity]
                    elif hue_band == 3:  # Yellow
                        img_array[y, x] = [intensity, intensity, 0]
                    elif hue_band == 4:  # Magenta
                        img_array[y, x] = [intensity, 0, intensity]
                    else:                # Cyan
                        img_array[y, x] = [0, intensity, intensity]
            
            test_img = Image.fromarray(img_array)
            test_image_path = "colorful_test.png"
            test_img.save(test_image_path)
            print(f"Created colorful test image: {test_image_path}")
            
        except ImportError:
            print("  Creating simple test image without PIL...")
            # Create image manually
            img = engine.PixImage(200, 150)
            for y in range(150):
                for x in range(200):
                    # Simple pattern
                    r = (x * 255) // 200
                    g = (y * 255) // 150
                    b = ((x + y) * 255) // 350
                    img.set_pixel(x, y, r, g, b, 255)
            
            test_image_path = "simple_test.png"
            img.save(test_image_path)
            print(f"Created simple test image: {test_image_path}")
            
            # Load it back
            loaded_img = engine.load_image(test_image_path)
            print(f"Loaded image: {loaded_img.info()}")
            return True
        
        # Load with our engine and test filters
        loaded_img = engine.load_image(test_image_path)
        print(f"Loaded image: {loaded_img.info()}")
        
        # Apply a filter chain
        graph = engine.NodeGraph()
        graph.set_source_image(loaded_img)
        
        # Add interesting filter combination
        filters = [
            ("enhance", "brightness", {"amount": 15.0}),
            ("pop", "contrast", {"amount": 20.0}),
            ("vibrant", "saturation", {"amount": 130.0}),
        ]
        
        for node_id, filter_type, params in filters:
            node = engine.FilterNode(node_id, filter_type)
            for key, value in params.items():
                node.set_parameter(key, value)
            graph.add_node(node)
        
        # Process and save result
        result = graph.process()
        output_path = "enhanced_output.png"
        result.save(output_path)
        print(f"Applied enhancement chain and saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Image I/O test failed: {e}")
        traceback.print_exc()
        return False

def quick_performance_benchmark():
    """Quick performance benchmark for development"""
    print("\n6. Quick performance benchmark...")
    
    try:
        import pixtrick_engine as engine
        
        # Test medium-sized image
        print("  Creating 500x500 test image...")
        img = engine.PixImage(500, 500)
        
        # Fill with test pattern (checkerboard)
        for y in range(500):
            for x in range(500):
                if (x // 20 + y // 20) % 2:
                    img.set_pixel(x, y, 255, 255, 255, 255)
                else:
                    img.set_pixel(x, y, 128, 64, 192, 255)
        
        # Test key filters
        test_filters = [
            ("brightness", {"amount": 20.0}),
            ("contrast", {"amount": 15.0}),
            ("saturation", {"amount": 120.0}),
            ("box_blur", {"radius": 2.0}),
        ]
        
        graph = engine.NodeGraph()
        graph.set_source_image(img)
        
        total_time = 0
        for filter_type, params in test_filters:
            # Clear previous nodes
            for node_id in graph.get_node_ids():
                graph.remove_node(node_id)
            
            node = engine.FilterNode("test", filter_type)
            for key, value in params.items():
                node.set_parameter(key, value)
            graph.add_node(node)
            
            # Benchmark 3 runs for stability
            times = []
            for _ in range(3):
                graph.clear_cache()
                start = time.perf_counter()
                result = graph.process()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = sum(times) / len(times)
            total_time += avg_time
            
            pixels_processed = img.width * img.height
            mpixels_per_sec = (pixels_processed / 1_000_000) / (avg_time / 1000)
            
            print(f"    {filter_type:12}: {avg_time:6.1f}ms ({mpixels_per_sec:4.1f} MP/s)")
        
        print(f"    {'Total':12}: {total_time:6.1f}ms")
        
        # Performance targets (rough estimates)
        target_time = 200  # 200ms total for 4 filters on 500x500
        if total_time < target_time:
            print(f"Performance target met! ({total_time:.1f}ms < {target_time}ms)")
        else:
            print(f"Performance target missed ({total_time:.1f}ms > {target_time}ms)")
        
        return True
        
    except Exception as e:
        print(f"Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files"""
    test_files = [
        "colorful_test.png", 
        "simple_test.png", 
        "enhanced_output.png",
    ]
    for file in test_files:
        try:
            Path(file).unlink(missing_ok=True)
        except:
            pass

def main():
    """Main test runner with optional performance tracking"""
    print("PixTrick Engine Test Suite")
    print("=" * 40)
    
    if PERFORMANCE_TRACKING:
        print("Performance tracking: ENABLED")
    else:
        print("Performance tracking: disabled (install performance_data_schema.py)")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Filter System + Tracking", test_filter_system_with_tracking),
        ("Node Graph Features", test_node_graph_features),
        ("Image I/O", test_image_loading),
        ("Quick Performance", quick_performance_benchmark),
    ]
    
    tests_passed = 0
    total_tests = len(tests)
    
    # Run tests
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                tests_passed += 1
                print(f"{test_name} PASSED")
            else:
                print(f"{test_name} FAILED")
        except Exception as e:
            print(f"{test_name} CRASHED: {e}")
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*40}")
    print(f"Test Results: {tests_passed}/{total_tests} passed")
        
    # Cleanup
    cleanup()
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)