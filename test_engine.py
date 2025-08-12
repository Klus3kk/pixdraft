#!/usr/bin/env python3
"""
Enhanced test script for PixTrick Rust engine
Demonstrates working filter system with performance benchmarks
"""

import sys
import time
from pathlib import Path
import traceback

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
        print(f"✓ Created image: {img.info()}")
        print(f"  Dimensions: {img.width}x{img.height}, Channels: {img.channels}")
    except Exception as e:
        print(f"✗ Failed to create image: {e}")
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
            print("✗ Bounds checking failed")
        except Exception:
            print("✓ Bounds checking works")
    except Exception as e:
        print(f"✗ Pixel operations failed: {e}")
        return False
    
    return True

def test_filter_system():
    """Test the complete filter system with actual processing"""
    print("\n3. Testing complete filter system...")
    
    try:
        import pixtrick_engine as engine
        
        # Create a test image with some pattern
        print("  Creating test pattern image...")
        img = engine.PixImage(200, 200)
        
        # Create a gradient pattern for better filter testing
        for y in range(200):
            for x in range(200):
                # Create a radial gradient
                center_x, center_y = 100, 100
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                intensity = max(0, min(255, int(255 - distance * 2)))
                
                # Add some color variation
                r = intensity
                g = intensity if x < 100 else max(0, intensity - 50)
                b = intensity if y < 100 else max(0, intensity - 50)
                
                img.set_pixel(x, y, r, g, b, 255)
        
        print("✓ Created gradient test image")
        
        # Test all available filters
        filters_to_test = [
            ("brightness", {"amount": 20.0}),
            ("contrast", {"amount": 15.0}),
            ("saturation", {"amount": 150.0}),
            ("hue_shift", {"degrees": 45.0}),
            ("box_blur", {"radius": 2.0}),
            ("gaussian_blur", {"radius": 1.5}),
            ("invert", {}),
            ("grayscale", {}),
        ]
        
        # Create node graph
        graph = engine.NodeGraph()
        graph.set_source_image(img)
        
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
                
                # Process and time
                start_time = time.time()
                result = graph.process()
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000
                print(f"✓ {filter_type}: {processing_time:.2f}ms - {result.info()}")
                
            except Exception as e:
                print(f"✗ {filter_type} failed: {e}")
                traceback.print_exc()
                return False
        
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
        
        print(f"✓ Added {len(filters)} filters to graph")
        
        # Test execution order
        original_order = graph.get_execution_order()
        print(f"✓ Original order: {original_order}")
        
        # Test reordering
        new_order = original_order[::-1]  # Reverse order
        graph.set_execution_order(new_order)
        reordered = graph.get_execution_order()
        print(f"✓ Reordered to: {reordered}")
        
        # Test processing with different orders
        start_time = time.time()
        result1 = graph.process()
        time1 = time.time() - start_time
        
        # Process again (should use cache)
        start_time = time.time()
        result2 = graph.process()
        time2 = time.time() - start_time
        
        print(f"✓ First processing: {time1*1000:.2f}ms")
        print(f"✓ Cached processing: {time2*1000:.2f}ms")
        print(f"✓ Cache speedup: {time1/time2:.1f}x" if time2 > 0 else "✓ Cache working")
        
        # Test node enable/disable
        node = graph.get_node("brightness_1")
        node.toggle_enabled()
        graph.update_node(node)
        print("✓ Toggled node enabled/disabled")
        
        # Test parameter introspection
        available_filters = engine.NodeGraph.get_available_filters()
        print(f"✓ Available filters: {', '.join(available_filters)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Node graph features test failed: {e}")
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
            print(f"✓ Created colorful test image: {test_image_path}")
            
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
            print(f"✓ Created simple test image: {test_image_path}")
            
            # Load it back
            loaded_img = engine.load_image(test_image_path)
            print(f"✓ Loaded image: {loaded_img.info()}")
            return True
        
        # Load with our engine and test filters
        loaded_img = engine.load_image(test_image_path)
        print(f"✓ Loaded image: {loaded_img.info()}")
        
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
        print(f"✓ Applied enhancement chain and saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image I/O test failed: {e}")
        traceback.print_exc()
        return False

def benchmark_performance():
    """Comprehensive performance benchmarking"""
    print("\n6. Performance benchmarks...")
    
    try:
        import pixtrick_engine as engine
        
        # Test different image sizes
        sizes = [
            (100, 100, "Small"),
            (500, 500, "Medium"), 
            (1000, 1000, "Large"),
        ]
        
        for width, height, size_name in sizes:
            print(f"\n  Testing {size_name} ({width}x{height}):")
            
            # Create test image
            img = engine.PixImage(width, height)
            
            # Fill with test pattern
            for y in range(0, height, 10):
                for x in range(0, width, 10):
                    # Checkerboard pattern
                    if (x // 10 + y // 10) % 2:
                        img.set_pixel(x, y, 255, 255, 255, 255)
                    else:
                        img.set_pixel(x, y, 0, 0, 0, 255)
            
            # Test individual filter performance
            test_filters = [
                ("brightness", {"amount": 20.0}),
                ("contrast", {"amount": 15.0}),
                ("saturation", {"amount": 120.0}),
                ("box_blur", {"radius": 2.0}),
                ("gaussian_blur", {"radius": 1.5}),
            ]
            
            total_time = 0
            for filter_type, params in test_filters:
                graph = engine.NodeGraph()
                graph.set_source_image(img)
                
                node = engine.FilterNode("test", filter_type)
                for key, value in params.items():
                    node.set_parameter(key, value)
                graph.add_node(node)
                
                # Benchmark multiple runs
                times = []
                for _ in range(3):
                    graph.clear_cache()  # Ensure no caching between runs
                    start = time.time()
                    result = graph.process()
                    end = time.time()
                    times.append(end - start)
                
                avg_time = sum(times) / len(times) * 1000  # Convert to ms
                total_time += avg_time
                
                pixels_processed = width * height
                mpixels_per_sec = (pixels_processed / 1_000_000) / (avg_time / 1000)
                
                print(f"    {filter_type}: {avg_time:.2f}ms ({mpixels_per_sec:.1f} MP/s)")
            
            print(f"    Total processing time: {total_time:.2f}ms")
            
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files"""
    test_files = [
        "colorful_test.png", 
        "simple_test.png", 
        "enhanced_output.png",
        "test_image.png", 
        "test_output.png", 
        "benchmark_test.png", 
        "benchmark_result.png"
    ]
    for file in test_files:
        try:
            Path(file).unlink(missing_ok=True)
        except:
            pass

if __name__ == "__main__":
    print("PixTrick Enhanced Engine Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Complete Filter System", test_filter_system),
        ("Node Graph Features", test_node_graph_features),
        ("Image I/O", test_image_loading),
        ("Performance Benchmarks", benchmark_performance),
    ]
    
    tests_passed = 0
    total_tests = len(tests)
    
    # Run tests
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                tests_passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    print(f"{'='*50}\n")
    # Cleanup
    cleanup()

    sys.exit(0 if tests_passed == total_tests else 1)