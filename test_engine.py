#!/usr/bin/env python3
"""
Test script for PixTrick Rust engine
"""

import sys
import time
from pathlib import Path

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

def test_node_system():
    """Test the node-based processing system"""
    print("\n3. Testing node system...")
    
    try:
        import pixtrick_engine as engine
        
        # Create a node graph
        graph = engine.NodeGraph()
        print("✓ Created node graph")
        
        # Create some filter nodes
        brightness_node = engine.FilterNode("brightness_1", "brightness")
        brightness_node.set_parameter("amount", 20.0)
        
        contrast_node = engine.FilterNode("contrast_1", "contrast")
        contrast_node.set_parameter("amount", 10.0)
        
        # Add nodes to graph
        graph.add_node(brightness_node)
        graph.add_node(contrast_node)
        print("✓ Added filter nodes to graph")
        
        # Test node retrieval
        retrieved = graph.get_node("brightness_1")
        amount = retrieved.get_parameter("amount")
        print(f"✓ Retrieved brightness node with amount: {amount}")
        
        # Test execution order
        order = graph.get_execution_order()
        print(f"✓ Execution order: {order}")
        
        return True
        
    except Exception as e:
        print(f"✗ Node system test failed: {e}")
        return False

def test_image_loading():
    """Test loading a real image file"""
    print("\n4. Testing image loading...")
    
    # Create a test image if it doesn't exist
    test_image_path = "test_image.png"
    
    try:
        import pixtrick_engine as engine
        from PIL import Image
        import numpy as np
        
        # Create a test image using PIL
        print("  Creating test image...")
        test_img = Image.new('RGB', (200, 150), color='red')
        # Add some variation
        pixels = np.array(test_img)
        pixels[50:100, 50:150] = [0, 255, 0]  # Green rectangle
        pixels[100:150, 100:200] = [0, 0, 255]  # Blue rectangle
        test_img = Image.fromarray(pixels)
        test_img.save(test_image_path)
        print(f"  ✓ Created test image: {test_image_path}")
        
        # Load with our engine
        loaded_img = engine.load_image(test_image_path)
        print(f"✓ Loaded image: {loaded_img.info()}")
        
        # Test saving
        output_path = "test_output.png"
        loaded_img.save(output_path)
        print(f"✓ Saved image to: {output_path}")
        
        return True
        
    except ImportError:
        print("  Skipping image loading test (PIL not available)")
        print("  Install with: pip install Pillow")
        return True
    except Exception as e:
        print(f"✗ Image loading test failed: {e}")
        return False

def benchmark_filters():
    """Benchmark filter performance"""
    print("\n5. Benchmarking filters...")
    
    try:
        import pixtrick_engine as engine
        from PIL import Image
        import numpy as np
        
        # Create a larger test image for performance testing
        print("  Creating 1000x1000 test image...")
        test_img = Image.new('RGB', (1000, 1000))
        pixels = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        test_img = Image.fromarray(pixels)
        test_img.save("benchmark_test.png")
        
        # Load and process
        img = engine.load_image("benchmark_test.png")
        print(f"✓ Loaded {img.width}x{img.height} image")
        
        # Test node-based processing
        graph = engine.NodeGraph()
        graph.set_source_image(img)
        
        # Add multiple filters
        filters = [
            ("brightness_1", "brightness", {"amount": 10.0}),
            ("contrast_1", "contrast", {"amount": 5.0}),
            ("brightness_2", "brightness", {"amount": -5.0}),
        ]
        
        for node_id, filter_type, params in filters:
            node = engine.FilterNode(node_id, filter_type)
            for key, value in params.items():
                node.set_parameter(key, value)
            graph.add_node(node)
        
        # Benchmark processing
        start_time = time.time()
        result = graph.process()
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        pixels_processed = img.width * img.height
        mpixels_per_sec = (pixels_processed / 1_000_000) / (processing_time / 1000)
        
        print(f"✓ Processed {len(filters)} filters in {processing_time:.2f}ms")
        print(f"✓ Performance: {mpixels_per_sec:.2f} megapixels/second")
        
        # Save result
        result.save("benchmark_result.png")
        print("✓ Saved benchmark result")
        
        return True
        
    except ImportError:
        print("  Skipping benchmark (PIL not available)")
        return True
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    test_files = ["test_image.png", "test_output.png", "benchmark_test.png", "benchmark_result.png"]
    for file in test_files:
        try:
            Path(file).unlink(missing_ok=True)
        except:
            pass

if __name__ == "__main__":
    print("PixTrick Engine Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 5
    
    # Run tests
    if test_basic_functionality():
        tests_passed += 1
    
    if test_node_system():
        tests_passed += 1
    
    if test_image_loading():
        tests_passed += 1
    
    if benchmark_filters():
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*40}")
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        
    # Cleanup
    cleanup()
    print("\n✓ Cleaned up test files")
    
    sys.exit(0 if tests_passed == total_tests else 1)