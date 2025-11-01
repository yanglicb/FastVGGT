#!/usr/bin/env python3
"""
Test script to verify coordinate axes are properly rendered with arrows attached.
"""

import numpy as np
import sys
sys.path.append(".")

from visual_util import create_coordinate_axes

def test_axes_creation():
    """Test that coordinate axes are created with proper geometry."""
    
    print("=" * 70)
    print("Testing Coordinate Axes Visualization")
    print("=" * 70)
    
    # Test 1: Default axes
    print("\nTest 1: Creating default coordinate axes")
    axes_scene = create_coordinate_axes(scene_scale=10.0)
    
    print(f"Number of geometries: {len(axes_scene.geometry)}")
    print("Geometry names:")
    for name in axes_scene.geometry.keys():
        geom = axes_scene.geometry[name]
        print(f"  - {name}: {type(geom).__name__}, {len(geom.vertices)} vertices")
    
    # Verify all expected components are present
    expected_components = ['x_axis', 'x_arrow', 'y_axis', 'y_arrow', 'z_axis', 'z_arrow']
    for component in expected_components:
        if component in axes_scene.geometry:
            print(f"  ✓ {component} exists")
        else:
            print(f"  ✗ {component} MISSING!")
    
    # Test 2: Axes with custom prefix and colors
    print("\n" + "-" * 70)
    print("Test 2: Creating axes with custom prefix and colors")
    custom_axes = create_coordinate_axes(
        scene_scale=5.0,
        prefix="test_",
        colors=[(200, 100, 100, 200), (100, 200, 100, 200), (100, 100, 200, 200)]
    )
    
    print(f"Number of geometries: {len(custom_axes.geometry)}")
    print("Geometry names:")
    for name in custom_axes.geometry.keys():
        print(f"  - {name}")
    
    # Test 3: Check positioning
    print("\n" + "-" * 70)
    print("Test 3: Checking arrow positioning")
    
    axis_length = 3.0
    test_axes = create_coordinate_axes(scene_scale=10.0, axis_length=axis_length)
    
    # Check X-axis arrow position
    x_arrow = test_axes.geometry['x_arrow']
    x_arrow_center = x_arrow.centroid
    print(f"\nX-axis arrow centroid: [{x_arrow_center[0]:.3f}, {x_arrow_center[1]:.3f}, {x_arrow_center[2]:.3f}]")
    print(f"  Expected X-position near: {axis_length - axis_length * 0.15 / 2:.3f}")
    
    # Check Y-axis arrow position
    y_arrow = test_axes.geometry['y_arrow']
    y_arrow_center = y_arrow.centroid
    print(f"\nY-axis arrow centroid: [{y_arrow_center[0]:.3f}, {y_arrow_center[1]:.3f}, {y_arrow_center[2]:.3f}]")
    print(f"  Expected Y-position near: {axis_length - axis_length * 0.15 / 2:.3f}")
    
    # Check Z-axis arrow position
    z_arrow = test_axes.geometry['z_arrow']
    z_arrow_center = z_arrow.centroid
    print(f"\nZ-axis arrow centroid: [{z_arrow_center[0]:.3f}, {z_arrow_center[1]:.3f}, {z_arrow_center[2]:.3f}]")
    print(f"  Expected Z-position near: {axis_length - axis_length * 0.15 / 2:.3f}")
    
    # Check cylinder endpoints
    x_cylinder = test_axes.geometry['x_axis']
    x_cyl_bounds = x_cylinder.bounds
    print(f"\nX-cylinder bounds: X=[{x_cyl_bounds[0][0]:.3f}, {x_cyl_bounds[1][0]:.3f}]")
    print(f"  Expected to end near: {axis_length - axis_length * 0.15:.3f}")
    
    print("\n" + "=" * 70)
    print("Axes visualization test complete!")
    print("Export the axes to a GLB file to visualize:")
    print("  axes_scene.export('test_axes.glb')")
    print("=" * 70)

if __name__ == "__main__":
    test_axes_creation()
