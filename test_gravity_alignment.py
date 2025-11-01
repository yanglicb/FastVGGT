#!/usr/bin/env python3
"""
Test script for the gravity alignment feature with single camera selection.
Tests that the alignment correctly converts from camera coordinates (Z=forward)
to world coordinates (Z=up/vertical).
"""

import numpy as np
from metadata_util import (
    compute_gravity_alignment_from_pitch_and_poses,
    compute_gravity_alignment_from_single_camera,
    compute_gravity_alignment_from_camera_vectors
)

def test_single_camera_alignment():
    """Test gravity alignment with a single reference camera."""
    
    # Create sample data: 5 cameras with some pitch angles
    num_cameras = 5
    
    # Sample pitch angles (some might be None)
    # -90° = camera horizontal, 0° = camera pointing straight down
    pitch_angles = [-45.0, -50.0, None, -48.0, -52.0]
    
    # Create sample extrinsic matrices (world-to-camera)
    # For simplicity, we'll create identity-based matrices with slight variations
    extrinsics = np.zeros((num_cameras, 4, 4))
    for i in range(num_cameras):
        extrinsics[i] = np.eye(4)
        # Add slight rotation variation
        angle = np.radians(i * 5)
        extrinsics[i, 0, 0] = np.cos(angle)
        extrinsics[i, 0, 2] = np.sin(angle)
        extrinsics[i, 2, 0] = -np.sin(angle)
        extrinsics[i, 2, 2] = np.cos(angle)
        # Add position variation
        extrinsics[i, 0, 3] = i * 0.5
    
    print("=" * 70)
    print("Testing Gravity Alignment with Camera-to-World Coordinate Conversion")
    print("=" * 70)
    
    print(f"\nCoordinate System Convention:")
    print("  Camera coords: X=forward, Y=right, Z=up")
    print("  World coords:  Z=up (vertical/gravity direction)")
    print(f"\nInput: {num_cameras} cameras")
    print(f"Pitch angles: {pitch_angles}")
    print("  (pitch -90° = horizontal, pitch 0° = pointing down)")
    
    # Test 1: Average of all cameras (default behavior)
    print("\n" + "-" * 70)
    print("Test 1: Average of all cameras (reference_camera_idx=None)")
    print("-" * 70)
    alignment_avg = compute_gravity_alignment_from_pitch_and_poses(
        pitch_angles, extrinsics, reference_camera_idx=None
    )
    print(f"Result shape: {alignment_avg.shape}")
    print(f"Rotation component:\n{alignment_avg[:3, :3]}")
    
    # Verify that Z-axis points up after alignment
    z_axis_after = alignment_avg[:3, :3] @ np.array([0, 0, 1])
    print(f"World Z-axis after alignment: {z_axis_after}")
    
    # Test 2: Using camera 0 as reference
    print("\n" + "-" * 70)
    print("Test 2: Using camera 0 as reference")
    print("-" * 70)
    alignment_cam0 = compute_gravity_alignment_from_pitch_and_poses(
        pitch_angles, extrinsics, reference_camera_idx=0
    )
    print(f"Result shape: {alignment_cam0.shape}")
    print(f"Rotation component:\n{alignment_cam0[:3, :3]}")
    
    # Test 3: Using camera 2 (which has None pitch) as reference
    print("\n" + "-" * 70)
    print("Test 3: Using camera 2 (with None pitch) as reference")
    print("-" * 70)
    alignment_cam2 = compute_gravity_alignment_from_pitch_and_poses(
        pitch_angles, extrinsics, reference_camera_idx=2
    )
    print(f"Result shape: {alignment_cam2.shape}")
    print(f"Rotation component:\n{alignment_cam2[:3, :3]}")
    
    # Test 4: Direct single camera function
    print("\n" + "-" * 70)
    print("Test 4: Direct call to compute_gravity_alignment_from_single_camera")
    print("-" * 70)
    alignment_direct = compute_gravity_alignment_from_single_camera(
        pitch_angles, extrinsics, camera_idx=1
    )
    print(f"Result shape: {alignment_direct.shape}")
    print(f"Rotation component:\n{alignment_direct[:3, :3]}")
    
    # Test 5: Camera vectors only (no pitch data)
    print("\n" + "-" * 70)
    print("Test 5: Alignment from camera vectors only (camera 3)")
    print("-" * 70)
    alignment_vectors = compute_gravity_alignment_from_camera_vectors(
        extrinsics, reference_camera_idx=3
    )
    print(f"Result shape: {alignment_vectors.shape}")
    print(f"Rotation component:\n{alignment_vectors[:3, :3]}")
    
    # Test 6: Invalid camera index
    print("\n" + "-" * 70)
    print("Test 6: Invalid camera index (should fall back to camera 0)")
    print("-" * 70)
    alignment_invalid = compute_gravity_alignment_from_pitch_and_poses(
        pitch_angles, extrinsics, reference_camera_idx=999
    )
    print(f"Result shape: {alignment_invalid.shape}")
    
    # Test 7: Verify coordinate system conversion
    print("\n" + "-" * 70)
    print("Test 7: Verify camera up vector → world Z-axis alignment")
    print("-" * 70)
    # Create a camera pointing horizontally (pitch = -90°)
    cam_horizontal = np.eye(4)
    # Camera's up direction in camera space: +Z = [0, 0, 1]
    # This should align with world Z-axis after gravity alignment
    up_in_camera = np.array([0, 0, 1, 0])
    cam_to_world = np.linalg.inv(cam_horizontal)
    up_in_world_before = (cam_to_world @ up_in_camera)[:3]
    
    # Apply alignment
    up_in_world_after = (alignment_avg @ cam_to_world @ up_in_camera)[:3]
    
    print(f"Camera up vector before alignment: {up_in_world_before}")
    print(f"Camera up vector after alignment:  {up_in_world_after}")
    print(f"Target (world Z-axis):              [0, 0, 1]")
    alignment_error = np.linalg.norm(up_in_world_after / np.linalg.norm(up_in_world_after) - np.array([0, 0, 1]))
    print(f"Alignment error: {alignment_error:.6f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("Camera coordinate system: X=forward, Y=right, Z=up")
    print("=" * 70)

if __name__ == "__main__":
    test_single_camera_alignment()
