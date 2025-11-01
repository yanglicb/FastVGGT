# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for extracting and using metadata from images.
"""

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from typing import Optional, List, Dict
from scipy.spatial.transform import Rotation


def extract_pitch_from_exif(image_path: str) -> Optional[float]:
    """
    Extract pitch angle from image EXIF metadata.
    Expected format in User Comment: "Pitch:-71.351295,Roll:-10.72634,Azimuth:-154.74142"
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Pitch angle in degrees, or None if not found
    """
    try:
        import re
        
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if exif_data is None:
            return None
            
        # Look for pitch in User Comment or other text fields
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            
            # Check UserComment, ImageDescription, and other text fields
            if tag in ['UserComment', 'ImageDescription', 'Comment', 'XPComment']:
                # Convert bytes to string if needed
                if isinstance(value, bytes):
                    try:
                        value_str = value.decode('utf-8', errors='ignore')
                    except:
                        value_str = str(value)
                else:
                    value_str = str(value)
                
                # Parse format: "Pitch:-71.351295,Roll:-10.72634,Azimuth:-154.74142"
                match = re.search(r'Pitch:\s*(-?\d+\.?\d*)', value_str, re.IGNORECASE)
                if match:
                    pitch = float(match.group(1))
                    return pitch
            
            # Also check if tag itself contains 'pitch'
            if 'pitch' in str(tag).lower():
                try:
                    return float(value)
                except:
                    pass
                
        return None
    except Exception as e:
        print(f"Warning: Could not extract EXIF from {image_path}: {e}")
        return None


def extract_pitch_from_filename(image_path: str) -> Optional[float]:
    """
    Extract pitch angle from filename if encoded there.
    Example: "image_pitch_-15.5.jpg" -> -15.5
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Pitch angle in degrees, or None if not found
    """
    try:
        filename = Path(image_path).stem
        if 'pitch' in filename.lower():
            # Try to extract number after 'pitch'
            parts = filename.lower().split('pitch')
            if len(parts) > 1:
                # Extract numeric value
                pitch_str = parts[1].strip('_-')
                # Find the numeric portion
                import re
                match = re.search(r'[-+]?\d*\.?\d+', pitch_str)
                if match:
                    return float(match.group())
        return None
    except Exception as e:
        print(f"Warning: Could not extract pitch from filename {image_path}: {e}")
        return None


def load_pitch_from_json(metadata_path: str, image_name: str) -> Optional[float]:
    """
    Load pitch angle from a JSON metadata file.
    
    Args:
        metadata_path: Path to JSON file containing metadata
        image_name: Name of the image to look up
        
    Returns:
        Pitch angle in degrees, or None if not found
    """
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Support different JSON structures
        if image_name in metadata:
            if isinstance(metadata[image_name], dict):
                return metadata[image_name].get('pitch')
            return metadata[image_name]
        
        return None
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        return None


def get_pitch_angles(image_paths: List[str], metadata_path: Optional[str] = None) -> List[Optional[float]]:
    """
    Get pitch angles for a list of images from various sources.
    
    Args:
        image_paths: List of image file paths
        metadata_path: Optional path to JSON metadata file
        
    Returns:
        List of pitch angles (in degrees) corresponding to each image.
        None values indicate no pitch information was found.
    """
    pitch_angles = []
    
    for image_path in image_paths:
        pitch = None
        
        # Try JSON metadata first if provided
        if metadata_path:
            image_name = Path(image_path).name
            pitch = load_pitch_from_json(metadata_path, image_name)
        
        # Try EXIF data
        if pitch is None:
            pitch = extract_pitch_from_exif(image_path)
        
        # Try filename
        if pitch is None:
            pitch = extract_pitch_from_filename(image_path)
        
        pitch_angles.append(pitch)
    
    return pitch_angles


def compute_gravity_alignment_from_pitch_and_poses(
    pitch_angles: List[Optional[float]],
    extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Compute gravity alignment by combining EXIF pitch data (absolute reference)
    with inferred camera poses (relative geometry).
    
    Args:
        pitch_angles: List of pitch angles in degrees from EXIF/metadata
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4) from model
        
    Returns:
        4x4 transformation matrix to align z-axis with world up vector
    """
    # Filter out None values
    valid_pitches = [p for p in pitch_angles if p is not None]

    if not valid_pitches:
        print("Warning: No valid pitch angles found. Computing alignment from camera poses only.")
        return compute_gravity_alignment_from_camera_vectors(extrinsics)

    # Compute average pitch from EXIF data (absolute reference)
    avg_pitch = np.mean(valid_pitches)
    print(f"Average EXIF pitch angle: {avg_pitch:.2f}° ({len(valid_pitches)}/{len(pitch_angles)} images)")

    # Prepare extrinsics as 4x4 matrices
    if extrinsics.shape[-2:] == (3, 4):
        num_cams = len(extrinsics)
        ext_4x4 = np.zeros((num_cams, 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1
    else:
        ext_4x4 = extrinsics

    predicted_pitches = []
    pred_forward_vectors = []
    target_forward_vectors = []
    aligned_pitch_values = []

    for idx, world_to_cam in enumerate(ext_4x4):
        try:
            cam_to_world = np.linalg.inv(world_to_cam)
        except np.linalg.LinAlgError:
            continue

        R_cw = cam_to_world[:3, :3]
        forward_world = R_cw[:, 2]
        forward_norm = np.linalg.norm(forward_world)
        if forward_norm < 1e-6:
            continue
        forward_world = forward_world / forward_norm
        pred_forward_vectors.append(forward_world)

        # Pitch convention: -90° = camera horizontal, 0° = camera vertical down
        val = np.clip(-forward_world[2], -1.0, 1.0)
        pred_pitch = np.degrees(np.arcsin(val)) - 90.0
        predicted_pitches.append(pred_pitch)

        # Build expected forward vector that matches EXIF pitch but keeps horizontal direction
        if idx < len(pitch_angles):
            pitch_value = pitch_angles[idx]
        else:
            pitch_value = None
        if pitch_value is None:
            pitch_value = avg_pitch

        # Unwrap pitch value: sensor reports 0→-90 when tilting down, then -90→0 when tilting up.
        # Adjust by ±180° to match predicted pitch direction.
        candidates = [pitch_value, pitch_value - 180.0, pitch_value + 180.0]
        adjusted_pitch = min(candidates, key=lambda ang: abs(ang - pred_pitch))
        aligned_pitch_values.append(adjusted_pitch)

        theta = np.radians(adjusted_pitch + 90.0)
        expected_vertical = -np.sin(theta)
        expected_horizontal_mag = np.cos(theta)

        horizontal = forward_world.copy()
        horizontal[2] = 0.0
        horiz_norm = np.linalg.norm(horizontal)

        if horiz_norm < 1e-6:
            expected_forward = np.array([0.0, 0.0, expected_vertical])
        else:
            horizontal_dir = horizontal / horiz_norm
            expected_forward = horizontal_dir * expected_horizontal_mag + np.array([0.0, 0.0, expected_vertical])

        exp_norm = np.linalg.norm(expected_forward)
        if exp_norm < 1e-6:
            expected_forward = forward_world
        else:
            expected_forward = expected_forward / exp_norm

        target_forward_vectors.append(expected_forward)

    if not predicted_pitches:
        print("Warning: Could not derive predicted pitches from camera poses. Falling back to EXIF-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    avg_pred_pitch = np.mean(predicted_pitches)
    print(f"Average predicted pitch (from camera poses): {avg_pred_pitch:.2f}°")

    if aligned_pitch_values:
        avg_aligned_pitch = np.mean(aligned_pitch_values)
        print(f"Average EXIF pitch after unwrapping: {avg_aligned_pitch:.2f}°")

    if not target_forward_vectors:
        print("Warning: Could not build expected forward vectors, using pitch-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    avg_pred_forward = np.mean(pred_forward_vectors, axis=0)
    avg_target_forward = np.mean(target_forward_vectors, axis=0)

    pred_norm = np.linalg.norm(avg_pred_forward)
    target_norm = np.linalg.norm(avg_target_forward)

    if pred_norm < 1e-6 or target_norm < 1e-6:
        print("Warning: Forward vectors degenerate, using pitch-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    avg_pred_forward = avg_pred_forward / pred_norm
    avg_target_forward = avg_target_forward / target_norm

    v = np.cross(avg_pred_forward, avg_target_forward)
    s = np.linalg.norm(v)
    c = np.dot(avg_pred_forward, avg_target_forward)

    gravity_alignment = np.eye(4)

    if s < 1e-6:
        if c < 0:
            # 180-degree rotation: choose an arbitrary axis orthogonal to avg_pred_forward
            axis = np.cross(avg_pred_forward, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(avg_pred_forward, np.array([0.0, 1.0, 0.0]))
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                return gravity_alignment  # give up, identity
            axis = axis / axis_norm
            gravity_alignment[:3, :3] = Rotation.from_rotvec(np.pi * axis).as_matrix()
            print("Applied 180° flip to align forward vectors.")
            return gravity_alignment
        else:
            print("Predicted forward already aligned with expected orientation.")
            return gravity_alignment

    vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

    rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    gravity_alignment[:3, :3] = rotation_matrix

    rotation_angle = np.degrees(np.arctan2(s, c))
    print(f"Forward-vector alignment rotation: {rotation_angle:.2f}°")

    return gravity_alignment


def compute_gravity_alignment_from_pitch(pitch_angles: List[Optional[float]]) -> np.ndarray:
    """
    Compute a rotation matrix to align the coordinate system with gravity
    based on average pitch angle from camera metadata.
    
    DEPRECATED: Use compute_gravity_alignment_from_pitch_and_poses() instead
    for better accuracy when camera extrinsics are available.
    
    In typical camera coordinate systems:
    - Pitch angle describes rotation around the X-axis (side-to-side tilt)
    - pitch = -90°: camera is horizontal (normal phone orientation)
    - pitch = 0°: camera is vertical (pointing straight down or up)
    - Negative pitch closer to 0 means camera pointing more downward
    
    Args:
        pitch_angles: List of pitch angles in degrees (can contain None values)
        
    Returns:
        4x4 transformation matrix to align z-axis with world up vector
    """
    # Filter out None values
    valid_pitches = [p for p in pitch_angles if p is not None]
    
    if not valid_pitches:
        print("Warning: No valid pitch angles found. Using identity transform.")
        return np.eye(4)
    
    # Compute average pitch
    avg_pitch = np.mean(valid_pitches)
    print(f"Average pitch angle: {avg_pitch:.2f} degrees ({len(valid_pitches)}/{len(pitch_angles)} images)")
    
    # Calculate rotation angle
    # pitch = -90 (horizontal) → need 90° rotation
    # pitch = 0 (vertical) → need 0° rotation
    rotation_angle = -(avg_pitch + 90)
    
    # Create rotation to compensate for pitch
    gravity_alignment = np.eye(4)
    gravity_alignment[:3, :3] = Rotation.from_euler('x', rotation_angle, degrees=True).as_matrix()
    
    return gravity_alignment


def compute_gravity_alignment_from_camera_vectors(extrinsics: np.ndarray) -> np.ndarray:
    """
    Compute gravity alignment by analyzing the up vectors of cameras.
    Uses the actual inferred camera poses from the reconstruction model.
    
    Args:
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4)
        
    Returns:
        4x4 transformation matrix to align z-axis with world up vector
    """
    num_cameras = len(extrinsics)
    
    # Ensure we have 4x4 matrices
    if extrinsics.shape[-2:] == (3, 4):
        ext_4x4 = np.zeros((num_cameras, 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1
    else:
        ext_4x4 = extrinsics
    
    # Extract camera-to-world transforms
    cam_to_world = np.zeros_like(ext_4x4)
    for i in range(num_cameras):
        cam_to_world[i] = np.linalg.inv(ext_4x4[i])
    
    # Extract up vectors from camera coordinate systems
    # In camera space: X=right, Y=down, Z=forward (OpenCV convention)
    # So the camera's "up" direction is -Y in camera space
    up_vectors = []
    for i in range(num_cameras):
        # Camera's up direction in camera coordinates (negative Y axis)
        up_cam = np.array([0, -1, 0, 0])
        # Transform to world coordinates
        up_world = cam_to_world[i] @ up_cam
        up_vectors.append(up_world[:3])
    
    # Average up vector across all cameras
    avg_up = np.mean(up_vectors, axis=0)
    avg_up_norm = np.linalg.norm(avg_up)
    
    if avg_up_norm < 1e-6:
        print("Warning: Average up vector is zero. Using identity transform.")
        return np.eye(4)
    
    avg_up = avg_up / avg_up_norm
    
    # Target: align average up with world Z-axis (0, 0, 1)
    target_up = np.array([0, 0, 1])
    
    # Compute rotation using Rodrigues' formula
    v = np.cross(avg_up, target_up)
    s = np.linalg.norm(v)
    c = np.dot(avg_up, target_up)
    
    # Check if already aligned or opposite
    if s < 1e-6:
        if c > 0:  # Already aligned
            print("Cameras already aligned with gravity (Z-axis up)")
            return np.eye(4)
        else:  # Opposite direction, need 180 degree rotation
            print("Warning: Cameras pointing opposite to gravity. Rotating 180 degrees.")
            R = -np.eye(3)
            R[2, 2] = 1  # Keep Z positive
    else:
        # Skew-symmetric cross-product matrix
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        
        # Rodrigues' rotation formula: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    
    alignment = np.eye(4)
    alignment[:3, :3] = R
    
    # Compute angle of rotation for logging
    rotation_angle = np.arccos(np.clip(c, -1.0, 1.0)) * 180 / np.pi
    
    print(f"Gravity alignment computed from {num_cameras} inferred camera poses")
    print(f"  Average camera up vector (world): [{avg_up[0]:.3f}, {avg_up[1]:.3f}, {avg_up[2]:.3f}]")
    print(f"  Rotation angle to align with Z-axis: {rotation_angle:.2f}°")
    
    return alignment
