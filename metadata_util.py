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


def compute_gravity_alignment_from_single_camera(
    pitch_angles: List[Optional[float]],
    extrinsics: np.ndarray,
    camera_idx: int,
) -> np.ndarray:
    """
    Compute gravity alignment using a single camera as reference.
    This provides more precise alignment when one camera has reliable pitch data.
    
    IMPORTANT: Camera coordinate system has Z-axis pointing forward (viewing direction),
    but physical world has Z-axis pointing up (vertical/gravity direction).
    This function computes the alignment to make world Z-axis point up.
    
    Args:
        pitch_angles: List of pitch angles in degrees from EXIF/metadata
                     (pitch relative to horizontal: -90° = horizontal, 0° = pointing down)
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4) from model
        camera_idx: Index of the camera to use as reference (0-based)
        
    Returns:
        4x4 transformation matrix to align world z-axis with gravity (vertical up)
    """
    # Validate camera index
    if camera_idx < 0 or camera_idx >= len(extrinsics):
        print(f"Warning: Invalid camera index {camera_idx}. Using camera 0.")
        camera_idx = 0
    
    # Get pitch angle for the reference camera
    if camera_idx >= len(pitch_angles):
        print(f"Warning: No pitch angle for camera {camera_idx}. Using average.")
        valid_pitches = [p for p in pitch_angles if p is not None]
        if valid_pitches:
            pitch_value = np.mean(valid_pitches)
        else:
            print("Warning: No valid pitch angles. Computing from camera pose only.")
            return compute_gravity_alignment_from_camera_vectors(extrinsics, camera_idx)
    else:
        pitch_value = pitch_angles[camera_idx]
        if pitch_value is None:
            valid_pitches = [p for p in pitch_angles if p is not None]
            if valid_pitches:
                pitch_value = np.mean(valid_pitches)
                print(f"Camera {camera_idx} has no pitch data, using average: {pitch_value:.2f}°")
            else:
                print("Warning: No valid pitch angles. Computing from camera pose only.")
                return compute_gravity_alignment_from_camera_vectors(extrinsics, camera_idx)
    
    print(f"Using camera {camera_idx} with pitch {pitch_value:.2f}° for gravity alignment")
    
    # Prepare extrinsics as 4x4 matrices
    if extrinsics.shape[-2:] == (3, 4):
        num_cams = len(extrinsics)
        ext_4x4 = np.zeros((num_cams, 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1
    else:
        ext_4x4 = extrinsics
    
    # Get the world-to-camera matrix for the reference camera
    world_to_cam = ext_4x4[camera_idx]
    
    try:
        cam_to_world = np.linalg.inv(world_to_cam)
    except np.linalg.LinAlgError:
        print(f"Warning: Could not invert camera matrix for camera {camera_idx}")
        return np.eye(4)
    
    # Extract rotation matrix
    R_cw = cam_to_world[:3, :3]
    
    # Camera coordinate system: X=down, Y=left, Z=forward
    # Camera's up direction is -X (opposite of down)
    up_camera = np.array([-1, 0, 0])
    
    # Transform camera's up vector to world coordinates
    up_world_from_pose = R_cw @ up_camera
    up_norm = np.linalg.norm(up_world_from_pose)
    
    if up_norm < 1e-6:
        print("Warning: Up vector is degenerate")
        return np.eye(4)
    
    up_world_from_pose = up_world_from_pose / up_norm
    
    # Calculate predicted pitch from the up vector
    # pitch = -90° means camera horizontal (up vector perpendicular to gravity)
    # pitch = 0° means camera vertical (up vector parallel to gravity)
    # The angle between up vector and world Z-axis tells us the pitch
    pred_pitch = 90.0 - np.degrees(np.arccos(np.clip(up_world_from_pose[2], -1.0, 1.0)))
    print(f"  Predicted pitch from camera up vector: {pred_pitch:.2f}°")
    
    # Unwrap pitch value to match predicted pitch
    candidates = [pitch_value, pitch_value - 180.0, pitch_value + 180.0]
    adjusted_pitch = min(candidates, key=lambda ang: abs(ang - pred_pitch))
    print(f"  Adjusted EXIF pitch: {adjusted_pitch:.2f}°")
    
    # Build expected up vector from pitch angle
    # pitch = -90° -> up vector horizontal (perpendicular to Z)
    # pitch = 0° -> up vector vertical (parallel to Z)
    pitch_angle_from_horizontal = adjusted_pitch + 90.0  # Convert to angle from horizontal
    
    # The up vector should have:
    # - Z component based on pitch: sin(pitch_from_horizontal)
    # - Horizontal component: cos(pitch_from_horizontal)
    expected_z = np.sin(np.radians(pitch_angle_from_horizontal))
    expected_horizontal_mag = np.cos(np.radians(pitch_angle_from_horizontal))
    
    # Keep horizontal direction from predicted up vector
    horizontal = up_world_from_pose.copy()
    horizontal[2] = 0.0
    horiz_norm = np.linalg.norm(horizontal)
    
    if horiz_norm < 1e-6:
        # Up vector is purely vertical - just use Z direction
        expected_up = np.array([0.0, 0.0, np.sign(expected_z) if abs(expected_z) > 1e-6 else 1.0])
    else:
        horizontal_dir = horizontal / horiz_norm
        expected_up = horizontal_dir * expected_horizontal_mag + np.array([0.0, 0.0, expected_z])
    
    exp_norm = np.linalg.norm(expected_up)
    if exp_norm < 1e-6:
        print("Warning: Expected up vector is degenerate")
        return np.eye(4)
    
    expected_up = expected_up / exp_norm
    
    # Now we need to align up_world_from_pose to expected_up
    # But our goal is to align the world Z-axis with gravity (up direction)
    # So we compute rotation to align expected_up to world Z-axis [0, 0, 1]
    target_up = np.array([0.0, 0.0, 1.0])
    
    # Compute rotation using Rodrigues' formula
    v = np.cross(expected_up, target_up)
    s = np.linalg.norm(v)
    c = np.dot(expected_up, target_up)
    
    gravity_alignment = np.eye(4)
    
    if s < 1e-6:
        if c < 0:
            # 180-degree rotation: choose an arbitrary axis orthogonal to expected_up
            axis = np.cross(expected_up, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(expected_up, np.array([0.0, 1.0, 0.0]))
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                return gravity_alignment
            axis = axis / axis_norm
            gravity_alignment[:3, :3] = Rotation.from_rotvec(np.pi * axis).as_matrix()
            print("  Applied 180° flip to align up vectors")
        else:
            print("  Up vector already aligned with world Z-axis (vertical)")
        return gravity_alignment
    
    # Rodrigues' rotation formula
    vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )
    
    rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    gravity_alignment[:3, :3] = rotation_matrix
    
    rotation_angle = np.degrees(np.arctan2(s, c))
    print(f"  Rotation angle to align: {rotation_angle:.2f}°")
    
    return gravity_alignment


def compute_gravity_alignment_from_pitch_and_poses(
    pitch_angles: List[Optional[float]],
    extrinsics: np.ndarray,
    reference_camera_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Compute gravity alignment by combining EXIF pitch data (absolute reference)
    with inferred camera poses (relative geometry).
    
    IMPORTANT: This aligns the world Z-axis with the gravity direction (vertical up),
    not with the camera's forward direction.
    
    Args:
        pitch_angles: List of pitch angles in degrees from EXIF/metadata
                     (pitch relative to horizontal: -90° = horizontal, 0° = pointing down)
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4) from model
        reference_camera_idx: Optional index of camera to use as reference (0-based).
                            If None, uses average of all cameras.
        
    Returns:
        4x4 transformation matrix to align world z-axis with gravity (vertical up)
    """
    # Filter out None values
    valid_pitches = [p for p in pitch_angles if p is not None]

    if not valid_pitches:
        print("Warning: No valid pitch angles found. Computing alignment from camera poses only.")
        return compute_gravity_alignment_from_camera_vectors(extrinsics, reference_camera_idx)

    # If reference camera is specified, use it for calibration
    if reference_camera_idx is not None:
        print(f"Using camera #{reference_camera_idx} as reference for gravity alignment")
        return compute_gravity_alignment_from_single_camera(
            pitch_angles, extrinsics, reference_camera_idx
        )

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
    pred_up_vectors = []
    target_up_vectors = []
    aligned_pitch_values = []

    for idx, world_to_cam in enumerate(ext_4x4):
        try:
            cam_to_world = np.linalg.inv(world_to_cam)
        except np.linalg.LinAlgError:
            continue

        R_cw = cam_to_world[:3, :3]
        
        # Camera coordinate system: X=down, Y=left, Z=forward
        # Camera's up direction is -X (opposite of down)
        up_camera = np.array([-1, 0, 0])
        up_world = R_cw @ up_camera
        up_norm = np.linalg.norm(up_world)
        if up_norm < 1e-6:
            continue
        up_world = up_world / up_norm
        pred_up_vectors.append(up_world)

        # Calculate predicted pitch from the up vector
        # pitch = -90° means camera horizontal (up vector perpendicular to Z)
        # pitch = 0° means camera pointing down (up vector parallel to Z)
        pred_pitch = 90.0 - np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0)))
        predicted_pitches.append(pred_pitch)

        # Build expected up vector that matches EXIF pitch but keeps horizontal direction
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

        # Convert pitch to up vector components
        pitch_angle_from_horizontal = adjusted_pitch + 90.0
        expected_z = np.sin(np.radians(pitch_angle_from_horizontal))
        expected_horizontal_mag = np.cos(np.radians(pitch_angle_from_horizontal))

        # Keep horizontal direction from predicted up vector
        horizontal = up_world.copy()
        horizontal[2] = 0.0
        horiz_norm = np.linalg.norm(horizontal)

        if horiz_norm < 1e-6:
            expected_up = np.array([0.0, 0.0, np.sign(expected_z) if abs(expected_z) > 1e-6 else 1.0])
        else:
            horizontal_dir = horizontal / horiz_norm
            expected_up = horizontal_dir * expected_horizontal_mag + np.array([0.0, 0.0, expected_z])

        exp_norm = np.linalg.norm(expected_up)
        if exp_norm < 1e-6:
            expected_up = up_world
        else:
            expected_up = expected_up / exp_norm

        target_up_vectors.append(expected_up)

    if not predicted_pitches:
        print("Warning: Could not derive predicted pitches from camera poses. Falling back to EXIF-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    if not predicted_pitches:
        print("Warning: Could not derive predicted pitches from camera poses. Falling back to EXIF-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    avg_pred_pitch = np.mean(predicted_pitches)
    print(f"Average predicted pitch (from camera poses): {avg_pred_pitch:.2f}°")

    if aligned_pitch_values:
        avg_aligned_pitch = np.mean(aligned_pitch_values)
        print(f"Average EXIF pitch after unwrapping: {avg_aligned_pitch:.2f}°")

    if not target_up_vectors:
        print("Warning: Could not build expected up vectors, using pitch-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    # Average the up vectors
    avg_pred_up = np.mean(pred_up_vectors, axis=0)
    avg_target_up = np.mean(target_up_vectors, axis=0)

    pred_norm = np.linalg.norm(avg_pred_up)
    target_norm = np.linalg.norm(avg_target_up)

    if pred_norm < 1e-6 or target_norm < 1e-6:
        print("Warning: Up vectors degenerate, using pitch-only alignment.")
        return compute_gravity_alignment_from_pitch(pitch_angles)

    avg_pred_up = avg_pred_up / pred_norm
    avg_target_up = avg_target_up / target_norm

    # Align target_up to world Z-axis (vertical up)
    target_world_up = np.array([0.0, 0.0, 1.0])
    
    v = np.cross(avg_target_up, target_world_up)
    s = np.linalg.norm(v)
    c = np.dot(avg_target_up, target_world_up)

    gravity_alignment = np.eye(4)

    if s < 1e-6:
        if c < 0:
            # 180-degree rotation: choose an arbitrary axis orthogonal to avg_target_up
            axis = np.cross(avg_target_up, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(avg_target_up, np.array([0.0, 1.0, 0.0]))
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                return gravity_alignment  # give up, identity
            axis = axis / axis_norm
            gravity_alignment[:3, :3] = Rotation.from_rotvec(np.pi * axis).as_matrix()
            print("Applied 180° flip to align up vectors with world Z-axis.")
            return gravity_alignment
        else:
            print("Up vectors already aligned with world Z-axis (vertical).")
            return gravity_alignment

    vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

    rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    gravity_alignment[:3, :3] = rotation_matrix

    rotation_angle = np.degrees(np.arctan2(s, c))
    print(f"Up-vector alignment rotation to vertical: {rotation_angle:.2f}°")

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


def compute_gravity_alignment_from_camera_vectors(extrinsics: np.ndarray, reference_camera_idx: Optional[int] = None) -> np.ndarray:
    """
    Compute gravity alignment by analyzing the up vectors of cameras.
    Uses the actual inferred camera poses from the reconstruction model.
    
    IMPORTANT: Aligns world Z-axis with gravity (vertical up direction).
    Camera coordinate system convention:
    - X-axis: down
    - Y-axis: left
    - Z-axis: forward (viewing direction)
    - Up direction: -X
    
    Args:
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4)
        reference_camera_idx: Optional index of camera to use as reference (0-based).
                            If None, uses average of all cameras.
        
    Returns:
        4x4 transformation matrix to align world z-axis with gravity (vertical up)
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
    
    # If using a single reference camera
    if reference_camera_idx is not None:
        if reference_camera_idx < 0 or reference_camera_idx >= num_cameras:
            print(f"Warning: Invalid camera index {reference_camera_idx}. Using camera 0.")
            reference_camera_idx = 0
        
        print(f"Computing gravity alignment from camera #{reference_camera_idx} up vector")
        
        # Camera coordinate system: X=down, Y=left, Z=forward
        # Camera's up direction is -X (opposite of down)
        up_cam = np.array([-1, 0, 0, 0])
        # Transform to world coordinates
        avg_up = (cam_to_world[reference_camera_idx] @ up_cam)[:3]
    else:
        # Extract up vectors from camera coordinate systems
        # Camera coordinate system: X=down, Y=left, Z=forward
        # Camera's up direction is -X (opposite of down)
        up_vectors = []
        for i in range(num_cameras):
            # Camera coordinate system: X=down, Y=left, Z=forward
            # Camera's up direction is -X (opposite of down)
            up_cam = np.array([-1, 0, 0, 0])
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
    
    if reference_camera_idx is not None:
        print(f"Gravity alignment computed from camera #{reference_camera_idx} pose")
        print(f"  Camera up vector (world): [{avg_up[0]:.3f}, {avg_up[1]:.3f}, {avg_up[2]:.3f}]")
        print(f"  Rotation angle to align with Z-axis: {rotation_angle:.2f}°")
    else:
        print(f"Gravity alignment computed from {num_cameras} inferred camera poses")
        print(f"  Average camera up vector (world): [{avg_up[0]:.3f}, {avg_up[1]:.3f}, {avg_up[2]:.3f}]")
        print(f"  Rotation angle to align with Z-axis: {rotation_angle:.2f}°")
    
    return alignment
