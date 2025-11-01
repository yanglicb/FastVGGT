# Coordinate System Fix for Gravity Alignment

## Problem Identified

The gravity alignment was incorrectly using the camera's **forward direction** (Z-axis in camera coordinates) instead of the **up direction** (gravity reference) to compute the alignment transformation.

### Coordinate System Conventions

**Camera Coordinate System** (OpenCV convention):
- **X-axis**: Points to the right
- **Y-axis**: Points down
- **Z-axis**: Points forward (viewing/optical axis direction)

**World/Physical Coordinate System** (desired):
- **Z-axis**: Points up (vertical, opposite to gravity)
- X and Y axes: Horizontal plane

### The Issue

The original code was:
1. Extracting the camera's **forward vector** (Z-axis in camera space)
2. Computing pitch from the forward vector's vertical component
3. Aligning the forward vector based on pitch metadata

This was incorrect because:
- **Pitch metadata** describes rotation around the horizontal axis (tilt up/down)
- Pitch relates to the **up direction** (perpendicular to gravity), not the forward direction
- Camera Z (forward) and World Z (up) are fundamentally different concepts

## Solution

Updated the alignment functions to work with the camera's **up vector** instead:

1. **Extract up vector**: Camera's up direction is `-Y` in camera coordinates (since Y points down)
2. **Transform to world**: Apply camera-to-world rotation to get up vector in world space
3. **Compute pitch from up vector**: `pitch = 90° - arccos(up_z)`
4. **Build expected up vector** from EXIF pitch metadata
5. **Align to world Z-axis**: Compute rotation to make world Z-axis point up (vertical)

### Mathematical Details

#### Camera Up Vector Extraction
```python
up_camera = np.array([0, -1, 0])  # -Y axis in camera coords
up_world = R_camera_to_world @ up_camera
```

#### Pitch from Up Vector
```python
# pitch = -90° → camera horizontal (up ⊥ gravity)
# pitch = 0° → camera vertical pointing down (up ∥ gravity)
pitch = 90° - arccos(up_world[2])
```

#### Expected Up Vector from Pitch
```python
pitch_from_horizontal = pitch + 90°
expected_z = sin(pitch_from_horizontal)
expected_horizontal = cos(pitch_from_horizontal) * horizontal_direction
expected_up = expected_horizontal + [0, 0, expected_z]
```

#### Gravity Alignment
```python
target = [0, 0, 1]  # World Z-axis (vertical up)
rotation = rodrigues_formula(expected_up, target)
```

## Changes Made

### `compute_gravity_alignment_from_single_camera()`
- Changed from forward vector to up vector analysis
- Updated pitch calculation to use up vector angle with Z-axis
- Aligned result to world Z-axis (vertical) instead of forward direction
- Added clarifying docstrings about coordinate systems

### `compute_gravity_alignment_from_pitch_and_poses()`
- Replaced `pred_forward_vectors` with `pred_up_vectors`
- Replaced `target_forward_vectors` with `target_up_vectors`
- Changed loop to extract and process up vectors
- Updated final alignment to target world Z-axis
- Added coordinate system documentation

### `compute_gravity_alignment_from_camera_vectors()`
- Already used up vectors (was correct)
- Added clarifying docstrings
- No algorithmic changes needed

## Verification

The test script (`test_gravity_alignment.py`) now includes:
- Coordinate system convention documentation
- Explicit verification that camera up vectors align to world Z-axis
- Error measurement for alignment quality

### Expected Behavior After Fix

1. **Before alignment**: Camera up vectors point in various directions in world space
2. **After alignment**: World Z-axis points vertically up (against gravity)
3. **Result**: The 3D reconstruction is properly oriented with respect to gravity

## Impact

This fix ensures that:
- ✅ Gravity alignment actually aligns with gravity (vertical up)
- ✅ Pitch metadata is correctly interpreted as tilt from horizontal
- ✅ The world coordinate system has Z pointing up (standard convention)
- ✅ Visualization axes correctly show vertical reference
- ✅ Camera poses are preserved while scene is properly oriented

## Files Modified

1. `/home/yang/3d-recon/FastVGGT/metadata_util.py`
   - `compute_gravity_alignment_from_single_camera()` - Complete rewrite
   - `compute_gravity_alignment_from_pitch_and_poses()` - Updated to use up vectors
   - `compute_gravity_alignment_from_camera_vectors()` - Documentation update

2. `/home/yang/3d-recon/FastVGGT/test_gravity_alignment.py`
   - Added coordinate system verification
   - Added explicit up vector alignment test

## Testing Recommendations

1. **Visual Check**: After alignment, the coordinate axes should show Z (blue) pointing up
2. **Pitch Verification**: Objects on the ground should be perpendicular to Z-axis
3. **Camera Orientation**: Cameras should maintain correct viewing directions
4. **Metadata Correlation**: Scenes with pitch metadata should align better than before

## Technical Notes

### Why This Matters

In 3D reconstruction and computer vision:
- Camera parameters describe **where the camera looks** (orientation)
- Gravity alignment describes **which way is up** in the world
- These are orthogonal concepts that must both be preserved

The camera's viewing direction (forward) can point anywhere while the world's up direction stays fixed (gravity). Confusing these leads to incorrect scene orientation.

### Rodrigues' Formula

The rotation from vector `a` to vector `b` is computed using:
```
v = a × b  (cross product)
s = ||v||  (sine of angle)
c = a · b  (cosine of angle)
R = I + [v]× + [v]×² * (1-c)/s²
```

This formula correctly handles all cases including:
- Small angles (numerical stability)
- 180° rotations (fallback to arbitrary orthogonal axis)
- Already aligned vectors (identity)
