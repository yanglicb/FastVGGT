# Camera Coordinate System Correction

## Issue
The camera coordinate system was incorrectly assumed to follow OpenCV convention (X=right, Y=down, Z=forward), but the actual system has **Y=right**.

## Correct Camera Coordinate System

Based on Y pointing right and assuming a right-handed coordinate system:

```
Camera Local Coordinates:
- X-axis: forward (optical/viewing axis)
- Y-axis: right  
- Z-axis: up
```

This is different from the commonly assumed OpenCV convention.

## Changes Made

Updated the camera up vector extraction in all gravity alignment functions:

### Before (Incorrect)
```python
# Assumed: X=right, Y=down, Z=forward (OpenCV)
up_camera = np.array([0, -1, 0])  # -Y axis (wrong!)
```

### After (Correct)
```python
# Actual: X=forward, Y=right, Z=up
up_camera = np.array([0, 0, 1])  # +Z axis (correct!)
```

## Updated Functions

1. **`compute_gravity_alignment_from_single_camera()`**
   - Changed `up_camera = [0, -1, 0]` → `up_camera = [0, 0, 1]`

2. **`compute_gravity_alignment_from_pitch_and_poses()`**
   - Changed `up_camera = [0, -1, 0]` → `up_camera = [0, 0, 1]`

3. **`compute_gravity_alignment_from_camera_vectors()`**
   - Changed `up_cam = [0, -1, 0, 0]` → `up_cam = [0, 0, 1, 0]`
   - Updated docstring

4. **Test file (`test_gravity_alignment.py`)**
   - Updated coordinate system documentation
   - Changed test vector from `[0, -1, 0, 0]` to `[0, 0, 1, 0]`

## Coordinate System Conventions

### Camera Frame (Local)
```
     Z (up)
     |
     |
     +---- Y (right)
    /
   X (forward/optical axis)
```

### World Frame (After Gravity Alignment)
```
     Z (vertical up, opposite to gravity)
     |
     |
     +---- Y
    /
   X

(X and Y span the horizontal plane)
```

## Verification

The gravity alignment now:
✅ Correctly extracts the camera's up direction (+Z in camera frame)
✅ Transforms it to world coordinates using camera-to-world rotation
✅ Computes the rotation to align world Z-axis with gravity (vertical up)
✅ Preserves the camera's forward direction (X-axis)
✅ Maintains right-handed coordinate system throughout

## Impact

This correction ensures:
- Camera up vectors are properly identified in the camera frame
- Gravity alignment correctly orients the scene vertically
- Pitch angles are interpreted correctly relative to the camera's actual up direction
- The visualization axes show the correct coordinate system

## Files Modified

1. `/home/yang/3d-recon/FastVGGT/metadata_util.py`
2. `/home/yang/3d-recon/FastVGGT/test_gravity_alignment.py`
