# Gravity Alignment Updates - Single Camera Calibration

## Summary of Changes

The gravity alignment feature has been enhanced to support single camera calibration, allowing for more precise alignment based on a specific reference camera rather than averaging across all cameras.

## Key Features Added

### 1. Single Camera Reference Selection
- **New UI Element**: Added a dropdown menu "Reference Camera for Gravity Alignment" in the Gradio interface
- **Default Behavior**: Uses "Average (All Cameras)" for backward compatibility
- **Per-Camera Selection**: Users can now select any specific camera (Camera 0, Camera 1, etc.) as the reference

### 2. Enhanced Visualization
- **Before/After Axes**: Coordinate axes are now shown both before and after gravity alignment
  - **Original axes** (lighter colors): Shows the coordinate system before alignment
  - **Aligned axes** (bright colors): Shows the coordinate system after alignment
- **Color Coding**:
  - Light colors (semi-transparent): Pre-alignment orientation
  - Bright colors (opaque): Post-alignment orientation
  - X = Red, Y = Green, Z = Blue

### 3. Updated Functions

#### `metadata_util.py`
- **`compute_gravity_alignment_from_single_camera()`**: NEW function
  - Computes gravity alignment using a single camera as reference
  - Provides more precise alignment when one camera has reliable pitch data
  - Handles pitch angle unwrapping and validation
  
- **`compute_gravity_alignment_from_pitch_and_poses()`**: UPDATED
  - Added `reference_camera_idx` parameter
  - Automatically delegates to single-camera function when reference is specified
  - Falls back to average-based computation when reference_camera_idx is None
  
- **`compute_gravity_alignment_from_camera_vectors()`**: UPDATED
  - Added `reference_camera_idx` parameter
  - Supports single-camera up-vector analysis
  - Maintains backward compatibility with average-based approach

#### `visual_util.py`
- **`predictions_to_glb()`**: UPDATED
  - Added `reference_camera_idx` parameter
  - Passes reference camera index to gravity alignment functions
  - Added before/after axis visualization
  
- **`create_coordinate_axes()`**: UPDATED
  - Added `prefix` parameter for unique geometry naming
  - Added `colors` parameter for custom RGBA colors
  - Supports multiple axis sets in the same scene

#### `demo_gradio.py`
- **UI Updates**:
  - Added reference camera dropdown
  - Dropdown is populated with camera options after reconstruction
  - All event handlers updated to include reference_camera_idx
  
- **`gradio_demo()`**: UPDATED
  - Added `reference_camera_idx` parameter
  - Converts dropdown selection to integer index
  - Updates camera dropdown choices based on number of frames
  
- **`update_visualization()`**: UPDATED
  - Added `reference_camera_idx` parameter
  - Converts dropdown selection to integer index for visualization updates
  
- **`example_pipeline()`**: UPDATED
  - Added `reference_camera_idx` parameter
  - Returns camera dropdown in outputs

## Usage

### Basic Usage (Average of All Cameras)
1. Upload images/video
2. Click "Reconstruct"
3. Enable "Align with Gravity" checkbox
4. Leave "Reference Camera for Gravity Alignment" set to "Average (All Cameras)"

### Advanced Usage (Single Camera Reference)
1. Upload images/video
2. Click "Reconstruct"
3. Enable "Align with Gravity" checkbox
4. Select a specific camera from the dropdown (e.g., "Camera 0", "Camera 1")
5. The alignment will be computed based solely on that camera's orientation

### Visualization Understanding
- **Lighter colored axes**: Show where the coordinate system was before alignment
- **Bright colored axes**: Show where the coordinate system is after alignment
- The transformation between these two sets of axes represents the gravity alignment correction

## Technical Details

### Single Camera Algorithm
1. Extracts pitch angle from EXIF metadata for the selected camera
2. Computes predicted pitch from camera pose (forward vector)
3. Unwraps pitch value to handle sensor reporting quirks (0° to -90° transitions)
4. Builds expected forward vector based on EXIF pitch
5. Computes rotation matrix using Rodrigues' formula
6. Returns 4x4 transformation matrix

### Benefits of Single Camera Reference
- **More Accurate**: When one camera has particularly reliable pitch metadata
- **Consistent**: Avoids averaging errors when cameras have varying pitch qualities
- **Debuggable**: Easier to identify which camera orientation is being used
- **Flexible**: Can test different cameras to find the best reference

## Backward Compatibility

All changes maintain full backward compatibility:
- Default behavior uses average of all cameras (existing behavior)
- Functions accept `None` for `reference_camera_idx` to trigger average mode
- UI defaults to "Average (All Cameras)"
- Existing code without the new parameter continues to work

## Files Modified

1. `/home/yang/3d-recon/FastVGGT/metadata_util.py`
2. `/home/yang/3d-recon/FastVGGT/visual_util.py`
3. `/home/yang/3d-recon/FastVGGT/demo_gradio.py`

## Testing Recommendations

1. Test with images that have reliable EXIF pitch data
2. Compare results between "Average" and individual camera selections
3. Verify that axes visualization clearly shows the transformation
4. Check that alignment quality improves with appropriate camera selection
