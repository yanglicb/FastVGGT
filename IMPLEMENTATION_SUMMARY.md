# Summary: Gravity Alignment Implementation for FastVGGT

## Overview
Successfully implemented gravity alignment feature that uses pitch angle metadata to align the output point cloud's z-axis with the real-world vertical direction.

## Changes Made

### 1. New File: `metadata_util.py`
Created comprehensive utilities for extracting and processing pitch angle metadata:

- `extract_pitch_from_exif()` - Extracts pitch from image EXIF data
- `extract_pitch_from_filename()` - Parses pitch from filename patterns
- `load_pitch_from_json()` - Loads pitch from JSON metadata files
- `get_pitch_angles()` - Main function that tries all methods in order
- `compute_gravity_alignment_from_pitch()` - Computes rotation matrix from pitch angles
- `compute_gravity_alignment_from_camera_vectors()` - Fallback method using camera orientations

### 2. Modified: `visual_util.py`
Updated visualization functions to support gravity alignment:

- Modified `predictions_to_glb()` to accept `pitch_angles` parameter
- Updated `apply_scene_alignment()` to apply optional gravity alignment transformation
- Gravity alignment is applied after standard scene alignment
- Affects both point cloud and camera visualizations

### 3. Modified: `demo_gradio.py`
Enhanced Gradio interface with gravity alignment controls:

- Added new UI checkbox: "Align with Gravity (from pitch metadata)"
- Import `get_pitch_angles` from metadata_util
- Updated `gradio_demo()` to load and use pitch angles
- Updated `update_visualization()` to support gravity alignment
- Modified all event handlers to pass gravity alignment parameter
- GLB filenames now include gravity alignment state for proper caching

### 4. Documentation: `GRAVITY_ALIGNMENT.md`
Comprehensive user guide covering:
- How the feature works
- Three methods for providing pitch data (JSON, EXIF, filename)
- Pitch angle conventions and coordinate system
- Example workflow with code samples
- Troubleshooting common issues
- Advanced programmatic usage

### 5. Example: `metadata_example.json`
Template JSON file showing metadata format for pitch angles

## How It Works

1. **Data Collection**: System attempts to load pitch angles from:
   - JSON metadata file (preferred)
   - Image EXIF tags
   - Filename parsing
   - Camera orientation analysis (fallback)

2. **Alignment Computation**: 
   - Averages valid pitch angles across all images
   - Creates rotation matrix around X-axis to compensate for pitch
   - Negative rotation applied to counter-rotate the world

3. **Scene Transformation**:
   - Standard scene alignment applied first (camera-relative)
   - Gravity alignment applied on top (world-relative)
   - Result: Z-axis points up in real-world coordinates

## Usage

### For Users:
1. Create a `metadata.json` file with pitch angles for your images
2. Place it in your input directory
3. Check the "Align with Gravity" checkbox in the UI
4. Run reconstruction

### Metadata Format:
```json
{
  "image001.jpg": -15.5,
  "image002.jpg": -12.3,
  "image003.jpg": -10.8
}
```

Or with additional fields:
```json
{
  "image001.jpg": {"pitch": -15.5, "roll": 0, "yaw": 45}
}
```

### Alternative Methods:
- Encode in filenames: `image_pitch_-15.5.jpg`
- Store in EXIF tags (automatic extraction)

## Benefits

1. **Real-World Alignment**: Point clouds align with actual gravity direction
2. **Flexibility**: Multiple input methods (JSON, EXIF, filename)
3. **Graceful Fallback**: Works even without metadata (uses camera analysis)
4. **User Control**: Toggle on/off via UI checkbox
5. **Proper Caching**: Different alignments generate different GLB files

## Technical Details

- Pitch convention: Positive = up, Negative = down, in degrees
- Rotation applied around X-axis (side-to-side tilt)
- Transformation matrix: 4x4 homogeneous coordinates
- Fallback uses camera up-vectors to estimate gravity direction
- Compatible with both "Depthmap" and "Pointmap" prediction modes

## Testing Recommendations

1. **Test with known pitch angles**: Use metadata.json with known values
2. **Verify z-axis direction**: Point cloud should have z-up after alignment
3. **Test fallback**: Try without metadata to verify camera analysis works
4. **Check UI toggle**: Verify alignment can be enabled/disabled
5. **Cache verification**: Different settings should create different GLB files

## Future Enhancements (Optional)

- Support for roll angle (side tilt correction)
- Support for yaw angle (compass direction)
- Full IMU/GPS metadata integration
- Automatic metadata extraction from video files
- Visualization of gravity vector in 3D viewer
- Multiple coordinate system export options
