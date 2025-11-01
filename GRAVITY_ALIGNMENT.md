# Gravity Alignment using Pitch Metadata

This feature allows you to align the output point cloud's z-axis with the real-world vertical direction using pitch angle information from your image metadata.

## How It Works

The system combines **EXIF pitch metadata** (absolute reference) with **inferred camera poses** (relative geometry) for accurate gravity alignment:

1. **Extract pitch angles** from image EXIF data (e.g., `Pitch:-71.351295`)
2. **Interpret the absolute reference**: 
   - pitch = -90° means camera is **horizontal** (normal phone orientation)
   - pitch = 0° means camera is **vertical** (pointing straight down or up)
   - pitch = -70° means camera is tilted ~20° from horizontal
3. **Compute rotation** needed to align with gravity based on this absolute reference
4. **Apply transformation** to the entire scene (point cloud, cameras, axes)

### Why Both Are Needed

- **EXIF Pitch**: Provides the **absolute orientation** relative to Earth's gravity (from physical sensors)
- **Inferred Poses**: Provide **relative camera geometry** (from the reconstruction model)
- **Combined**: EXIF pitch calibrates the inferred poses to real-world coordinates

Without EXIF pitch data, the model has no way to know which direction is "up" in the real world - it only knows the relative orientations between cameras.

## Usage Methods

### Method 1: JSON Metadata File (Recommended)

Create a `metadata.json` file in your input directory with the following format:

```json
{
  "image001.jpg": {"pitch": -15.5},
  "image002.jpg": {"pitch": -12.3},
  "image003.jpg": {"pitch": -18.7}
}
```

Or simplified format:

```json
{
  "image001.jpg": -15.5,
  "image002.jpg": -12.3,
  "image003.jpg": -18.7
}
```

Place this file in the same directory as your images or in the target directory created by the demo.

### Method 2: EXIF Metadata

If your images contain pitch information in EXIF tags (common with drones or specialized cameras), the system will automatically extract it.

**Supported EXIF Formats:**

1. **User Comment field with CSV format** (most common for drones):
   ```
   Pitch:-71.351295,Roll:-10.72634,Azimuth:-154.74142
   ```
   The system will parse this format and extract the pitch value.

2. **Standard EXIF tags**: Any EXIF tag containing "pitch" in its name
3. **ImageDescription or Comment fields**: Similar CSV or plain text format

**Testing EXIF extraction:**

You can test if your images contain pitch data:

```bash
# Extract metadata from your images
python create_metadata.py --exif ./my_images --output test_metadata.json

# This will show which images have pitch data and the extracted values
```

### Method 3: Filename Encoding

Encode the pitch angle in your filenames:

```
image_pitch_-15.5.jpg
frame001_pitch_12.3.png
scene_pitch_-8.0_frame001.jpg
```

The system will parse any number following "pitch" in the filename.

## Pitch Angle Convention

Understanding the pitch convention from your camera/phone sensors:

- **pitch = -90°**: Camera is **horizontal** (normal phone usage, held upright)
- **pitch = -70°**: Camera tilted **20° downward** from horizontal (typical drone angle)
- **pitch = -45°**: Camera tilted **45° downward** from horizontal  
- **pitch = 0°**: Camera is **vertical** (pointing straight down or up)
- **pitch = +90°**: Camera is **horizontal** (upside down orientation)

### Example: Drone Photography

For aerial/drone imagery where the camera points downward:
- EXIF shows: `Pitch:-71.351295` 
- Interpretation: Camera is 18.65° from horizontal, pointing mostly downward
- System rotation: ~19° to align Z-axis with real-world up


## Coordinate System

After gravity alignment:
- **Z-axis**: Points upward (aligned with real-world vertical)
- **X-axis**: Points to the right
- **Y-axis**: Points forward (perpendicular to Z and X)

## UI Control

When running the Gradio demo, you'll see a checkbox:

**"Align with Gravity (from pitch metadata)"**

- ✓ Checked (default): Apply gravity alignment using pitch metadata
- ☐ Unchecked: Use default alignment (scene-relative)

## Example Workflow

1. **Prepare your data**:
   ```bash
   my_dataset/
   ├── images/
   │   ├── frame001.jpg
   │   ├── frame002.jpg
   │   └── frame003.jpg
   └── metadata.json
   ```

2. **Create metadata.json**:
   ```json
   {
     "frame001.jpg": -12.5,
     "frame002.jpg": -15.0,
     "frame003.jpg": -10.8
   }
   ```

3. **Upload and reconstruct**:
   - Upload your images through the Gradio interface
   - Ensure "Align with Gravity" is checked
   - Click "Reconstruct"

4. **Result**:
   The output point cloud will be aligned with the real-world vertical direction!

## Fallback Behavior

**EXIF pitch data is required** for gravity alignment to work correctly. If no pitch metadata is found:

- The system will display a warning
- No gravity alignment will be applied (identity transform)
- The reconstruction will use only the model's relative camera poses
- The scene orientation will be arbitrary relative to real-world gravity

To ensure proper alignment, make sure your images contain EXIF pitch data or provide a metadata.json file.

## Technical Details

The gravity alignment is computed by:

1. **Extracting pitch from EXIF**: Read pitch angles from image metadata (UserComment field: `Pitch:-71.351295,Roll:...`)
2. **Averaging pitch values**: Compute mean pitch across all images to handle variations
3. **Aligning forward vectors**:
   - Compute average forward direction from inferred camera poses
   - Unwrap EXIF pitch (sensor reports 0→-90 when tilting down, then -90→0 when tilting up)
   - Adjust vertical component to match the unwrapped pitch while keeping horizontal heading
   - Build rotation that aligns predicted forward direction with the EXIF-consistent target
4. **Applying transformation**: Rotate the entire scene so Z-axis aligns with real-world up

### Camera Coordinate Convention

The system uses the standard camera coordinate convention:
- **X-axis**: Points to the right
- **Y-axis**: Points downward  
- **Z-axis**: Points forward (viewing direction)

### Pitch Sensor Convention

The EXIF pitch follows the device orientation convention:
- **Horizontal phone** (normal usage): pitch ≈ -90°
- **Tilted down**: pitch between -90° and 0°
- **Vertical down**: pitch ≈ 0°

### Why This Matters

The inferred camera poses from the reconstruction model are **relative** - they have no absolute reference to Earth's gravity. The EXIF pitch provides that missing absolute reference, allowing the system to correctly orient the 3D reconstruction in real-world coordinates.

## Troubleshooting

**"No pitch angles found"**: 
- Verify your metadata.json is in the correct location
- Check that pitch values are in the JSON file
- Try encoding pitch in filenames as a test
- System will use camera orientation analysis as fallback

**"Scene appears rotated incorrectly"**:
- Verify pitch angle sign (positive = up, negative = down)
- Check that pitch angles are in degrees, not radians
- Ensure all images share the same coordinate convention

**"Z-axis still not aligned with gravity"**:
- Pitch only corrects for up/down tilt, not roll (side tilt)
- If cameras have significant roll, additional metadata may be needed
- The current implementation assumes minimal roll variation

## Advanced: Programmatic Usage

```python
from metadata_util import get_pitch_angles, compute_gravity_alignment_from_pitch
from visual_util import predictions_to_glb

# Get pitch angles from your metadata
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
pitch_angles = get_pitch_angles(image_paths, metadata_path="metadata.json")

# Create GLB with gravity alignment
glb_scene = predictions_to_glb(
    predictions,
    pitch_angles=pitch_angles,
    # ... other parameters
)
```

## Notes

- The gravity alignment is applied **after** the standard scene alignment
- It affects both the point cloud and camera visualizations
- Different pitch values will generate different GLB filenames (cached)
- Alignment is recomputed when you toggle the checkbox in the UI
