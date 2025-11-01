# Coordinate Axes Feature

## Overview

A 3D coordinate system visualization has been added to the point cloud viewer. This helps users understand the orientation of the 3D reconstruction and verify gravity alignment.

## Features

### Visual Representation

The coordinate axes are displayed as colored arrows:
- **X-axis**: Red arrow pointing right
- **Y-axis**: Green arrow pointing forward
- **Z-axis**: Blue arrow pointing upward

### Automatic Scaling

The axes automatically scale based on the scene size:
- Length: 30% of the scene scale (derived from point cloud extents)
- Thickness: Proportional to axis length
- Arrow heads: 2.5x the cylinder thickness for visibility

### UI Control

In the Gradio demo interface, you'll find a checkbox:

**"Show Coordinate Axes (X=Red, Y=Green, Z=Blue)"** ✓ (checked by default)

- ✓ **Checked**: Coordinate axes are visible in the 3D view
- ☐ **Unchecked**: Coordinate axes are hidden

The visualization updates in real-time when you toggle the checkbox.

## Usage

### Basic Usage

1. **Upload your images** or video
2. **Click "Reconstruct"**
3. The 3D view will show:
   - Point cloud
   - Camera frustums (if enabled)
   - **Coordinate axes** (if enabled)

### Toggle Axes

Simply check/uncheck the **"Show Coordinate Axes"** checkbox to show/hide the axes.

### Verify Gravity Alignment

When using gravity alignment with pitch metadata:

1. **Enable both**: 
   - ✓ Align with Gravity
   - ✓ Show Coordinate Axes

2. **Check the result**:
   - The **Z-axis (blue)** should point upward
   - Vertical structures should align with the blue axis
   - The ground plane should be perpendicular to the blue axis

## Coordinate System Convention

After scene alignment and gravity correction:

```
       Z (Blue)
       ↑
       |
       |
       +----→ X (Red)
      /
     /
    Y (Green)
```

- **Origin**: Located at the scene center
- **X-axis (Red)**: Points to the right
- **Y-axis (Green)**: Points forward (in the viewing direction)
- **Z-axis (Blue)**: Points upward (aligned with gravity when enabled)

## Technical Details

### Implementation

The axes are created using `trimesh` primitives:
- Cylinders for the axis shafts
- Cones for the arrow heads
- Vertex colors for RGB coding

### Transformation

The coordinate axes are:
1. Created at the origin
2. Transformed along with the entire scene
3. Subject to the same alignment as the point cloud
4. Affected by gravity alignment when enabled

### Performance

- **Minimal overhead**: Only 6 simple geometric objects (3 cylinders + 3 cones)
- **Cached**: Different axes settings generate different GLB files
- **Real-time toggle**: Switching between cached files is instant

## Examples

### With Axes Enabled (Default)
- Helps identify which direction is "up" in your reconstruction
- Useful for verifying orientation
- Essential when working with gravity-aligned data

### With Axes Disabled
- Cleaner visualization
- Better for presentations or screenshots
- Reduces visual clutter for very dense point clouds

## Troubleshooting

### "Axes are too small/large"
- The axes auto-scale to 30% of the scene size
- If your scene has very dispersed points, the axes will be larger
- Consider filtering outliers with the confidence threshold

### "Axes don't align with my expected directions"
- Check if gravity alignment is enabled
- Verify your pitch metadata is correct
- The axes follow OpenGL/OpenCV conventions

### "Axes appear in wrong orientation"
- This is usually correct - different software uses different conventions
- The Z-axis pointing up is the standard after gravity alignment
- Without gravity alignment, axes are relative to the first camera

## Related Features

- **Gravity Alignment**: See `GRAVITY_ALIGNMENT.md` for details on aligning Z-axis with real-world up
- **Camera Visualization**: Shows camera positions and orientations
- **Confidence Filtering**: Removes low-confidence points to clean up the scene

## Code Reference

### Visual Utilities (`visual_util.py`)

- `create_coordinate_axes(scene_scale)`: Creates the axes geometry
- `predictions_to_glb(..., show_axes=True)`: Main conversion function with axes parameter

### Demo Interface (`demo_gradio.py`)

- UI checkbox: `show_axes = gr.Checkbox(...)`
- Parameter passing to all reconstruction and visualization functions
- Real-time updates when checkbox state changes

## Tips

1. **Always enable for initial reconstruction** - Helps verify the orientation is correct
2. **Disable for final exports** - If you want clean visualizations without reference axes
3. **Use with gravity alignment** - The combination helps verify your pitch data is working
4. **Compare with/without** - Toggle to understand how your scene is oriented

## Future Enhancements

Potential improvements for future versions:
- Adjustable axis length
- Optional axis labels (X, Y, Z text)
- Grid plane at Z=0
- Scale ruler/measurements
- Multiple coordinate systems (world vs. camera)
