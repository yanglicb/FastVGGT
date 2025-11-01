# Quick Start: Gravity Alignment for Point Clouds

Your images contain pitch data in EXIF format: `Pitch:-71.351295,Roll:-10.72634,Azimuth:-154.74142`

This pitch data provides the **absolute reference** needed to align your point cloud with real-world gravity. The system uses this to calibrate the inferred camera poses from the reconstruction model.

## 🎯 Understanding Pitch Values

Your EXIF pitch convention:
- **pitch = -90°**: Camera horizontal (normal phone orientation)
- **pitch = -71°**: Camera tilted ~19° from horizontal (typical drone/aerial)
- **pitch = 0°**: Camera vertical (pointing straight down)

## 🚀 Quick Start

### Option 1: Automatic EXIF Extraction (Easiest)

The system will automatically extract pitch from your EXIF data:

1. **Place your images in a folder**
2. **Run the demo**:
   ```bash
   python demo_gradio.py
   ```
3. **Upload your images** through the web interface
4. **Check "Align with Gravity"** ✓
5. **Click "Reconstruct"**

The system will:
- Extract pitch from EXIF UserComment field
- Use it to calibrate the inferred camera poses
- Align the scene so Z-axis points upward

### Option 2: Create Metadata JSON (For Better Control)

If you want to verify or manually adjust pitch values:

1. **Generate metadata.json from your images**:
   ```bash
   python create_metadata.py --exif ./your_images --output metadata.json
   ```

2. **Review the extracted values**:
   ```json
   {
     "image001.jpg": {"pitch": -71.35},
     "image002.jpg": {"pitch": -68.20},
     "image003.jpg": {"pitch": -72.10}
   }
   ```

3. **Place metadata.json** in the same folder as your images
4. **Run the demo** and upload as usual

## 📊 Understanding the Output

The system will display information like:

```
Attempting to extract pitch angles from image EXIF/filenames
Found 17/17 pitch angles
Average EXIF pitch angle: -71.35° (17/17 images)
Gravity alignment rotation: 18.65° around X-axis
  (pitch=-71.35° → camera tilted 18.65° from horizontal after unwrapping)
```

This tells you:
- How many images have valid pitch data
- The average pitch angle from EXIF
- How much rotation is needed to align with Z-axis
- The camera's tilt angle from horizontal

## 🎯 What to Expect

With pitch ≈ -71° (typical aerial/drone imagery):

- **Before alignment**: Scene is tilted ~19° from vertical
- **After alignment**: Z-axis points straight up
- **Result**: Buildings, trees, and vertical structures appear upright

The EXIF pitch provides the **absolute reference** that tells the system which direction is "up" in the real world.

## 🔍 Validation

After reconstruction, check that:

1. ✅ Vertical structures (buildings, poles) are aligned with z-axis
2. ✅ Ground plane is roughly perpendicular to z-axis
3. ✅ The point cloud "looks right" when viewed from the side

## 🛠️ Troubleshooting

### "No pitch data found"
```bash
# Run the test script to diagnose
python test_exif_extraction.py your_image.jpg

# This will show ALL EXIF fields and help debug
```

### "Scene still looks tilted"
- Check that all images have similar pitch values (±5° is normal)
- Large variations might indicate the camera orientation changed
- Use `--validate` to check your metadata:
  ```bash
  python create_metadata.py --validate metadata.json
  ```

### "Point cloud is upside down"
- Your pitch convention might be inverted
- Try negating all pitch values in metadata.json
- Or edit the extraction function to add a negative sign

## 📁 Files Created

This implementation added several files:

- **`metadata_util.py`** - Functions for extracting and using pitch data
- **`create_metadata.py`** - Tool to generate metadata.json files
- **`test_exif_extraction.py`** - Test script to verify EXIF parsing
- **`GRAVITY_ALIGNMENT.md`** - Detailed documentation
- **`metadata_example.json`** - Example metadata file

## 🎓 Next Steps

1. **Test with a small dataset first** (3-5 images)
2. **Verify the alignment looks correct**
3. **Scale up to your full dataset**
4. **Consider caching** - Metadata JSON is faster than EXIF parsing

## 💡 Tips

- **Performance**: Using metadata.json is faster than parsing EXIF every time
- **Consistency**: Ensure all images from the same scene have similar pitch values
- **Validation**: Always validate metadata before running reconstruction
- **Caching**: Different pitch values create different cached GLB files

## ❓ Questions?

See the full documentation in `GRAVITY_ALIGNMENT.md` for:
- Technical details about the algorithm
- Advanced usage examples
- Programmatic API usage
- Coordinate system conventions
