#!/usr/bin/env python3
"""
Quick test script to verify EXIF pitch extraction from your images.
This helps verify the format is being parsed correctly.
"""

import sys
from pathlib import Path
from metadata_util import extract_pitch_from_exif
from PIL import Image
from PIL.ExifTags import TAGS


def print_all_exif(image_path):
    """Print all EXIF data from an image for debugging."""
    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if exif_data is None:
            print("No EXIF data found in this image.")
            return
        
        print("\nAll EXIF tags:")
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, f"Unknown_{tag_id}")
            
            # Handle bytes values
            if isinstance(value, bytes):
                try:
                    value_str = value.decode('utf-8', errors='ignore')[:200]
                except:
                    value_str = f"<binary data, {len(value)} bytes>"
            else:
                value_str = str(value)[:200]
            
            print(f"  {tag:30s}: {value_str}")
            
            # Highlight fields that might contain pitch
            if tag in ['UserComment', 'ImageDescription', 'Comment', 'XPComment']:
                print(f"    ⚠️  This field might contain pitch data!")
        
    except Exception as e:
        print(f"Error reading EXIF: {e}")


def test_pitch_extraction(image_path):
    """Test pitch extraction using the actual function."""
    print(f"\n{'='*60}")
    print(f"Testing pitch extraction from: {image_path}")
    print(f"{'='*60}")
    
    pitch = extract_pitch_from_exif(str(image_path))
    
    if pitch is not None:
        print(f"✓ Successfully extracted pitch: {pitch}°")
        print(f"  This looks like a {'downward' if pitch < 0 else 'upward'} facing camera")
    else:
        print("✗ Could not extract pitch from EXIF data")
        print("  Showing all EXIF tags for debugging:")
        print_all_exif(image_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_exif_extraction.py <image_file_or_directory>")
        print("\nExamples:")
        print("  python test_exif_extraction.py image.jpg")
        print("  python test_exif_extraction.py ./my_images/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # Single image
        test_pitch_extraction(path)
    elif path.is_dir():
        # Directory - process all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
            image_files.extend(path.glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No images found in {path}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images in {path}")
        print(f"Testing pitch extraction...\n")
        
        successful = 0
        failed = 0
        pitches = []
        
        for img_path in image_files:
            pitch = extract_pitch_from_exif(str(img_path))
            if pitch is not None:
                print(f"✓ {img_path.name:40s} pitch: {pitch:7.2f}°")
                successful += 1
                pitches.append(pitch)
            else:
                print(f"✗ {img_path.name:40s} no pitch found")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total images:    {len(image_files)}")
        print(f"  With pitch:      {successful}")
        print(f"  Without pitch:   {failed}")
        
        if pitches:
            import numpy as np
            print(f"\nPitch statistics:")
            print(f"  Average: {np.mean(pitches):7.2f}°")
            print(f"  Min:     {np.min(pitches):7.2f}°")
            print(f"  Max:     {np.max(pitches):7.2f}°")
            print(f"  Std dev: {np.std(pitches):7.2f}°")
        
        if failed > 0:
            print(f"\nTo see detailed EXIF data for debugging, run:")
            print(f"  python test_exif_extraction.py {image_files[0]}")
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
