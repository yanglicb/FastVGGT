#!/usr/bin/env python3
"""
Script to help users create metadata.json files for their image datasets.
This can extract EXIF data or help create templates for manual entry.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from metadata_util import extract_pitch_from_exif, extract_pitch_from_filename


def create_metadata_from_exif(image_dir: str, output_path: str) -> Dict[str, float]:
    """
    Scan directory for images and extract pitch from EXIF data.
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save metadata.json
        
    Returns:
        Dictionary mapping image names to pitch angles
    """
    image_dir = Path(image_dir)
    metadata = {}
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    print(f"Scanning {len(image_files)} images for pitch metadata...")
    
    found_count = 0
    for img_path in image_files:
        pitch = extract_pitch_from_exif(str(img_path))
        if pitch is not None:
            metadata[img_path.name] = {"pitch": pitch}
            found_count += 1
            print(f"  ✓ {img_path.name}: {pitch}°")
        else:
            # Try filename as fallback
            pitch_from_name = extract_pitch_from_filename(str(img_path))
            if pitch_from_name is not None:
                metadata[img_path.name] = {"pitch": pitch_from_name}
                found_count += 1
                print(f"  ✓ {img_path.name}: {pitch_from_name}° (from filename)")
    
    print(f"\nFound pitch data for {found_count}/{len(image_files)} images")
    
    if metadata:
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {output_path}")
    else:
        print("No pitch data found in images.")
    
    return metadata


def create_metadata_template(image_dir: str, output_path: str, default_pitch: Optional[float] = None):
    """
    Create a template metadata.json file for manual editing.
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save metadata.json template
        default_pitch: Default pitch value to use (None for placeholder)
    """
    image_dir = Path(image_dir)
    metadata = {}
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    print(f"Creating template for {len(image_files)} images...")
    
    pitch_value = default_pitch if default_pitch is not None else 0.0
    note = "REPLACE WITH ACTUAL PITCH" if default_pitch is None else f"Default: {default_pitch}°"
    
    for img_path in image_files:
        metadata[img_path.name] = {
            "pitch": pitch_value,
            "note": note
        }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Template saved to: {output_path}")
    print(f"Please edit the file and replace pitch values with actual measurements.")


def validate_metadata(metadata_path: str):
    """
    Validate a metadata.json file.
    
    Args:
        metadata_path: Path to metadata.json file
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Validating metadata from: {metadata_path}")
        print(f"Total entries: {len(metadata)}")
        
        valid_count = 0
        issues = []
        
        for img_name, data in metadata.items():
            if isinstance(data, dict):
                if 'pitch' not in data:
                    issues.append(f"  ⚠ {img_name}: Missing 'pitch' field")
                    continue
                pitch = data['pitch']
            else:
                pitch = data
            
            try:
                pitch_float = float(pitch)
                if -90 <= pitch_float <= 90:
                    valid_count += 1
                else:
                    issues.append(f"  ⚠ {img_name}: Pitch {pitch_float}° out of range [-90, 90]")
            except (ValueError, TypeError):
                issues.append(f"  ✗ {img_name}: Invalid pitch value: {pitch}")
        
        print(f"Valid entries: {valid_count}/{len(metadata)}")
        
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(issue)
        else:
            print("✓ All entries valid!")
        
        # Statistics
        if valid_count > 0:
            pitches = []
            for img_name, data in metadata.items():
                try:
                    if isinstance(data, dict):
                        pitches.append(float(data['pitch']))
                    else:
                        pitches.append(float(data))
                except:
                    pass
            
            if pitches:
                import numpy as np
                print(f"\nStatistics:")
                print(f"  Mean pitch: {np.mean(pitches):.2f}°")
                print(f"  Min pitch:  {np.min(pitches):.2f}°")
                print(f"  Max pitch:  {np.max(pitches):.2f}°")
                print(f"  Std dev:    {np.std(pitches):.2f}°")
        
        return valid_count == len(metadata)
        
    except FileNotFoundError:
        print(f"Error: File not found: {metadata_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create or validate metadata.json files for gravity alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract pitch from EXIF data
  python create_metadata.py --exif ./images --output metadata.json
  
  # Create template for manual entry
  python create_metadata.py --template ./images --output metadata.json
  
  # Create template with default pitch value
  python create_metadata.py --template ./images --output metadata.json --default-pitch -15.0
  
  # Validate existing metadata
  python create_metadata.py --validate metadata.json
        """
    )
    
    parser.add_argument('--exif', type=str, help='Extract pitch from EXIF data in this directory')
    parser.add_argument('--template', type=str, help='Create template for images in this directory')
    parser.add_argument('--validate', type=str, help='Validate existing metadata.json file')
    parser.add_argument('--output', type=str, default='metadata.json', help='Output file path (default: metadata.json)')
    parser.add_argument('--default-pitch', type=float, help='Default pitch value for template (degrees)')
    
    args = parser.parse_args()
    
    if args.exif:
        create_metadata_from_exif(args.exif, args.output)
    elif args.template:
        create_metadata_template(args.template, args.output, args.default_pitch)
    elif args.validate:
        validate_metadata(args.validate)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
