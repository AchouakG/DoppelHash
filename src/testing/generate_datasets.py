from PIL import Image, ImageEnhance
import os
from pathlib import Path
import shutil

def create_test_dataset(source_folder, output_folder, num_originals=4):
    """Create a test dataset with known duplicates from source folder."""
    
    output_path = Path(output_folder)
    
    # Clear existing output folder if it exists
    if output_path.exists():
        print(f"\n>>> Clearing existing folder: {output_folder}")
        shutil.rmtree(output_path)
        print("✓ Folder cleared")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_folder)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    source_images = [f for f in source_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if len(source_images) < num_originals:
        print(f"Warning: Only {len(source_images)} images found, using all of them")
        num_originals = len(source_images)
    
    # first n images used as originals
    originals = source_images[:num_originals]
    print("\n" +"\n" + "="*60)
    print(f"Creating test dataset with {num_originals} original images...")
    print("="*60)
    
    duplicate_map = {}
    
    for idx, original_path in enumerate(originals, 1):
        img = Image.open(original_path)
        base_name = f"image_{idx:02d}"
        
        duplicates_created = []
        
        # Save original
        original_name = f"{base_name}_original{original_path.suffix}"
        img.save(output_path / original_name, quality=95)
        duplicates_created.append(original_name)
        
        # Exact copy
        copy_name = f"{base_name}_exact_copy{original_path.suffix}"
        img.save(output_path / copy_name, quality=95)
        duplicates_created.append(copy_name)
        
        # resized
        img_resized = img.resize((int(img.width * 0.8), int(img.height * 0.8)), Image.LANCZOS)
        resized_name = f"{base_name}_resized{original_path.suffix}"
        img_resized.save(output_path / resized_name, quality=95)
        duplicates_created.append(resized_name)
        
        # compressed
        compressed_name = f"{base_name}_compressed{original_path.suffix}"
        img.save(output_path / compressed_name, quality=60)
        duplicates_created.append(compressed_name)
        
        # slightly cropped
        crop_margin = 20
        if img.width > crop_margin*2 and img.height > crop_margin*2:
            img_cropped = img.crop((crop_margin, crop_margin, img.width-crop_margin, img.height-crop_margin))
            cropped_name = f"{base_name}_cropped{original_path.suffix}"
            img_cropped.save(output_path / cropped_name, quality=95)
            duplicates_created.append(cropped_name)
        
        # brightness difference
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.2)
        bright_name = f"{base_name}_brighter{original_path.suffix}"
        img_bright.save(output_path / bright_name, quality=95)
        duplicates_created.append(bright_name)
        
        duplicate_map[base_name] = duplicates_created
        print(f"✓ Created {len(duplicates_created)} versions of {base_name}")
    
    # Different images
    print("\nAdding different images...")
    non_duplicates = source_images[num_originals:num_originals+3]
    for idx, img_path in enumerate(non_duplicates, 1):
        different_name = f"different_{idx:02d}{img_path.suffix}"
        shutil.copy(img_path, output_path / different_name)
        print(f"✓ Added {different_name}")
    
    total_images = sum(len(dups) for dups in duplicate_map.values()) + len(non_duplicates)
    print(f"\n{'='*60}")
    print(f"Total images created: {total_images}")
    print(f"Different images: {len(non_duplicates)}")
    print(f"{'='*60}\n")
    
    return duplicate_map


if __name__ == "__main__":
    # Configuration
    SOURCE_FOLDER = "../../datasets/Airbnb_data/Test_data/outdoor"
    OUTPUT_FOLDER = "../../datasets/Airbnb_data/Test_data/myData"
    NUM_ORIGINALS = 3  # How many unique images to use
    
    duplicate_map = create_test_dataset(SOURCE_FOLDER, OUTPUT_FOLDER, NUM_ORIGINALS)