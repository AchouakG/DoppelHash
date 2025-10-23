import numpy as np
from PIL import Image # Python Imaging Library
from scipy.fftpack import dct


def perceptual_hash(image_path, hash_size=32):
    """
    Generate perceptual hash (pHash) for an image using DCT (Discrete Cosine Transform).
    Returns: Binary hash string (64 bits by default)
    """
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    img = img.convert('L')
    img = img.resize((hash_size, hash_size), Image.LANCZOS)
    
    pixels = np.array(img, dtype=np.float32)
    
    dct_coeffs = dct(dct(pixels.T).T)
    
    dct_low = dct_coeffs[:8, :8]  # Keep top-left 8x8 block to reduce brightness effect
    
    median = np.median(dct_low[1:])
    
    # 1 if coefficient > median, else 0, then flattened
    hash_bits = ''.join('1' if dct_low[i, j] > median else '0' for i in range(8) for j in range(8))
    
    return hash_bits


def hamming_distance(hash1, hash2):
    """
    Calculate Hamming distance between two hashes.
    Returns the number of differing bits.
    """
    if len(hash1) != len(hash2):
        raise ValueError(f"Hash lengths don't match: {len(hash1)} vs {len(hash2)}")
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def similarity_score(hash1, hash2):
    """
    Calculate similarity percentage between two hashes.
    """
    distance = hamming_distance(hash1, hash2)
    max_distance = len(hash1)
    return ((max_distance - distance) / max_distance) * 100.0


def are_duplicates(hash1, hash2, threshold=50.0):
    """
    returns a boolean indicating if images are likely duplicates
    """
    return similarity_score(hash1, hash2) >= threshold


if __name__ == "__main__":

    image1_path = "seattle_3269390_1.jpg"
    image2_path = "seattle_3269390_2.jpg"
    
    hash1 = perceptual_hash(image1_path)
    hash2 = perceptual_hash(image2_path)
    
    print(f"\n\nHash 1: {hash1}")
    print(f"Hash 2: {hash2}")
    print(f"\nHamming Distance: {hamming_distance(hash1, hash2)}")
    print("="*50)
    print(" \t \t Results")
    print("="*50, ' \n')

    print(f"Similarity: {similarity_score(hash1, hash2):.2f}%")
    print(f"Duplicates: {are_duplicates(hash1, hash2)}")
    print("\n" + "="*50)

