import numpy as np
from PIL import Image
from scipy.fftpack import dct
import os
from pathlib import Path

def perceptual_hash(image_path, hash_size=32):
    """
    Generate perceptual hash for an image using DCT.
    Returns: Binary hash string 64 bits
    """
    try:
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
        
        img = img.convert('L')
        img = img.resize((hash_size, hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
        dct_coeffs = dct(dct(pixels.T).T)
        dct_low = dct_coeffs[:8, :8]
        median = np.median(dct_low[1:])
        hash_bits = ''.join('1' if dct_low[i, j] > median else '0' for i in range(8) for j in range(8))
        return hash_bits
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes.
    Returns the number of differing bits."""
    
    if hash1 is None or hash2 is None:
        return float('inf')
    if len(hash1) != len(hash2):
        raise ValueError(f"Hash lengths don't match: {len(hash1)} vs {len(hash2)}")
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def similarity_score(hash1, hash2):
    """Calculate similarity percentage between two hashes."""
    
    if hash1 is None or hash2 is None:
        return 0.0
    distance = hamming_distance(hash1, hash2)
    if distance == float('inf'):
        return 0.0
    max_distance = len(hash1)
    return ((max_distance - distance) / max_distance) * 100.0


class UnionFind:
    """Union-Find data structure for grouping duplicates."""
    
    def __init__(self, elements):
        self.parent = {element: element for element in elements} # each elements points to itself
        self.rank = {element: 0 for element in elements} # each is a root
    
    def find(self, element):
        """Find root of element with _path compression_"""
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]
    
    def union(self, element1, element2):
        """Union two sets by rank."""
        root1 = self.find(element1)
        root2 = self.find(element2)
        
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
    
    def get_groups(self):
        """Get all groups of connected elements."""
        groups = {}
        for element in self.parent:
            root = self.find(element)
            if root not in groups:
                groups[root] = []
            groups[root].append(element)
        return [group for group in groups.values() if len(group) > 1]


def find_duplicates(folder_path, algorithm, threshold):
    """ Find duplicate images in a folder using Union-Find"""
    hash_functions = {
        'phash': perceptual_hash,
    }
    
    hash_func = hash_functions[algorithm]
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Get all image files
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return []
    
    hashes = {}
    for img_path in image_files:
        img_hash = hash_func(str(img_path))
        if img_hash is not None:
            hashes[img_path.name] = img_hash
        
    image_names = list(hashes.keys())
    unionf = UnionFind(image_names)
    
    comparison_count = 0
    for i, img1 in enumerate(image_names):
        for img2 in image_names[i+1:]:
            comparison_count += 1
            similarity = similarity_score(hashes[img1], hashes[img2])
            
            if similarity >= threshold:
                unionf.union(img1, img2)
    
    print(f"Made {comparison_count} comparisons")
    
    duplicate_groups = unionf.get_groups()
    
    return duplicate_groups


def print_duplicates(duplicate_groups):
    """Print duplicates."""
    if not duplicate_groups:
        print("No duplicates found!")
        return
    
    print("\nDuplicates:")
    formatted_groups = ", ".join([str(group) for group in duplicate_groups])
    print(formatted_groups)
