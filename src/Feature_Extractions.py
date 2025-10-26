import numpy as np
from PIL import Image
from scipy.fftpack import dct
from pathlib import Path
import sys
import time
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))# ! streamlit nested files error

from src.utils import UnionFind, similarity_score, hamming_distance, LSH

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


def find_duplicates(folder_path, algorithm, threshold, sim_method='Bruteforce', num_bands=8, rows_per_band=8):
    """
    Find duplicate images  """
    
    hash_functions = {
        'phash': perceptual_hash,
    }
    
    hash_func = hash_functions[algorithm]
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return [], 0, {}
    
    print(f"Hashing {len(image_files)} images...")
    start_time = time.time()
    
    hashes = {}
    for img_path in image_files:
        img_hash = hash_func(str(img_path))
        if img_hash is not None:
            hashes[img_path.name] = img_hash
    
    hash_time = time.time() - start_time
    print(f"Hashing completed in {hash_time:.2f}s")
    
    image_names = list(hashes.keys())
    unionf = UnionFind(image_names)
    
    comparison_count = 0
    similarity_matrix = {}
    
    start_time = time.time()
    
    if sim_method == 'Bruteforce':
        
        for i, img1 in enumerate(image_names):
            
            for img2 in image_names[i+1:]:
                comparison_count += 1
                distance = hamming_distance(hashes[img1], hashes[img2])
                similarity = similarity_score(hashes[img1], hashes[img2])
                
                similarity_matrix[(img1, img2)] = {
                    'hamming': distance,
                    'similarity': similarity
                }
                
                if similarity >= threshold:
                    unionf.union(img1, img2)
    
    elif sim_method == 'lsh':
        
        lsh = LSH(num_bands=num_bands, rows_per_band=rows_per_band)
        
        for img_name, img_hash in hashes.items():
            lsh.index(img_name, img_hash)
        
        compared_pairs = set()
        
        for i, img1 in enumerate(image_names):
            candidates = lsh.get_candidates(img1, hashes[img1])
            
            for img2 in candidates:
                pair = tuple(sorted([img1, img2]))
                if pair in compared_pairs:
                    continue
                compared_pairs.add(pair)
                
                comparison_count += 1
                distance = hamming_distance(hashes[img1], hashes[img2])
                similarity = similarity_score(hashes[img1], hashes[img2])
                
                similarity_matrix[(img1, img2)] = {
                    'hamming': distance,
                    'similarity': similarity
                }
                
                if similarity >= threshold:
                    unionf.union(img1, img2)
        
    
    else:
        raise ValueError(f"Invalid sim_method: {sim_method}. Use 'bruteforce' or 'lsh'")
    
    comparison_time = time.time() - start_time

    
    duplicate_groups = unionf.get_groups()
    
    group_scores = []
    for group in duplicate_groups:
        n = len(group)
        if n < 2:
            continue
        
        pairwise_scores = []
        for i in range(n):
            for j in range(i+1, n):
                img1, img2 = group[i], group[j]
                score = (similarity_matrix.get((img1, img2)) or
                        similarity_matrix.get((img2, img1)))
                if score:
                    pairwise_scores.append(score)
        
        if pairwise_scores:
            avg_similarity = np.mean([s['similarity'] for s in pairwise_scores])
            group_scores.append({
                'group': group,
                'avg_similarity': round(avg_similarity, 2)
            })
    
    max_possible_comparisons = len(image_names) * (len(image_names) - 1) // 2
    reduction_pct = 100 * (1 - comparison_count / max_possible_comparisons) if max_possible_comparisons > 0 else 0
    
    stats = {
        'method': sim_method,
        'total_images': len(image_names),
        'hash_time': round(hash_time, 2),
        'comparison_time': round(comparison_time, 2),
        'total_time': round(hash_time + comparison_time, 2),
        'comparisons_made': comparison_count,
        'max_possible_comparisons': max_possible_comparisons,
        'comparison_reduction': round(reduction_pct, 1),
        'duplicate_groups_found': len(group_scores)
    }
    
    return group_scores, len(group_scores), stats
