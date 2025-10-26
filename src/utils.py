from collections import defaultdict


class UnionFind:
    """ Union-Find data structure for grouping duplicates."""
    
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}
    
    def find(self, element):
        """
        Find root of element with path compression.
        Path compression flattens the tree for faster future lookups.
        """
        if self.parent[element] != element:
            # Recursively find root and compress path
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]
    
    def union(self, element1, element2):
        """
        Union two sets by rank
        """
        root1 = self.find(element1)
        root2 = self.find(element2)
        
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                # equal rank: attach root2 under root1 and increment rank
                self.parent[root2] = root1
                self.rank[root1] += 1
    
    def get_groups(self):
        """ Get groups of connected elements."""
        groups = {}
        for element in self.parent:
            root = self.find(element)
            if root not in groups:
                groups[root] = []
            groups[root].append(element)
        
        return [group for group in groups.values() if len(group) > 1]


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

class LSH:
    """
    Locality-Sensitive Hashing for fast candidate generation.
    Reduces comparison complexity from O(n²) to approximately O(n).
    
    How it works:
    1. Split each hash into multiple bands
    2. Hash each band to a bucket ID
    3. Images landing in the same bucket are candidates for comparison
    """
    
    def __init__(self, num_bands=8, rows_per_band=8):
        """
        Initialize LSH index.
        
        Args:
            num_bands: Number of bands to split hash into (default 8)
            rows_per_band: Number of bits per band (default 8)
            
        Note:
            - Total hash size must be num_bands * rows_per_band
            - More bands = more sensitive (more false positives)
            - More rows per band = more strict
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = defaultdict(set)
        
        self.hash_size = num_bands * rows_per_band
    
    def _hash_band(self, image_hash, band_idx):
        """
        Extract and hash a specific band from the image hash.
        
        Args:
            image_hash: Full hash string (e.g., 64-bit binary string)
            band_idx: Which band to extract (0 to num_bands-1)
            
        Returns:
            Bucket ID for this band (string)
        """
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band = image_hash[start:end]
        
        # unique bucket ID: "0_10101101", "1_11001100"...
        return f"{band_idx}_{band}"
    
    def index(self, image_name, image_hash):
        """
        Add an image to the LSH index.
        
        Args:
            image_name: Filename or identifier
            image_hash: Binary hash string (must be num_bands × rows_per_band bits)
        """
        if len(image_hash) != self.hash_size:
            raise ValueError(
                f"Hash size mismatch: expected {self.hash_size} bits, "
                f"got {len(image_hash)} bits"
            )
        
        for band_idx in range(self.num_bands):
            bucket_id = self._hash_band(image_hash, band_idx)
            self.buckets[bucket_id].add(image_name)
    
    def get_candidates(self, image_name, image_hash):
        """
        Get candidate duplicates for an image.
        Only images in the same bucket(s) are returned as candidates.
        
        Args:
            image_name: Image to find candidates for
            image_hash: Hash of the image
            
        Returns:
            Set of candidate image names (excluding the query image itself)
        """
        if len(image_hash) != self.hash_size:
            raise ValueError(
                f"Hash size mismatch: expected {self.hash_size} bits, "
                f"got {len(image_hash)} bits"
            )
        
        candidates = set()
        
        # Collect candidates from all band buckets
        for band_idx in range(self.num_bands):
            bucket_id = self._hash_band(image_hash, band_idx)
            candidates.update(self.buckets[bucket_id])
        
        # Remove the query image itself
        candidates.discard(image_name)
        
        return candidates
    
    def get_stats(self): #! should delete later
        """
        Get statistics about the LSH index.
        
        Returns:
            Dictionary with index statistics
        """
        bucket_sizes = [len(images) for images in self.buckets.values()]
        
        return {
            'total_buckets': len(self.buckets),
            'total_bands': self.num_bands,
            'rows_per_band': self.rows_per_band,
            'hash_size': self.hash_size,
            'avg_bucket_size': sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
            'min_bucket_size': min(bucket_sizes) if bucket_sizes else 0
        }