"""
Distributed Cache Simulator
---------------------------
A simulation framework for evaluating caching algorithms under distributed systems conditions
with support for network latency, disk I/O, concurrency, and adaptive ML-based caching.
"""

import numpy as np
import pandas as pd
import time
import random
import threading
import queue
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, Counter
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

class CacheEntry:
    """Represents a single cache entry with metadata for algorithms."""
    def __init__(self, key: str, value: Any, size: int = 1):
        self.key = key
        self.value = value
        self.size = size  # Size in arbitrary units
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.access_history = [(time.time(), 1)]  # (timestamp, count)

    def access(self):
        """Record an access to this cache entry."""
        now = time.time()
        self.last_accessed = now
        self.access_count += 1
        self.access_history.append((now, self.access_count))
        
    def get_recency(self) -> float:
        """Get recency score (lower is more recent)."""
        return time.time() - self.last_accessed
    
    def get_frequency(self) -> int:
        """Get frequency score."""
        return self.access_count
    
    def get_size(self) -> int:
        """Get entry size."""
        return self.size

class NetworkSimulator:
    """Simulates network conditions in a distributed system."""
    def __init__(self, 
                 base_latency_ms: float = 5.0,
                 jitter_ms: float = 2.0, 
                 packet_loss_percent: float = 0.1,
                 partition_probability: float = 0.001):
        self.base_latency = base_latency_ms / 1000  # Convert to seconds
        self.jitter = jitter_ms / 1000
        self.packet_loss_percent = packet_loss_percent
        self.partition_probability = partition_probability
        self.partitioned = False
        self.partition_check_interval = 10  # seconds
        self.last_partition_check = time.time()
        
    def simulate_network_delay(self) -> float:
        """Simulate network delay with jitter."""
        # Check for network partition
        if time.time() - self.last_partition_check > self.partition_check_interval:
            self.partitioned = random.random() < self.partition_probability
            self.last_partition_check = time.time()
            
        if self.partitioned:
            # Simulate a network partition by returning a very large delay
            return 30.0  # 30 seconds timeout
            
        # Simulate packet loss
        if random.random() < (self.packet_loss_percent / 100):
            # Retry delay
            return random.uniform(0.5, 2.0)
            
        # Normal delay with jitter
        return max(0, self.base_latency + random.uniform(-self.jitter, self.jitter))

class DiskIOSimulator:
    """Simulates disk I/O latency."""
    def __init__(self, 
                 read_latency_ms: float = 8.0, 
                 write_latency_ms: float = 15.0,
                 read_variance: float = 0.3, 
                 write_variance: float = 0.5,
                 disk_failure_rate: float = 0.0001):
        self.read_latency = read_latency_ms / 1000
        self.write_latency = write_latency_ms / 1000
        self.read_variance = read_variance
        self.write_variance = write_variance
        self.disk_failure_rate = disk_failure_rate
        
    def simulate_read_delay(self, size: int = 1) -> float:
        """Simulate read delay based on size and with variance."""
        if random.random() < self.disk_failure_rate:
            # Simulate disk failure (very slow read)
            return random.uniform(1.0, 5.0)
            
        base_delay = self.read_latency * size
        variance_factor = random.uniform(1 - self.read_variance, 1 + self.read_variance)
        return base_delay * variance_factor
        
    def simulate_write_delay(self, size: int = 1) -> float:
        """Simulate write delay based on size and with variance."""
        if random.random() < self.disk_failure_rate:
            # Simulate disk failure (very slow write or failure)
            return random.uniform(2.0, 10.0)
            
        base_delay = self.write_latency * size
        variance_factor = random.uniform(1 - self.write_variance, 1 + self.write_variance)
        return base_delay * variance_factor

class BaseCache:
    """Base cache implementation with common functionality."""
    def __init__(self, 
                 capacity: int = 100,
                 name: str = "BaseCache",
                 network_simulator: Optional[NetworkSimulator] = None,
                 disk_simulator: Optional[DiskIOSimulator] = None):
        self.capacity = capacity
        self.name = name
        self.cache: Dict[str, CacheEntry] = {}
        self.size_used = 0
        self.hits = 0
        self.misses = 0
        self.network = network_simulator or NetworkSimulator()
        self.disk = disk_simulator or DiskIOSimulator()
        self.stats = {
            "get_times": [],
            "put_times": [],
            "hit_ratio_history": [],
            "latency_history": []
        }
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
    def get(self, key: str) -> Tuple[Any, float]:
        """Get an item from the cache, returning the value and total latency."""
        start_time = time.time()
        
        with self.lock:
            network_delay = self.network.simulate_network_delay()
            time.sleep(network_delay)  # Simulate network delay
            
            if key in self.cache:
                # Cache hit
                entry = self.cache[key]
                entry.access()
                self.hits += 1
                
                # Simulate read latency
                disk_delay = self.disk.simulate_read_delay(entry.size)
                time.sleep(disk_delay)  # Simulate disk read delay
                
                # Update cache based on the access (implementation-specific)
                self._update_on_hit(key, entry)
                
                latency = time.time() - start_time
                self.stats["get_times"].append(latency)
                self.stats["hit_ratio_history"].append(self._get_hit_ratio())
                self.stats["latency_history"].append(latency)
                
                return entry.value, latency
            else:
                # Cache miss
                self.misses += 1
                latency = time.time() - start_time
                self.stats["get_times"].append(latency)
                self.stats["hit_ratio_history"].append(self._get_hit_ratio())
                self.stats["latency_history"].append(latency)
                
                return None, latency

    def put(self, key: str, value: Any, size: int = 1) -> float:
        """Add an item to the cache, returning the total latency."""
        start_time = time.time()
        
        with self.lock:
            network_delay = self.network.simulate_network_delay()
            time.sleep(network_delay)  # Simulate network delay
            
            # Simulate write latency 
            disk_delay = self.disk.simulate_write_delay(size)
            time.sleep(disk_delay)  # Simulate disk write delay
            
            # Check if key already exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.size_used -= old_entry.size
                
            # Create new entry
            entry = CacheEntry(key, value, size)
            
            # Make space if needed
            while self.size_used + size > self.capacity and self.cache:
                self._evict()
                
            # Add to cache
            self.cache[key] = entry
            self.size_used += size
            
            # Implementation-specific update
            self._update_on_put(key, entry)
            
            latency = time.time() - start_time
            self.stats["put_times"].append(latency)
            
            return latency
            
    def _evict(self) -> None:
        """Evict an item from the cache (implementation-specific)."""
        raise NotImplementedError("Subclasses must implement _evict")
        
    def _update_on_hit(self, key: str, entry: CacheEntry) -> None:
        """Update cache state on hit (implementation-specific)."""
        pass  # Optional for subclasses to implement
        
    def _update_on_put(self, key: str, entry: CacheEntry) -> None:
        """Update cache state on put (implementation-specific)."""
        pass  # Optional for subclasses to implement
        
    def _get_hit_ratio(self) -> float:
        """Calculate the current hit ratio."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            avg_get_time = np.mean(self.stats["get_times"]) if self.stats["get_times"] else 0
            avg_put_time = np.mean(self.stats["put_times"]) if self.stats["put_times"] else 0
            
            return {
                "name": self.name,
                "capacity": self.capacity,
                "size_used": self.size_used,
                "item_count": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self._get_hit_ratio(),
                "avg_get_latency": avg_get_time,
                "avg_put_latency": avg_put_time
            }

class LRUCache(BaseCache):
    """Least Recently Used (LRU) Cache Implementation."""
    def __init__(self, **kwargs):
        super().__init__(name="LRU Cache", **kwargs)
        self.access_order = OrderedDict()  # Track access order
        
    def _update_on_hit(self, key: str, entry: CacheEntry) -> None:
        # Move accessed item to the end (most recently used position)
        self.access_order.pop(key, None)
        self.access_order[key] = None
        
    def _update_on_put(self, key: str, entry: CacheEntry) -> None:
        # Add new item at the end (most recently used position)
        self.access_order[key] = None
        
    def _evict(self) -> None:
        # Remove least recently used item (from the beginning)
        if self.access_order:
            lru_key, _ = self.access_order.popitem(last=False)
            if lru_key in self.cache:
                evicted = self.cache.pop(lru_key)
                self.size_used -= evicted.size

class LFUCache(BaseCache):
    """Least Frequently Used (LFU) Cache Implementation."""
    def __init__(self, **kwargs):
        super().__init__(name="LFU Cache", **kwargs)
        # Frequency -> set of keys with that frequency
        self.frequency_map: Dict[int, set] = defaultdict(set)
        # Key -> frequency
        self.key_frequency: Dict[str, int] = {}
        self.min_frequency = 1
        
    def _update_on_hit(self, key: str, entry: CacheEntry) -> None:
        # Update frequency information
        freq = self.key_frequency[key]
        self.frequency_map[freq].remove(key)
        
        # Remove the empty frequency set
        if not self.frequency_map[freq]:
            del self.frequency_map[freq]
            
        # If we removed the min frequency and it's now empty, increment min_frequency
        if freq == self.min_frequency and freq not in self.frequency_map:
            self.min_frequency += 1
        
        # Increment the frequency and update mappings
        self.key_frequency[key] = freq + 1
        self.frequency_map[freq + 1].add(key)
        
    def _update_on_put(self, key: str, entry: CacheEntry) -> None:
        # Set initial frequency
        self.key_frequency[key] = 1
        self.frequency_map[1].add(key)
        self.min_frequency = 1  # New item has the lowest frequency
        
    def _evict(self) -> None:
        # Evict an item with the lowest frequency
        if not self.frequency_map:
            return
            
        # Get keys with minimum frequency
        min_freq_keys = self.frequency_map[self.min_frequency]
        if not min_freq_keys:
            # Find the new minimum frequency
            self.min_frequency = min(self.frequency_map.keys())
            min_freq_keys = self.frequency_map[self.min_frequency]
            
        # Choose a key to evict (arbitrarily take the first one)
        key_to_evict = next(iter(min_freq_keys))
        
        # Remove from frequency tracking
        min_freq_keys.remove(key_to_evict)
        if not min_freq_keys:
            del self.frequency_map[self.min_frequency]
            if self.frequency_map:
                self.min_frequency = min(self.frequency_map.keys())
        
        # Remove from key->frequency mapping
        del self.key_frequency[key_to_evict]
        
        # Remove from cache
        if key_to_evict in self.cache:
            evicted = self.cache.pop(key_to_evict)
            self.size_used -= evicted.size

class ARCache(BaseCache):
    """Adaptive Replacement Cache (ARC) Implementation."""
    def __init__(self, **kwargs):
        super().__init__(name="ARC Cache", **kwargs)
        # T1: Recently used items that are in the cache (LRU)
        self.t1 = OrderedDict()
        # T2: Frequently used items that are in the cache (LFU-like)
        self.t2 = OrderedDict()
        # B1: Ghost list for recently evicted items from T1
        self.b1 = OrderedDict()
        # B2: Ghost list for recently evicted items from T2
        self.b2 = OrderedDict()
        # p: Target size for T1 (adaptive parameter)
        self.p = 0
        # c: Total cache size (T1 + T2)
        self.c = kwargs.get('capacity', 100)
        
    def _update_on_hit(self, key: str, entry: CacheEntry) -> None:
        # Case 1: Item is in T1 (recent tier)
        if key in self.t1:
            # Move from T1 to T2 (recent -> frequent)
            self.t1.pop(key)
            self.t2[key] = None
        # Case 2: Item is in T2 (frequent tier)
        elif key in self.t2:
            # Move to MRU position in T2
            self.t2.pop(key)
            self.t2[key] = None
            
    def _update_on_put(self, key: str, entry: CacheEntry) -> None:
        # Case 1: Item is in B1 (recently evicted from T1)
        if key in self.b1:
            # Adapt p: increase T1's target size
            self.p = min(self.c, self.p + max(1, len(self.b2) // len(self.b1)))
            self._replace(key)
            self.b1.pop(key)
            self.t2[key] = None  # Add to T2 (frequent tier)
        # Case 2: Item is in B2 (recently evicted from T2)
        elif key in self.b2:
            # Adapt p: decrease T1's target size
            self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
            self._replace(key)
            self.b2.pop(key)
            self.t2[key] = None  # Add to T2 (frequent tier)
        # Case 3: New item
        else:
            # Case 3a: L1 (T1 + B1) has c items
            l1_size = len(self.t1) + len(self.b1)
            if l1_size == self.c:
                if len(self.t1) < self.c:
                    # B1 is not empty, remove LRU from B1
                    if self.b1:
                        self.b1.popitem(last=False)
                    self._replace(key)
                else:
                    # T1 is full, remove LRU from T1
                    lru_key, _ = self.t1.popitem(last=False)
                    if lru_key in self.cache:
                        evicted = self.cache.pop(lru_key)
                        self.size_used -= evicted.size
            # Case 3b: L1 + L2 has 2c items
            elif l1_size < self.c and (l1_size + len(self.t2) + len(self.b2)) >= self.c:
                l2_size = len(self.t2) + len(self.b2)
                if l1_size + l2_size == 2 * self.c:
                    # B2 is not empty, remove LRU from B2
                    if self.b2:
                        self.b2.popitem(last=False)
                    self._replace(key)
            
            # Add new item to T1 (recent tier)
            self.t1[key] = None
            
    def _replace(self, key: str) -> None:
        """Replace an item using the ARC policy."""
        if self.t1 and ((key in self.b2 and len(self.t1) > self.p) or len(self.t1) >= self.p):
            # Replace from T1
            lru_key, _ = self.t1.popitem(last=False)
            # Move to B1 (ghost list)
            self.b1[lru_key] = None
            if lru_key in self.cache:
                evicted = self.cache.pop(lru_key)
                self.size_used -= evicted.size
        elif self.t2:
            # Replace from T2
            lru_key, _ = self.t2.popitem(last=False)
            # Move to B2 (ghost list)
            self.b2[lru_key] = None
            if lru_key in self.cache:
                evicted = self.cache.pop(lru_key)
                self.size_used -= evicted.size
        
    def _evict(self) -> None:
        """Evict an item based on the ARC policy."""
        self._replace(None)

class TwoQCache(BaseCache):
    """2Q Cache Implementation (A1in, A1out, Am)."""
    def __init__(self, **kwargs):
        super().__init__(name="2Q Cache", **kwargs)
        self.kin = kwargs.get('kin', 0.25)  # Portion for A1in queue
        self.kout = kwargs.get('kout', 0.5)  # Portion for A1out queue
        
        # Calculate queue sizes
        c = kwargs.get('capacity', 100)
        self.a1in_size = int(c * self.kin)
        self.a1out_size = int(c * self.kout)
        self.am_size = c - self.a1in_size
        
        # A1in: Short-term queue (FIFO)
        self.a1in = OrderedDict()
        # A1out: Ghost list for A1in
        self.a1out = OrderedDict()
        # Am: Long-term queue (LRU)
        self.am = OrderedDict()
        
    def _update_on_hit(self, key: str, entry: CacheEntry) -> None:
        # Case 1: Item is in A1in
        if key in self.a1in:
            # Stay in A1in (note: original 2Q would move to Am here,
            # but variants keep it in A1in for the first hit)
            pass
        # Case 2: Item is in Am
        elif key in self.am:
            # Move to MRU position in Am
            self.am.pop(key)
            self.am[key] = None
        # Case 3: Item is in A1out (ghost list) - handled in get/put
        
    def _update_on_put(self, key: str, entry: CacheEntry) -> None:
        # Case 1: Item is in A1out (ghost list)
        if key in self.a1out:
            # Move from A1out to Am
            self.a1out.pop(key)
            self.am[key] = None
        # Case 2: New item
        else:
            # Always add to A1in first
            self.a1in[key] = None
            # Make space in A1in if needed
            while len(self.a1in) > self.a1in_size and self.a1in:
                # Move LRU item from A1in to A1out
                lru_key, _ = self.a1in.popitem(last=False)
                # Remove from cache but keep in ghost list
                if lru_key in self.cache:
                    evicted = self.cache.pop(lru_key)
                    self.size_used -= evicted.size
                    # Add to A1out (ghost list)
                    self.a1out[lru_key] = None
                    # Trim A1out if needed
                    while len(self.a1out) > self.a1out_size and self.a1out:
                        self.a1out.popitem(last=False)
            
    def _evict(self) -> None:
        """Evict an item based on the 2Q policy."""
        # First try to evict from A1in
        if self.a1in:
            lru_key, _ = self.a1in.popitem(last=False)
            # Move to A1out (ghost list)
            self.a1out[lru_key] = None
            # Trim A1out if needed
            while len(self.a1out) > self.a1out_size and self.a1out:
                self.a1out.popitem(last=False)
        # Then try to evict from Am
        elif self.am:
            lru_key, _ = self.am.popitem(last=False)
        else:
            # No items to evict
            return
            
        if lru_key in self.cache:
            evicted = self.cache.pop(lru_key)
            self.size_used -= evicted.size

class MLFeatureExtractor:
    """Extracts features from cache access patterns for ML prediction."""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    def extract_features(self, 
                         key: str, 
                         cache: BaseCache, 
                         global_access_history: List[Tuple[str, float]]) -> List[float]:
        """Extract features for a given key and cache state."""
        features = []
        
        # 1. Recent history features
        recent_history = global_access_history[-self.window_size:] if global_access_history else []
        
        # Calculate recency
        key_timestamps = [ts for k, ts in recent_history if k == key]
        if key_timestamps:
            last_access = max(key_timestamps)
            time_since_last_access = time.time() - last_access
            features.append(time_since_last_access)
        else:
            # Never seen this key before
            features.append(float('inf'))
            
        # Calculate frequency in recent window
        key_freq = sum(1 for k, _ in recent_history if k == key)
        features.append(key_freq)
        
        # 2. Key-specific patterns
        if key in cache.cache:
            entry = cache.cache[key]
            # Age of entry in cache
            features.append(time.time() - entry.created_at)
            # Access count
            features.append(entry.access_count)
            # Size of entry
            features.append(entry.size)
        else:
            # Default values for new entries
            features.append(0)  # Age
            features.append(0)  # Access count
            features.append(1)  # Default size
        
        # 3. Global cache state features
        features.append(len(cache.cache) / cache.capacity)  # Fill ratio
        features.append(cache.size_used / cache.capacity)  # Size ratio
        features.append(cache._get_hit_ratio())  # Hit ratio
        
        return features

class AdaptiveCache(BaseCache):
    """Adaptive cache that uses ML to predict item utility."""
    def __init__(self, 
                 base_cache: str = "lru",
                 ml_update_frequency: int = 1000,
                 **kwargs):
        # Remove 'name' from kwargs to avoid duplicate argument
        kwargs.pop('name', None)
        super().__init__(name="Adaptive ML Cache", **kwargs)
        
        # Create the base cache
        self.base_cache_type = base_cache.lower()
        if self.base_cache_type == "lru":
            self.base_cache = LRUCache(**kwargs)
        elif self.base_cache_type == "lfu":
            self.base_cache = LFUCache(**kwargs)
        elif self.base_cache_type == "arc":
            self.base_cache = ARCache(**kwargs)
        elif self.base_cache_type == "2q":
            self.base_cache = TwoQCache(**kwargs)
        else:
            self.base_cache = LRUCache(**kwargs)
            
        # ML components
        self.feature_extractor = MLFeatureExtractor()
        self.model = RandomForestRegressor(n_estimators=10, max_depth=5)
        self.model_trained = False
        self.training_data_x = []
        self.training_data_y = []
        
        # Access history for ML
        self.global_access_history = []
        self.ml_update_frequency = ml_update_frequency
        self.access_count = 0
        
        # Utility scores for items (key -> score)
        self.utility_scores = {}
        
    def get(self, key: str) -> Tuple[Any, float]:
        """Get an item using ML prediction for cache update."""
        # Record access in history
        self.global_access_history.append((key, time.time()))
        if len(self.global_access_history) > 10000:  # Limit history size
            self.global_access_history = self.global_access_history[-5000:]
        
        # Use base cache to get the item
        result, latency = super().get(key)
        
        # Prepare training data if item was in cache
        if result is not None:
            # Extract features before hit
            features = self.feature_extractor.extract_features(
                key, self, self.global_access_history[:-1])
            
            # The label is 1 for a hit (useful item)
            self.training_data_x.append(features)
            self.training_data_y.append(1)
            
        # Update ML model periodically
        self.access_count += 1
        if self.access_count % self.ml_update_frequency == 0:
            self._update_ml_model()
            
        return result, latency
        
    def _update_ml_model(self):
        """Update the ML model with collected training data."""
        if len(self.training_data_x) > 100:  # Need sufficient data
            # Train the model
            try:
                self.model.fit(self.training_data_x, self.training_data_y)
                self.model_trained = True
                
                # Update utility scores for all items in cache
                for key in list(self.cache.keys()):
                    features = self.feature_extractor.extract_features(
                        key, self, self.global_access_history)
                    prediction = self.model.predict([features])[0]
                    self.utility_scores[key] = prediction
            except Exception as e:
                print(f"Error training ML model: {e}")
                
            # Trim training data to avoid memory issues
            max_samples = 10000
            if len(self.training_data_x) > max_samples:
                self.training_data_x = self.training_data_x[-max_samples:]
                self.training_data_y = self.training_data_y[-max_samples:]
                
    def _evict(self) -> None:
        """Evict an item based on ML utility prediction."""
        if not self.model_trained or len(self.cache) < 10:
            # Fall back to base policy if model not ready
            if self.base_cache_type == "lru":
                # Find least recently used item
                lru_key = next(iter(self.base_cache.access_order)) if self.base_cache.access_order else None
                if lru_key in self.cache:
                    evicted = self.cache.pop(lru_key)
                    self.size_used -= evicted.size
                    if hasattr(self.base_cache, 'access_order'):
                        self.base_cache.access_order.pop(lru_key, None)
            else:
                # Use base cache's eviction for other types
                self.base_cache._evict()
                # Sync our cache with base cache
                keys_to_remove = set(self.cache.keys()) - set(self.base_cache.cache.keys())
                for key in keys_to_remove:
                    if key in self.cache:
                        evicted = self.cache.pop(key)
                        self.size_used -= evicted.size
        else:
            # Use ML to find item with lowest utility
            min_utility = float('inf')
            min_key = None
            
            # Update predictions for all items
            for key in list(self.cache.keys()):
                if key not in self.utility_scores:
                    features = self.feature_extractor.extract_features(
                        key, self, self.global_access_history)
                    try:
                        prediction = self.model.predict([features])[0]
                        self.utility_scores[key] = prediction
                    except:
                        # Fall back to base cache order if prediction fails
                        self.utility_scores[key] = 0.5
                        
                if self.utility_scores[key] < min_utility:
                    min_utility = self.utility_scores[key]
                    min_key = key
                    
            # Evict the item with lowest utility
            if min_key in self.cache:
                evicted = self.cache.pop(min_key)
                self.size_used -= evicted.size
                self.utility_scores.pop(min_key, None)
                
                # Also remove from base cache for consistency

# Continue from where the provided code left off...

class DistributedCacheCluster:
    """Simulates a distributed cache cluster with multiple nodes."""
    def __init__(self, 
                 node_count: int = 3,
                 cache_type: str = "lru",
                 capacity_per_node: int = 100,
                 replication_factor: int = 2,
                 consistency_level: str = "quorum",
                 network_simulator: Optional[NetworkSimulator] = None,
                 disk_simulator: Optional[DiskIOSimulator] = None):
        """
        Initialize a distributed cache cluster.
        
        Args:
            node_count: Number of cache nodes in the cluster
            cache_type: Type of cache to use ("lru", "lfu", "arc", "2q", "adaptive")
            capacity_per_node: Cache capacity for each node
            replication_factor: Number of nodes each item is replicated to
            consistency_level: Consistency model ("one", "quorum", "all")
            network_simulator: Custom network simulator or None for default
            disk_simulator: Custom disk simulator or None for default
        """
        self.node_count = node_count
        self.replication_factor = min(replication_factor, node_count)
        self.consistency_level = consistency_level
        
        # Create network and disk simulators
        self.network = network_simulator or NetworkSimulator()
        self.disk = disk_simulator or DiskIOSimulator()
        
        # Create cache nodes
        self.nodes = []
        for i in range(node_count):
            if cache_type.lower() == "lru":
                cache = LRUCache(capacity=capacity_per_node, 
                                name=f"LRU-Node-{i}",
                                network_simulator=self.network,
                                disk_simulator=self.disk)
            elif cache_type.lower() == "lfu":
                cache = LFUCache(capacity=capacity_per_node, 
                                name=f"LFU-Node-{i}",
                                network_simulator=self.network,
                                disk_simulator=self.disk)
            elif cache_type.lower() == "arc":
                cache = ARCache(capacity=capacity_per_node, 
                               name=f"ARC-Node-{i}",
                               network_simulator=self.network,
                               disk_simulator=self.disk)
            elif cache_type.lower() == "2q":
                cache = TwoQCache(capacity=capacity_per_node, 
                                 name=f"2Q-Node-{i}",
                                 network_simulator=self.network,
                                 disk_simulator=self.disk)
            elif cache_type.lower() == "adaptive":
                cache = AdaptiveCache(capacity=capacity_per_node, 
                                     name=f"ML-Node-{i}",
                                     network_simulator=self.network,
                                     disk_simulator=self.disk)
            else:
                cache = LRUCache(capacity=capacity_per_node, 
                                name=f"Default-Node-{i}",
                                network_simulator=self.network,
                                disk_simulator=self.disk)
            self.nodes.append(cache)
            
        # Consistent hashing ring for node selection
        self.build_hash_ring()
        
        # Stats
        self.stats = {
            "total_gets": 0,
            "total_puts": 0,
            "get_hits": 0,
            "get_misses": 0,
            "avg_latency": 0,
            "p95_latency": 0,
            "p99_latency": 0,
            "replication_failures": 0,
            "latency_history": [],
            "consistency_violations": 0
        }
        
        # For simulating eventual consistency
        self.pending_updates = queue.Queue()
        self.eventual_consistency_thread = threading.Thread(
            target=self._process_eventual_updates, daemon=True)
        self.eventual_consistency_thread.start()
        
    def build_hash_ring(self, virtual_nodes: int = 100):
        """Build a consistent hashing ring for node selection."""
        self.hash_ring = []
        for node_idx in range(self.node_count):
            for v in range(virtual_nodes):
                # Create a hash point for this virtual node
                hash_key = f"node{node_idx}-vn{v}"
                hash_val = hash(hash_key) % (2**32)
                self.hash_ring.append((hash_val, node_idx))
        
        # Sort the hash ring
        self.hash_ring.sort(key=lambda x: x[0])
        
    def get_node_for_key(self, key: str) -> List[int]:
        """
        Get the set of nodes responsible for a key using consistent hashing.
        Returns list of node indices in order of preference.
        """
        if not self.hash_ring:
            return list(range(min(self.replication_factor, self.node_count)))
            
        # Find the first point in the ring >= the hash of the key
        key_hash = hash(key) % (2**32)
        nodes = []
        
        # Find primary node
        for i, (hash_val, node_idx) in enumerate(self.hash_ring):
            if hash_val >= key_hash:
                nodes.append(node_idx)
                break
        
        # If we didn't find a node or need more replicas, wrap around
        if not nodes:
            nodes.append(self.hash_ring[0][1])
        
        # Find replica nodes (next nodes in the ring)
        if len(nodes) < self.replication_factor:
            seen = set(nodes)
            idx = 0 if not nodes else self.hash_ring.index((key_hash, nodes[0]))
            
            while len(nodes) < self.replication_factor and len(seen) < self.node_count:
                idx = (idx + 1) % len(self.hash_ring)
                next_node = self.hash_ring[idx][1]
                if next_node not in seen:
                    nodes.append(next_node)
                    seen.add(next_node)
        
        return nodes[:self.replication_factor]
    
    def get(self, key: str) -> Tuple[Any, float]:
        """
        Get a value from the distributed cache.
        Returns (value, latency) tuple.
        """
        start_time = time.time()
        self.stats["total_gets"] += 1
        
        # Get nodes responsible for this key
        responsible_nodes = self.get_node_for_key(key)
        
        # Determine how many nodes we need responses from
        if self.consistency_level == "one":
            required_responses = 1
        elif self.consistency_level == "all":
            required_responses = len(responsible_nodes)
        else:  # "quorum"
            required_responses = (len(responsible_nodes) // 2) + 1
            
        # Query the responsible nodes
        results = []
        latencies = []
        
        with ThreadPoolExecutor(max_workers=len(responsible_nodes)) as executor:
            # Create a future for each node query
            future_to_node = {
                executor.submit(self.nodes[node_idx].get, key): node_idx
                for node_idx in responsible_nodes
            }
            
            # Collect results as they complete
            for future in future_to_node:
                try:
                    value, node_latency = future.result()
                    node_idx = future_to_node[future]
                    results.append((value, node_idx))
                    latencies.append(node_latency)
                except Exception as e:
                    # Node failure simulation
                    print(f"Node failure during get: {e}")
            
        # Check if we have enough responses for our consistency level
        if len(results) < required_responses:
            # Not enough responses to satisfy consistency
            self.stats["get_misses"] += 1
            total_latency = time.time() - start_time
            self.stats["latency_history"].append(total_latency)
            return None, total_latency
            
        # Check for consistency among the values
        values = [r[0] for r in results if r[0] is not None]
        if not values:
            # All responses were cache misses
            self.stats["get_misses"] += 1
            total_latency = time.time() - start_time
            self.stats["latency_history"].append(total_latency)
            return None, total_latency
            
        # Check if values are consistent
        if len(set(str(v) for v in values)) > 1:
            # Inconsistency detected!
            self.stats["consistency_violations"] += 1
            
        # Return the first non-None value
        self.stats["get_hits"] += 1
        total_latency = time.time() - start_time
        
        # Update latency stats
        self.stats["latency_history"].append(total_latency)
        self._update_latency_stats()
        
        return values[0], total_latency
        
    def put(self, key: str, value: Any, size: int = 1) -> float:
        """
        Put a value into the distributed cache.
        Returns the total latency.
        """
        start_time = time.time()
        self.stats["total_puts"] += 1
        
        # Get nodes responsible for this key
        responsible_nodes = self.get_node_for_key(key)
        
        # Determine how many nodes we need responses from
        if self.consistency_level == "one":
            required_responses = 1
        elif self.consistency_level == "all":
            required_responses = len(responsible_nodes)
        else:  # "quorum"
            required_responses = (len(responsible_nodes) // 2) + 1
        
        # Strong consistency: synchronous writes
        if self.consistency_level in ["quorum", "all"]:
            successful_writes = 0
            latencies = []
            
            with ThreadPoolExecutor(max_workers=len(responsible_nodes)) as executor:
                # Create a future for each node update
                future_to_node = {
                    executor.submit(self.nodes[node_idx].put, key, value, size): node_idx
                    for node_idx in responsible_nodes
                }
                
                # Collect results as they complete
                for future in future_to_node:
                    try:
                        node_latency = future.result()
                        latencies.append(node_latency)
                        successful_writes += 1
                    except Exception as e:
                        # Node failure simulation
                        print(f"Node failure during put: {e}")
            
            # Check if we have enough successful writes
            if successful_writes < required_responses:
                self.stats["replication_failures"] += 1
        
        # Eventual consistency: asynchronous writes
        else:  # self.consistency_level == "one"
            # Just write to one node synchronously
            try:
                self.nodes[responsible_nodes[0]].put(key, value, size)
                
                # Queue updates for other nodes
                for node_idx in responsible_nodes[1:]:
                    self.pending_updates.put((node_idx, key, value, size))
            except Exception as e:
                # Primary node failure
                print(f"Primary node failure during put: {e}")
                self.stats["replication_failures"] += 1
                
        total_latency = time.time() - start_time
        self.stats["latency_history"].append(total_latency)
        self._update_latency_stats()
        
        return total_latency
    
    def _process_eventual_updates(self):
        """Background thread to process eventual consistency updates."""
        while True:
            try:
                # Get an update from the queue
                node_idx, key, value, size = self.pending_updates.get(timeout=1)
                
                # Try to apply the update
                try:
                    self.nodes[node_idx].put(key, value, size)
                except Exception as e:
                    # Replication failure
                    print(f"Replication failure for node {node_idx}: {e}")
                    self.stats["replication_failures"] += 1
                    
                self.pending_updates.task_done()
            except queue.Empty:
                # No updates in queue, sleep briefly
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in eventual consistency thread: {e}")
                time.sleep(1)
    
    def _update_latency_stats(self):
        """Update latency statistics."""
        latencies = self.stats["latency_history"][-1000:]  # Use last 1000 operations
        if latencies:
            self.stats["avg_latency"] = np.mean(latencies)
            self.stats["p95_latency"] = np.percentile(latencies, 95)
            self.stats["p99_latency"] = np.percentile(latencies, 99)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from all nodes and cluster."""
        node_stats = [node.get_stats() for node in self.nodes]
        
        # Calculate hit rate
        total_ops = self.stats["total_gets"]
        hit_rate = self.stats["get_hits"] / total_ops if total_ops > 0 else 0
        
        cluster_stats = {
            "node_count": self.node_count,
            "replication_factor": self.replication_factor,
            "consistency_level": self.consistency_level,
            "get_operations": self.stats["total_gets"],
            "put_operations": self.stats["total_puts"],
            "hit_rate": hit_rate,
            "avg_latency_ms": self.stats["avg_latency"] * 1000,
            "p95_latency_ms": self.stats["p95_latency"] * 1000,
            "p99_latency_ms": self.stats["p99_latency"] * 1000,
            "replication_failures": self.stats["replication_failures"],
            "consistency_violations": self.stats["consistency_violations"],
            "nodes": node_stats
        }
        
        return cluster_stats

class WorkloadGenerator:
    """Generates configurable workloads for cache simulation."""
    def __init__(self, 
                 key_space_size: int = 1000,
                 value_size_range: Tuple[int, int] = (1, 10),
                 read_write_ratio: float = 0.8,  # 80% reads, 20% writes
                 zipf_alpha: float = 0.9,  # Zipfian distribution parameter
                 key_pattern: str = "zipfian",  # zipfian, uniform, shifting
                 burst_factor: float = 1.0,  # Multiplier for bursts
                 temporal_locality: float = 0.3):  # Probability of repeating recent keys
        """
        Initialize a workload generator.
        
        Args:
            key_space_size: Number of unique keys in the workload
            value_size_range: Range of sizes for values (min, max)
            read_write_ratio: Ratio of read to total operations (0.8 = 80% reads)
            zipf_alpha: Zipfian distribution parameter (higher = more skewed)
            key_pattern: Distribution pattern for key selection
            burst_factor: Multiplier for simulating traffic bursts
            temporal_locality: Probability of selecting recently accessed keys
        """
        # Validate and correct zipf_alpha
        if not isinstance(zipf_alpha, (int, float)) or zipf_alpha <= 1:
            print(f"Invalid zipf_alpha ({zipf_alpha}), setting to default value of 1.1.")
            zipf_alpha = 1.1
        
        self.key_space_size = key_space_size
        self.value_size_range = value_size_range
        self.read_write_ratio = read_write_ratio
        self.zipf_alpha = zipf_alpha
        self.key_pattern = key_pattern
        self.burst_factor = burst_factor
        self.temporal_locality = temporal_locality
        
        # Generate zipfian distribution for keys
        self.zipf_dist = np.random.zipf(self.zipf_alpha, size=10000)
        self.zipf_dist = self.zipf_dist % self.key_space_size
        
        # Shifting pattern support
        self.current_shift = 0
        self.shift_interval = 100  # operations per shift
        self.shift_amount = int(key_space_size * 0.1)  # 10% of key space
        
        # Recent keys for temporal locality
        self.recent_keys = []
        self.max_recent_keys = 50
        
        # Burst simulation
        self.burst_mode = False
        self.burst_countdown = 0
        self.burst_duration = (20, 100)  # Range of burst duration
        self.burst_interval = (100, 500)  # Range of intervals between bursts
    
    def generate_key(self) -> str:
        """Generate a key based on the configured pattern."""
        # Check for temporal locality first
        if self.recent_keys and random.random() < self.temporal_locality:
            return random.choice(self.recent_keys)
            
        if self.key_pattern == "uniform":
            # Uniform distribution
            key_idx = random.randint(0, self.key_space_size - 1)
        elif self.key_pattern == "shifting":
            # Shifting pattern - keys shift over time
            if random.random() < 0.7:  # 70% in current hot zone
                base = self.current_shift % self.key_space_size
                key_idx = (base + random.randint(0, self.shift_amount)) % self.key_space_size
            else:
                key_idx = random.randint(0, self.key_space_size - 1)
        else:  # "zipfian" (default)
            # Zipfian distribution - some keys are much more popular
            key_idx = self.zipf_dist[random.randint(0, len(self.zipf_dist) - 1)]
            
        key = f"key-{key_idx}"
        
        # Add to recent keys
        self.recent_keys.append(key)
        if len(self.recent_keys) > self.max_recent_keys:
            self.recent_keys.pop(0)
            
        return key
    
    def generate_value(self) -> Tuple[Any, int]:
        """Generate a value and its size."""
        # Generate a random size for the value
        size = random.randint(self.value_size_range[0], self.value_size_range[1])
        
        # Generate a random value (just using a string of that size)
        value = f"value-{random.randint(0, 1000000)}-{'-' * size}"
        
        return value, size
    
    def next_operation(self) -> Tuple[str, str, Any, int]:
        """
        Generate the next cache operation.
        Returns (operation, key, value, size) where operation is "get" or "put".
        """
        # Update shifting pattern if applicable
        if self.key_pattern == "shifting":
            self.current_shift = (self.current_shift + 1) // self.shift_interval * self.shift_amount
        
        # Check for burst mode transitions
        if not self.burst_mode and random.random() < 0.01:  # 1% chance to start a burst
            self.burst_mode = True
            self.burst_countdown = random.randint(*self.burst_duration)
        elif self.burst_mode:
            self.burst_countdown -= 1
            if self.burst_countdown <= 0:
                self.burst_mode = False
                self.burst_countdown = random.randint(*self.burst_interval)
        
        # Apply burst factor if in burst mode
        current_burst = self.burst_factor if self.burst_mode else 1.0
        
        # Determine operation type (get or put)
        is_read = random.random() < self.read_write_ratio
        operation = "get" if is_read else "put"
        
        # Generate key and value
        key = self.generate_key()
        value, size = self.generate_value() if operation == "put" else (None, 0)
        
        # Apply burst factor for faster operations
        if self.burst_mode:
            # For simulation, we don't actually change timing,
            # but we could modify metadata here
            pass
        
        return operation, key, value, size

class CacheSimulation:
    """Runs a complete cache simulation with metrics collection."""
    def __init__(self,
                 cache_type: str = "adaptive",
                 num_nodes: int = 3,
                 replication_factor: int = 2,
                 consistency_level: str = "quorum",
                 capacity_per_node: int = 1000,
                 network_latency_ms: float = 5.0,
                 disk_read_latency_ms: float = 8.0,
                 disk_write_latency_ms: float = 15.0,
                 key_space_size: int = 10000,
                 operations_count: int = 100000,
                 read_write_ratio: float = 0.8,
                 zipf_alpha: float = 0.9,
                 key_pattern: str = "zipfian",
                 concurrency_level: int = 5,
                 enable_scaling: bool = False,
                 scaling_threshold: float = 0.7,
                 burst_factor: float = 2.0,
                 failure_rate: float = 0.0001):
        """
        Initialize a complete cache simulation.
        
        Args:
            cache_type: Type of cache algorithm to use
            num_nodes: Initial number of nodes in the cluster
            replication_factor: Number of replicas for each item
            consistency_level: Consistency model to use
            capacity_per_node: Cache capacity per node
            network_latency_ms: Base network latency in milliseconds
            disk_read_latency_ms: Base disk read latency in milliseconds
            disk_write_latency_ms: Base disk write latency in milliseconds
            key_space_size: Number of unique keys in the workload
            operations_count: Total number of operations to simulate
            read_write_ratio: Ratio of reads to total operations
            zipf_alpha: Skew parameter for Zipfian key distribution
            key_pattern: Pattern for key access distribution
            concurrency_level: Number of concurrent client threads
            enable_scaling: Whether to enable auto-scaling
            scaling_threshold: Latency threshold for scaling
            burst_factor: Multiplier for traffic bursts
            failure_rate: Probability of node failures
        """
        # Simulation parameters
        self.cache_type = cache_type
        self.num_nodes = num_nodes
        self.replication_factor = replication_factor
        self.consistency_level = consistency_level
        self.capacity_per_node = capacity_per_node
        self.operations_count = operations_count
        self.concurrency_level = concurrency_level
        self.enable_scaling = enable_scaling
        self.scaling_threshold = scaling_threshold
        self.failure_rate = failure_rate
        
        # Create network and disk simulators
        self.network_simulator = NetworkSimulator(
            base_latency_ms=network_latency_ms,
            jitter_ms=network_latency_ms * 0.4,
            packet_loss_percent=0.1
        )
        
        self.disk_simulator = DiskIOSimulator(
            read_latency_ms=disk_read_latency_ms,
            write_latency_ms=disk_write_latency_ms,
            read_variance=0.3,
            write_variance=0.5,
            disk_failure_rate=failure_rate
        )
        
        # Create the distributed cache cluster
        self.cluster = DistributedCacheCluster(
            node_count=num_nodes,
            cache_type=cache_type,
            capacity_per_node=capacity_per_node,
            replication_factor=replication_factor,
            consistency_level=consistency_level,
            network_simulator=self.network_simulator,
            disk_simulator=self.disk_simulator
        )
        
        # Create workload generator
        self.workload = WorkloadGenerator(
            key_space_size=key_space_size,
            read_write_ratio=read_write_ratio,
            zipf_alpha=zipf_alpha,
            key_pattern=key_pattern,
            burst_factor=burst_factor
        )
        
        # Metrics collection
        self.metrics = {
            "throughput_history": [],
            "latency_history": [],
            "hit_ratio_history": [],
            "node_count_history": [],
            "operation_counts": {"get": 0, "put": 0},
            "start_time": time.time(),
            "end_time": None,
            "scaling_events": [],
            "failure_events": []
        }
        
        # For concurrent operations
        self.operation_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.workers = []
        self.stop_flag = threading.Event()
        
    def _worker_thread(self, worker_id: int):
        """Worker thread for processing cache operations."""
        while not self.stop_flag.is_set():
            try:
                # Get an operation from the queue
                try:
                    op_data = self.operation_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                operation, key, value, size = op_data
                
                # Execute the operation
                if operation == "get":
                    result, latency = self.cluster.get(key)
                    self.results_queue.put(("get", key, result, latency))
                else:  # "put"
                    latency = self.cluster.put(key, value, size)
                    self.results_queue.put(("put", key, True, latency))
                    
                # Mark task as done
                self.operation_queue.task_done()
                
                # Simulate node failure
                if random.random() < self.failure_rate:
                    self.metrics["failure_events"].append({
                        "time": time.time() - self.metrics["start_time"],
                        "worker_id": worker_id
                    })
                    # Simulate worker restart
                    time.sleep(random.uniform(0.5, 2.0))
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)
    
    def start_workers(self):
        """Start worker threads for concurrent operations."""
        for i in range(self.concurrency_level):
            worker = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        """Stop all worker threads."""
        self.stop_flag.set()
        for worker in self.workers:
            worker.join(timeout=2)
            
    def check_scaling(self, current_latency: float):
        """Check if we need to scale the cluster."""
        if not self.enable_scaling:
            return
            
        # Scale up if latency exceeds threshold
        latency_ms = current_latency * 1000
        if latency_ms > self.scaling_threshold and self.num_nodes < 10:
            # Add a node
            self.num_nodes += 1
            
            # Create a new cluster with more nodes
            old_cluster = self.cluster
            self.cluster = DistributedCacheCluster(
                node_count=self.num_nodes,
                cache_type=self.cache_type,
                capacity_per_node=self.capacity_per_node,
                replication_factor=self.replication_factor,
                consistency_level=self.consistency_level,
                network_simulator=self.network_simulator,
                disk_simulator=self.disk_simulator
            )
            
            # Record scaling event
            self.metrics["scaling_events"].append({
                "time": time.time() - self.metrics["start_time"],
                "direction": "up",
                "nodes": self.num_nodes,
                "trigger_latency_ms": latency_ms
            })
            
            # Log the scaling event
            print(f"Scaling up to {self.num_nodes} nodes due to high latency ({latency_ms:.2f}ms)")
            
        # Scale down if latency is low
        elif latency_ms < (self.scaling_threshold * 0.5) and self.num_nodes > 1:
            # Remove a node
            self.num_nodes -= 1
            
            # Create a new cluster with fewer nodes
            old_cluster = self.cluster
            self.cluster = DistributedCacheCluster(
                node_count=self.num_nodes,
                cache_type=self.cache_type,
                capacity_per_node=self.capacity_per_node,
                replication_factor=min(self.replication_factor, self.num_nodes),
                consistency_level=self.consistency_level,
                network_simulator=self.network_simulator,
                disk_simulator=self.disk_simulator
            )
            
            # Record scaling event
            self.metrics["scaling_events"].append({
                "time": time.time() - self.metrics["start_time"],
                "direction": "down",
                "nodes": self.num_nodes,
                "trigger_latency_ms": latency_ms
            })
            
            # Log the scaling event
            print(f"Scaling down to {self.num_nodes} nodes due to low latency ({latency_ms:.2f}ms)")
    
    def run_simulation(self):
        """Run the complete simulation."""
        print(f"Starting simulation with {self.cache_type} cache on {self.num_nodes} nodes...")
        print(f"Simulating {self.operations_count} operations with {self.concurrency_level} concurrent clients")
        
        # Start worker threads
        self.start_workers()
        
        # Process operations
        ops_complete = 0
        last_progress = 0
        last_latencies = []
        last_progress_time = time.time()
        
        # Generate and queue all operations
        for _ in range(self.operations_count):
            op_data = self.workload.next_operation()
            self.operation_queue.put(op_data)
            
        # Process results as they complete
        while ops_complete < self.operations_count:
            try:
                # Get result from the results queue
                try:
                    operation, key, result, latency = self.results_queue.get(timeout=1)
                    self.results_queue.task_done()
                except queue.Empty:
                    continue
                
                # Update metrics
                self.metrics["operation_counts"][operation] += 1
                self.metrics["latency_history"].append(latency)
                last_latencies.append(latency)
                
                # Update hit ratio if we have cluster stats
                cluster_stats = self.cluster.get_stats()
                self.metrics["hit_ratio_history"].append(cluster_stats["hit_rate"])
                self.metrics["node_count_history"].append(self.num_nodes)
                
                # Check scaling every 1000 operations
                if ops_complete % 1000 == 0 and last_latencies:
                    avg_latency = np.mean(last_latencies)
                    self.check_scaling(avg_latency)
                    last_latencies = []
                
                # Update progress
                ops_complete += 1
                progress = int(ops_complete / self.operations_count * 100)
                
                # Calculate and update throughput every 5%
                if progress % 5 == 0 and progress != last_progress:
                    current_time = time.time()
                    elapsed_time = current_time - last_progress_time
                    throughput = (ops_complete - last_progress) / elapsed_time
                    self.metrics["throughput_history"].append(throughput)
                    last_progress = progress
                    last_progress_time = current_time
                    print(f"Progress: {progress}% - Throughput: {throughput:.2f} ops/sec")
                    
            except Exception as e:
                print(f"Error processing result: {e}")
                time.sleep(0.1)
        
        # Final metrics collection
        self.metrics["end_time"] = time.time()
        print("Simulation complete.")
    
    def get_final_metrics(self):
        """Return final metrics including scaling and failure events."""
        return {
            "scaling_events": self.metrics["scaling_events"],
            "failure_events": self.metrics["failure_events"],
            "metrics": self.metrics,
        }

# Main entry point for running the simulation
if __name__ == "__main__":
    simulation = CacheSimulation(
        cache_type="adaptive",
        num_nodes=3,
        replication_factor=2,
        consistency_level="quorum",
        capacity_per_node=1000,
        operations_count=100000,
        read_write_ratio=0.8,
        zipf_alpha=0.9,
        key_pattern="zipfian",
        concurrency_level=5,
        enable_scaling=True,
        scaling_threshold=100,
        burst_factor=2.0,
        failure_rate=0.001,
    )
    simulation.run_simulation()
    final_metrics = simulation.get_final_metrics()
    print(final_metrics)