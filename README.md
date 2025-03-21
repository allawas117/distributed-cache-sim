# Distributed Cache Simulator

## Overview

**Currently a work in progress!!!**

The Distributed Cache Simulator is a Python-based framework for evaluating caching algorithms under distributed systems conditions. It supports features such as:

- **Network Latency Simulation**: Models network delays, jitter, packet loss, and partitions.
- **Disk I/O Simulation**: Simulates read/write latencies and disk failures.
- **Concurrency**: Supports multi-threaded workloads.
- **Adaptive Caching**: Includes ML-based adaptive caching with feature extraction and prediction.
- **Workload Generation**: Configurable workloads with Zipfian, uniform, and shifting key patterns.
- **Auto-Scaling**: Simulates Kubernetes-like scaling based on latency thresholds.
- **Metrics Collection**: Tracks throughput, latency, hit ratios, and scaling events.

## Features

- **Caching Algorithms**:
  - LRU (Least Recently Used)
  - LFU (Least Frequently Used)
  - ARC (Adaptive Replacement Cache)
  - 2Q Cache
  - Adaptive ML-based Cache

- **Distributed Cluster**:
  - Consistent hashing for node selection.
  - Configurable replication factor and consistency levels (`one`, `quorum`, `all`).

- **Workload Generator**:
  - Supports Zipfian, uniform, and shifting key patterns.
  - Configurable read/write ratio, burst traffic, and temporal locality.

- **Simulation**:
  - Simulates network and disk failures.
  - Auto-scaling based on latency thresholds.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd distributed-cache-simulator
   ```

2. Ensure Python 3.7+ is installed.

## Usage

Run the simulation with default parameters:
```bash
python distributedcachesim.py
```

### Example Configuration

Modify the `CacheSimulation` parameters in `distributedcachesim.py` to customize the simulation:
```python
simulation = CacheSimulation(
    cache_type="adaptive",          # Cache type: lru, lfu, arc, 2q, adaptive
    num_nodes=3,                    # Number of nodes in the cluster
    replication_factor=2,           # Number of replicas per item
    consistency_level="quorum",     # Consistency model: one, quorum, all
    capacity_per_node=1000,         # Cache capacity per node
    operations_count=100000,        # Total number of operations
    read_write_ratio=0.8,           # Ratio of reads to total operations
    zipf_alpha=1.2,                 # Zipfian distribution parameter (>1)
    key_pattern="zipfian",          # Key pattern: zipfian, uniform, shifting
    concurrency_level=5,            # Number of concurrent clients
    enable_scaling=True,            # Enable auto-scaling
    scaling_threshold=100,          # Latency threshold for scaling (ms)
    burst_factor=2.0,               # Traffic burst multiplier
    failure_rate=0.001              # Probability of node failures
)
```

### Output

The simulation outputs metrics such as:
- Throughput
- Latency (average, p95, p99)
- Hit ratio
- Scaling events
- Failure events

Example output:
```json
{
    "scaling_events": [...],
    "failure_events": [...],
    "metrics": {
        "throughput_history": [...],
        "latency_history": [...],
        "hit_ratio_history": [...],
        "node_count_history": [...],
        "operation_counts": {"get": 80000, "put": 20000},
        "start_time": 1690000000.0,
        "end_time": 1690003600.0
    }
}
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, please contact [ash.lastimosa@gmail.com].
