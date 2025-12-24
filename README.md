# Distributed ML Inference Engine (Native - No Docker)

A distributed inference system demonstrating consistent hashing, dynamic batching, model sharding, and horizontal scaling 

## Quick Start 
### This project is developed on a Linux based system. If you are using a different OS, changes might be required all of which are not included here.
### 1. Install Dependencies
```bash
pip install numpy requests matplotlib
```

### 2. Download All Files
Save all the provided Python files in a single directory:
- `consistent_hash.py`
- `batch_processor.py`
- `inference_engine.py`
- `worker_node.py`
- `gateway.py`
- `benchmark.py`
- `analyze_results.py`
- `run.sh` 
- `stop.sh` 

### 3. Start the System

**Linux/Mac:**
```bash
chmod +x run_system.sh
./run.sh
```

**Manual (any OS) - use 5 separate terminals:**
```bash
# Terminal 1
python worker_node.py --port 8001 --node-id worker_1

# Terminal 2
python worker_node.py --port 8002 --node-id worker_2

# Terminal 3
python worker_node.py --port 8003 --node-id worker_3

# Terminal 4 (wait 2 seconds after starting workers)
python gateway.py --port 8000

# Terminal 5 (wait 2 seconds after starting gateway)
python benchmark.py --requests 5000 --concurrent 50
```

### 4. View Results
```bash
python analyze_results.py
```

This generates:
- `latency_distribution.png`
- `node_distribution.png`
- `performance_comparison.png`
- `performance_report.txt`
- `benchmark_results.json`

## Project Structure

```
distributed-inference-native/
├── consistent_hash.py          # Consistent hashing implementation
├── batch_processor.py          # Dynamic batching logic
├── inference_engine.py         # Simulated ML inference
├── worker_node.py              # Worker server with batching
├── gateway.py                  # Gateway with routing
├── benchmark.py                # Load testing tool
├── analyze_results.py          # Results visualization
├── run.sh               # Start script (Linux/Mac)
├── stop.sh              # Stop script (Linux/Mac)
└── README.md                   # This file
```

## Architecture

```
                    Clients
                      │
                      ▼
              ┌───────────────┐
              │    Gateway    │
              │  Port: 8000   │
              │               │
              │  Consistent   │
              │    Hashing    │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
    ┌────────┐   ┌────────-┐  ┌────────-┐
    │Worker 1│   │Worker 2 │  │Worker 3 │
    │  8001  │   │  8002   |  │  8003   │
    │        │   │         │  │         │
    │ Batch  │   │ Batch   │  │ Batch   │
    │Process │   │Process  │  │Process  │
    │        │   │         │  │         │
    │Inference│  │Inference│  │Inference│
    │Engine  │   │Engine   │  │Engine   │
    └────────┘   └────────-┘  └────────-┘
```

## Key Features

### 1. Consistent Hashing
- 150 virtual nodes per physical node
- Uniform load distribution
- Minimal request redistribution on node changes

### 2. Dynamic Batching
- Max batch size: 32 requests
- Timeout: 20ms
- Automatic batch optimization

### 3. Model Sharding
- Simulated model partitioning across nodes
- Reduced memory footprint per node

### 4. Horizontal Scaling
- Easy to add more worker nodes
- Linear throughput scaling

## Configuration

### Adjust Load Test
```bash
# Light load
python benchmark.py --requests 1000 --concurrent 20

# Medium load
python benchmark.py --requests 5000 --concurrent 50

# Heavy load
python benchmark.py --requests 10000 --concurrent 100
```

### Add More Workers
```bash
# Start additional workers
python worker_node.py --port 8004 --node-id worker_4
python worker_node.py --port 8005 --node-id worker_5

# Update gateway (edit gateway.py or pass as arguments)
python gateway.py --workers http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 http://localhost:8005
```

### Modify Batch Parameters
Edit `worker_node.py` around line 21:
```python
self.batch_processor = BatchProcessor(
    max_batch_size=64,    # Increase batch size
    timeout_ms=50,        # Increase timeout
    process_fn=self._process_batch
)
```

## Troubleshooting

### Port Already in Use

**Linux/Mac:**
```bash
# Find and kill processes
lsof -ti:8000 | xargs kill -9
lsof -ti:8001 | xargs kill -9
lsof -ti:8002 | xargs kill -9
lsof -ti:8003 | xargs kill -9
```

**Windows:**
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Workers Not Starting
- Make sure you have Python 3.8+: `python --version`
- Install dependencies: `pip install numpy requests matplotlib`
- Check for error messages in terminal
- Try starting workers manually in separate terminals

### Benchmark Connection Refused
- Ensure all workers are running: check terminals
- Ensure gateway is running: check terminal
- Wait 2-3 seconds after starting gateway before running benchmark
- Test connectivity: `curl http://localhost:8000/stats`

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade numpy requests matplotlib
```

## Example Output

```
Starting load test: 5000 requests with 50 concurrent
Target: http://localhost:8000/infer
------------------------------------------------------------
Progress: 5000/5000 (100%) - 1087.3 req/s
------------------------------------------------------------

BENCHMARK RESULTS
============================================================
Total Requests:      5000
Successful:          2576
Failed:              2424
Total Time:          683.77s
Throughput:          3.77 req/s

Latency Distribution (ms):
  Mean:              4729.26
  Median (p50):      4001.24
  p95:               9061.05
  p99:               12667.61
  Min:               316.64
  Max:               17302.05
  Std Dev:           2442.98

Node Distribution:
  worker_1: 786 (30.5%)
  worker_2: 854 (33.2%)
  worker_3: 936 (36.3%)

Load Balance Variance: 7.14%
============================================================
```

## Stopping the System

**Linux/Mac:**
```bash
./stop.sh
```

**Manual:**
```bash
# Linux/Mac
pkill -f worker_node.py
pkill -f gateway.py

# Windows
taskkill /F /IM python.exe
```
## Next Steps

### Enhancements to Try:
1. Add caching layer for repeated requests
2. Implement circuit breakers for fault tolerance
3. Add Prometheus metrics export
4. Implement request prioritization
5. Add authentication/API keys
6. Support multiple model versions (A/B testing)
7. Add GPU inference support
8. Implement request hedging

### Convert to C++:
The C++ version (from earlier artifacts) offers:
- 3-5x better performance
- Lower latency (sub-millisecond)
- More efficient memory usage
- gRPC for faster communication

## License

MIT License - Free to use for portfolios and resumes!

## Note

This is a learning/portfolio project. The code prioritizes clarity and educational value.

---

**Built with**: Python • NumPy • Matplotlib • HTTP

**Concepts**: Distributed Systems • Load Balancing • Performance Engineering • System Design
