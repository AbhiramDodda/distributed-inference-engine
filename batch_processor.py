import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Callable, Any, List

@dataclass
class BatchMetrics:
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    timeout_batches: int = 0
    full_batches: int = 0

class BatchProcessor:
    def __init__(self, max_batch_size=32, timeout_ms=20, 
                 process_fn: Callable[[List[Any]], List[Any]] = None):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds
        self.process_fn = process_fn
        self.request_queue = Queue()
        self.metrics = BatchMetrics()
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
    
    def process(self, request):
        result_queue = Queue(maxsize=1)
        self.request_queue.put((request, result_queue))
        with self.lock:
            self.metrics.total_requests += 1
        
        # Wait for result
        try:
            result = result_queue.get(timeout=10.0)
            return result
        except Empty:
            raise TimeoutError("Request processing timeout")
    
    def _processing_loop(self):
        batch = []
        result_queues = []
        last_batch_time = time.perf_counter()
        
        while self.running:
            try:
                # Calculate how much time is left in the current batch window
                elapsed = time.perf_counter() - last_batch_time
                timeout_remaining = max(0, self.timeout_ms - elapsed)
                # Wait for next request
                request, result_queue = self.request_queue.get(timeout=timeout_remaining)
                batch.append(request)
                result_queues.append(result_queue)
                
                # Trigger batch if full
                if len(batch) >= self.max_batch_size:
                    self._process_batch(batch, result_queues, timeout=False)
                    batch, result_queues = [], []
                    last_batch_time = time.perf_counter()
                    
            except Empty:
                # Trigger batch if timeout reached
                if batch:
                    self._process_batch(batch, result_queues, timeout=True)
                    batch, result_queues = [], []
                last_batch_time = time.perf_counter()
    
    def _process_batch(self, batch, result_queues, timeout=False):
        """Process a batch of requests"""
        if not batch:
            return
        
        try:
            # Process the batch
            if self.process_fn:
                results = self.process_fn(batch)
            else:
                # Default: return requests as-is
                results = batch
            
            # Send results back
            for result, result_queue in zip(results, result_queues):
                result_queue.put(result)
            with self.lock:
                self.metrics.total_batches += 1
                batch_size = len(batch)
                
                # Update average batch size
                prev_total = self.metrics.avg_batch_size * (self.metrics.total_batches - 1)
                self.metrics.avg_batch_size = (prev_total + batch_size) / self.metrics.total_batches
                
                if timeout:
                    self.metrics.timeout_batches += 1
                else:
                    self.metrics.full_batches += 1
                    
        except Exception as e:
            for result_queue in result_queues:
                result_queue.put({"error": str(e)})
    
    def get_metrics(self):
        """Get current batch processing metrics"""
        with self.lock:
            return BatchMetrics(
                total_requests=self.metrics.total_requests,
                total_batches=self.metrics.total_batches,
                avg_batch_size=self.metrics.avg_batch_size,
                timeout_batches=self.metrics.timeout_batches,
                full_batches=self.metrics.full_batches
            )


if __name__ == "__main__":
    import random
    def process_batch(batch):
        time.sleep(0.01)  # Simulate computation
        return [f"result_{req}" for req in batch]
    
    processor = BatchProcessor(max_batch_size=10, timeout_ms=50, process_fn=process_batch)
    processor.start()
    
    print("Testing Batch Processor")
    print("=" * 50)
    results = []
    start_time = time.time()
    
    for i in range(50):
        result = processor.process(f"req_{i}")
        results.append(result)
        time.sleep(random.uniform(0.001, 0.005))
    
    elapsed = time.time() - start_time
    metrics = processor.get_metrics()
    print(f"Processed {len(results)} requests in {elapsed:.2f}s")
    print(f"Total batches: {metrics.total_batches}")
    print(f"Average batch size: {metrics.avg_batch_size:.2f}")
    print(f"Timeout batches: {metrics.timeout_batches}")
    print(f"Full batches: {metrics.full_batches}")
    print(f"Throughput: {len(results)/elapsed:.2f} req/s")
    processor.stop()
    print("=" * 50)