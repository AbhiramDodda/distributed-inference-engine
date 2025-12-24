import time
import numpy as np
from typing import List, Tuple


class InferenceEngine:
    def __init__(self, model_name="resnet50", shard_id=0):
        self.model_name = model_name
        self.shard_id = shard_id
        self.num_classes = 1000
        # Increased weight size to create real CPU load
        self.hidden_size = 1024 
        np.random.seed(42 + shard_id)
        self.weights = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
    
    def predict(self, input_data: List[float], input_shape: List[int]) -> Tuple[List[float], int]:
        """Performs a real computation instead of sleeping."""
        start_time = time.perf_counter()
        
        # 1. Transform input to match hidden size
        x = np.array(input_data, dtype=np.float32)
        # Use only enough data to fill our matrix rows
        x = x[:self.hidden_size] if x.size >= self.hidden_size else np.pad(x, (0, self.hidden_size - x.size))
        x = x.reshape(1, self.hidden_size)

        # 2. Simulate deep layer processing with actual MatMul
        # This will naturally take time based on your CPU speed
        for _ in range(5):
            x = np.matmul(x, self.weights)
            x = np.tanh(x) # Add non-linearity to keep CPU busy

        # 3. Project to classes
        output = np.abs(x[0, :self.num_classes])
        output = (output / output.sum()).tolist()
        
        inference_time_us = int((time.perf_counter() - start_time) * 1_000_000)
        return output, inference_time_us
    
    def batch_predict(self, inputs: List[List[float]], shapes: List[List[int]]) -> List[Tuple[List[float], int]]:
        """Uses vectorized batch processing for efficiency."""
        if not inputs: return []
        start_time = time.perf_counter()
        batch_size = len(inputs)
        
        # 1. Vectorized Batching: Convert list of lists to a single large matrix
        # This is where the 'distributed' efficiency actually comes from
        batch_array = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        for i, inp in enumerate(inputs):
            arr = np.array(inp, dtype=np.float32)[:self.hidden_size]
            batch_array[i, :len(arr)] = arr

        # 2. Single large MatMul (much faster than looping over predict())
        x = batch_array
        for _ in range(5):
            x = np.matmul(x, self.weights)
            x = np.tanh(x)

        total_time_us = int((time.perf_counter() - start_time) * 1_000_000)
        per_item_time = total_time_us // batch_size
        
        results = []
        for i in range(batch_size):
            out = np.abs(x[i, :self.num_classes])
            results.append(((out / out.sum()).tolist(), per_item_time))
            
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': self.model_name,
            'shard_id': self.shard_id,
            'num_classes': self.num_classes,
            'weights_size_mb': self.weights.nbytes / (1024 * 1024)
        }
    
if __name__ == "__main__":
    # Test inference engine
    print("Testing Inference Engine")
    print("=" * 50)
    
    engine = InferenceEngine(model_name="resnet50", shard_id=0)
    
    # Model info
    info = engine.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Shard: {info['shard_id']}")
    print(f"Classes: {info['num_classes']}")
    print(f"Weights: {info['weights_size_mb']:.2f} MB")
    print()
    
    # Test single inference
    print("Single Inference Test:")
    input_data = np.random.rand(224 * 224 * 3).tolist()
    input_shape = [1, 224, 224, 3]
    
    output, inference_time = engine.predict(input_data, input_shape)
    print(f"  Input size: {len(input_data)}")
    print(f"  Output size: {len(output)}")
    print(f"  Inference time: {inference_time/1000:.2f} ms")
    print(f"  Top-5 classes: {sorted(output, reverse=True)[:5]}")
    print()
    
    # Test batch inference
    print("Batch Inference Test (batch_size=8):")
    batch_inputs = [np.random.rand(224 * 224 * 3).tolist() for _ in range(8)]
    batch_shapes = [[1, 224, 224, 3]] * 8
    
    start = time.time()
    batch_results = engine.batch_predict(batch_inputs, batch_shapes)
    batch_time = (time.time() - start) * 1000
    
    print(f"  Batch size: {len(batch_inputs)}")
    print(f"  Total time: {batch_time:.2f} ms")
    print(f"  Per-item time: {batch_time/len(batch_inputs):.2f} ms")
    print(f"  Speedup vs single: {(inference_time/1000) / (batch_time/len(batch_inputs)):.2f}x")
    
    print("=" * 50)