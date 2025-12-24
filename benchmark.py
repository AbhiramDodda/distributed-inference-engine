import argparse
import json
import time
import numpy as np
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


class LoadGenerator:
    def __init__(self, target_url, num_requests, concurrent):
        self.target_url = target_url
        self.num_requests = num_requests
        self.concurrent = concurrent
        self.latencies = []
        self.errors = 0
        self.node_distribution = defaultdict(int)
    
    def generate_request_data(self, req_id):
        # image input(a random matrix) is used for simulation
        input_size = 224 * 224 * 3
        input_data = np.random.rand(input_size).tolist()
        return {
            'request_id': f'req_{req_id}',
            'model_name': 'resnet50',
            'input_data': input_data,
            'input_shape': [1, 224, 224, 3],
            'timestamp': int(time.time() * 1_000_000)
        }
    
    def send_request(self, req_id):
        request_data = self.generate_request_data(req_id)
        start_time = time.time()
        try:
            req = urllib.request.Request(
                f"{self.target_url}/infer",
                data=json.dumps(request_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read())
                
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'latency': latency_ms,
                'node_id': result.get('node_id', 'unknown'),
                'inference_time': result.get('inference_time_us', 0) / 1000.0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run(self):
        print(f"Starting load test: {self.num_requests} requests with {self.concurrent} concurrent")
        print(f"Target: {self.target_url}/infer")
        print("-" * 60)
        start_time = time.time()
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.concurrent) as executor:
            futures = [executor.submit(self.send_request, i) 
                      for i in range(self.num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                
                if result['success']:
                    self.latencies.append(result['latency'])
                    self.node_distribution[result['node_id']] += 1
                else:
                    self.errors += 1
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    throughput = completed / elapsed
                    progress_pct = (completed * 100) // self.num_requests
                    print(f"Progress: {completed}/{self.num_requests} "
                          f"({progress_pct}%) - {throughput:.1f} req/s", 
                          end='\r')
        
        total_time = time.time() - start_time
        print("\n" + "-" * 60)
        
        return self.analyze_results(total_time)
    
    def analyze_results(self, total_time):
        if not self.latencies:
            print("ERROR: No successful requests!")
            return None
        latencies = np.array(self.latencies)
        
        results = {
            'total_requests': self.num_requests,
            'successful_requests': len(self.latencies),
            'failed_requests': self.errors,
            'total_time': total_time,
            'throughput': len(self.latencies) / total_time,
            'latency': {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'std': float(np.std(latencies))
            },
            'node_distribution': dict(self.node_distribution)
        }
        
        print("\nBENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Requests:      {results['total_requests']}")
        print(f"Successful:          {results['successful_requests']}")
        print(f"Failed:              {results['failed_requests']}")
        print(f"Total Time:          {results['total_time']:.2f}s")
        print(f"Throughput:          {results['throughput']:.2f} req/s")
        print()
        print("Latency Distribution (ms):")
        print(f"  Mean:              {results['latency']['mean']:.2f}")
        print(f"  Median (p50):      {results['latency']['p50']:.2f}")
        print(f"  p95:               {results['latency']['p95']:.2f}")
        print(f"  p99:               {results['latency']['p99']:.2f}")
        print(f"  Min:               {results['latency']['min']:.2f}")
        print(f"  Max:               {results['latency']['max']:.2f}")
        print(f"  Std Dev:           {results['latency']['std']:.2f}")
        print()
        print("Node Distribution:")
        total_dist = sum(results['node_distribution'].values())
        for node, count in sorted(results['node_distribution'].items()):
            percentage = (count / total_dist) * 100
            print(f"  {node}: {count} ({percentage:.1f}%)")
        
        # Calculate load balance variance
        if len(results['node_distribution']) > 1:
            dist_values = list(results['node_distribution'].values())
            mean_dist = np.mean(dist_values)
            variance = (np.std(dist_values) / mean_dist) * 100
            print(f"\nLoad Balance Variance: {variance:.2f}%")
        print("=" * 60)
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results


def main():
    parser = argparse.ArgumentParser(description='Load generator for distributed inference')
    parser.add_argument('--target', default='http://localhost:8000', 
                       help='Target gateway URL')
    parser.add_argument('--requests', type=int, default=1000, 
                       help='Total number of requests')
    parser.add_argument('--concurrent', type=int, default=50, 
                       help='Concurrent requests')
    
    args = parser.parse_args()
    
    try:
        with urllib.request.urlopen(f"{args.target}/stats", timeout=2) as response:
            stats = json.loads(response.read())
            print(f"Gateway is accessible")
            print(f"Workers: {stats.get('num_workers', 0)}")
    except Exception as e:
        print(f"Cannot connect to gateway: {e}")
        print(f"Make sure gateway is running on {args.target}")
        return
    generator = LoadGenerator(args.target, args.requests, args.concurrent)
    generator.run()

if __name__ == '__main__':
    main()