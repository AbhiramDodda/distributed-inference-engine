import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from inference_engine import InferenceEngine
from batch_processor import BatchProcessor

class WorkerNode:
    """Worker node with inference engine and batch processor"""
    
    def __init__(self, node_id, port):
        self.node_id = node_id
        self.port = port
        self.engine = InferenceEngine(model_name="resnet50", shard_id=port % 3)
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            max_batch_size=32,
            timeout_ms=20,
            process_fn=self._process_batch
        )
        self.batch_processor.start()
        self.total_requests = 0
        self.active_requests = 0
    
    def _process_batch(self, requests):
        inputs = [req['input_data'] for req in requests]
        shapes = [req['input_shape'] for req in requests]
        batch_results = self.engine.batch_predict(inputs, shapes)
        responses = []
        for i, (output, inference_time) in enumerate(batch_results):
            response = {
                'request_id': requests[i]['request_id'],
                'output_data': output,
                'output_shape': [len(output)],
                'inference_time_us': inference_time,
                'node_id': self.node_id
            }
            responses.append(response)
        
        return responses
    
    def handle_infer(self, request_data):
        self.total_requests += 1
        self.active_requests += 1
        try:
            response = self.batch_processor.process(request_data)
            return response
        finally:
            self.active_requests -= 1
    
    def handle_health(self):
        metrics = self.batch_processor.get_metrics()
        return {
            'healthy': True,
            'node_id': self.node_id,
            'active_requests': self.active_requests,
            'total_requests': self.total_requests,
            'batch_metrics': {
                'total_batches': metrics.total_batches,
                'avg_batch_size': metrics.avg_batch_size,
                'timeout_batches': metrics.timeout_batches,
                'full_batches': metrics.full_batches
            }
        }

class WorkerRequestHandler(BaseHTTPRequestHandler):
    worker = None  
    def do_POST(self):
        if self.path == '/infer':
            try:
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                request_data = json.loads(body.decode('utf-8'))
                response = self.worker.handle_infer(request_data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_error(500, f"Error: {str(e)}")
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
        if self.path == '/health':
            try:
                response = self.worker.handle_health()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Error: {str(e)}")
        else:
            self.send_error(404, "Not Found")
    
    def log_message(self, format, *args):
        pass

def main():
    parser = argparse.ArgumentParser(description='Worker Node Server')
    parser.add_argument('--port', type=int, default=8001, help='Port to listen on')
    parser.add_argument('--node-id', type=str, default=None, help='Node ID')
    args = parser.parse_args()
    node_id = args.node_id or f"worker_{args.port}"
    worker = WorkerNode(node_id, args.port)
    WorkerRequestHandler.worker = worker
    server = HTTPServer(('localhost', args.port), WorkerRequestHandler)
    
    print(f"Worker Node: {node_id}")
    print(f"Port: {args.port}")
    print(f"Model: {worker.engine.model_name}")
    print(f"Shard: {worker.engine.shard_id}")
    print(f"Batch size: {worker.batch_processor.max_batch_size}")
    print(f"Batch timeout: {worker.batch_processor.timeout_ms*1000:.0f}ms")
    print(f"Ready to accept requests!")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopping {node_id}...")
        worker.batch_processor.stop()
        server.shutdown()

if __name__ == '__main__':
    main()
