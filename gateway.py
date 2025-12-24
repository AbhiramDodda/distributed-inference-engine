import argparse
import json
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from consistent_hash import ConsistentHash

class Gateway:
    def __init__(self, workers):
        self.workers = workers
        self.hash_ring = ConsistentHash(workers, virtual_nodes=150)
        self.request_count = 0
        for worker in workers:
            try:
                health_url = f"{worker}/health"
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    data = json.loads(response.read())
                    print(f"{worker} - {data.get('node_id', 'unknown')}")
            except Exception as e:
                print(f"{worker} - Error: {e}")
    
    def route_request(self, request_data):
        self.request_count += 1
        # Getting target node using consistent hashing
        request_id = request_data.get('request_id', f'req_{self.request_count}')
        target_node = self.hash_ring.get_node(request_id)
        
        if not target_node:
            raise Exception("No workers available")
        
        # Forward request to worker
        url = f"{target_node}/infer"
        req = urllib.request.Request(
            url,
            data=json.dumps(request_data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read())
                return result
        except urllib.error.URLError as e:
            # Retry with a different node
            nodes = self.hash_ring.get_nodes()
            for node in nodes:
                if node != target_node:
                    try:
                        retry_url = f"{node}/infer"
                        retry_req = urllib.request.Request(
                            retry_url,
                            data=json.dumps(request_data).encode('utf-8'),
                            headers={'Content-Type': 'application/json'}
                        )
                        with urllib.request.urlopen(retry_req, timeout=10) as response:
                            result = json.loads(response.read())
                            return result
                    except:
                        continue
            
            raise Exception(f"All workers failed: {str(e)}")
    
    def get_stats(self):
        """Get gateway statistics"""
        return {
            'total_requests': self.request_count,
            'num_workers': len(self.workers),
            'workers': self.workers
        }

class GatewayRequestHandler(BaseHTTPRequestHandler):
    gateway = None
    
    def do_POST(self):
        if self.path == '/infer':
            try:
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                request_data = json.loads(body.decode('utf-8'))
                response = self.gateway.route_request(request_data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Error: {str(e)}")
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
        if self.path == '/stats':
            try:
                stats = self.gateway.get_stats()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Error: {str(e)}")
        else:
            self.send_error(404, "Not Found")
    
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description='Gateway Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--workers', nargs='+', 
                       default=['http://localhost:8001', 'http://localhost:8002', 'http://localhost:8003'],
                       help='Worker addresses')
    
    args = parser.parse_args()
    gateway = Gateway(args.workers)
    # Set gateway instance in handler class
    GatewayRequestHandler.gateway = gateway

    server = HTTPServer(('localhost', args.port), GatewayRequestHandler)
    
    print()
    print("=" * 60)
    print("GATEWAY SERVER")
    print("=" * 60)
    print(f"   Listening on: http://localhost:{args.port}")
    print(f"   Workers: {len(args.workers)}")
    for i, worker in enumerate(args.workers, 1):
        print(f"{i}. {worker}")
    print(f"Routing: Consistent Hashing (150 virtual nodes)")
    print(f"Ready to route requests!")
    print("=" * 60)
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping gateway...")
        server.shutdown()

if __name__ == '__main__':
    main()