echo "================================================"
echo "Starting Distributed Inference System"
echo "================================================"
echo ""

# Kill existing processes
echo "Cleaning up old processes..."
pkill -f worker_node.py 2>/dev/null
pkill -f gateway.py 2>/dev/null
sleep 1

# Start worker nodes
echo "Starting worker nodes..."
python3 worker_node.py --port 8001 --node-id worker_1 > worker1.log 2>&1 &
WORKER1_PID=$!
echo "Worker 1 (PID: $WORKER1_PID) - Port 8001"
python3 worker_node.py --port 8002 --node-id worker_2 > worker2.log 2>&1 &
WORKER2_PID=$!
echo "Worker 2 (PID: $WORKER2_PID) - Port 8002"
python3 worker_node.py --port 8003 --node-id worker_3 > worker3.log 2>&1 &
WORKER3_PID=$!
echo "Worker 3 (PID: $WORKER3_PID) - Port 8003"
echo ""
echo "Waiting for workers to initialize..."
sleep 3

# Start gateway
echo ""
echo "Starting gateway..."
python3 gateway.py --port 8000 > gateway.log 2>&1 &
GATEWAY_PID=$!
echo "Gateway (PID: $GATEWAY_PID) - Port 8000"

# Wait for gateway to start
echo ""
echo "Waiting for gateway..."
sleep 2

# Save PIDs
echo "$WORKER1_PID $WORKER2_PID $WORKER3_PID $GATEWAY_PID" > .system_pids
echo ""
echo "================================================"
echo "System is running!"
echo "================================================"
echo ""
echo "Components:"
echo "  Gateway:  http://localhost:8000"
echo "  Worker 1: http://localhost:8001"
echo "  Worker 2: http://localhost:8002"
echo "  Worker 3: http://localhost:8003"
echo ""
echo "Logs:"
echo "  gateway.log, worker1.log, worker2.log, worker3.log"
echo ""
echo "Next steps:"
echo "  1. Run benchmark:"
echo "     python3 benchmark.py --requests 5000 --concurrent 50"
echo ""
echo "  2. Analyze results:"
echo "     python3 analyze_results.py"
echo ""
echo "  3. Stop system:"
echo "     ./stop_system.sh"
echo ""
echo "================================================"