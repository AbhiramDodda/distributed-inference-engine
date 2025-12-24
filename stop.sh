echo "================================================"
echo "Stopping Distributed Inference System"
echo "================================================"
echo ""

# Kill processes by PID
if [ -f .system_pids ]; then
    echo "Stopping processes from PID file..."
    read -r PIDS < .system_pids
    for pid in $PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null
            echo "  ✓ Stopped process $pid"
        fi
    done
    rm .system_pids
fi

# Kill any remaining processes by name
echo ""
echo "Cleaning up any remaining processes..."
pkill -f worker_node.py 2>/dev/null && echo "  ✓ Stopped worker nodes"
pkill -f gateway.py 2>/dev/null && echo "  ✓ Stopped gateway"

sleep 1

echo ""
echo "End"