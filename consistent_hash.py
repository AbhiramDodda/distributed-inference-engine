import hashlib
from bisect import bisect_right

class ConsistentHash:
    def __init__(self, nodes=None, virtual_nodes=150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}  # hash -> node mapping
        self.sorted_keys = []  # sorted hash keys for binary search
        self.nodes = set()
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def _hash(self, key):
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}#{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node):
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        
        # Remove all virtual nodes
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}#{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key):
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        index = bisect_right(self.sorted_keys, hash_value)
        if index == len(self.sorted_keys):
            index = 0
        
        return self.ring[self.sorted_keys[index]]
    
    def get_nodes(self):
        return list(self.nodes)
    
    def get_distribution(self, keys):
        distribution = {node: 0 for node in self.nodes}
        for key in keys:
            node = self.get_node(key)
            if node:
                distribution[node] += 1
        
        return distribution
    
    def get_load_balance_variance(self, num_keys=10000):
        if not self.nodes:
            return 0.0
        
        keys = [f"key_{i}" for i in range(num_keys)]
        distribution = self.get_distribution(keys)
        
        counts = list(distribution.values())
        if not counts:
            return 0.0
        
        mean = sum(counts) / len(counts)
        variance = sum((x - mean) ** 2 for x in counts) / len(counts)
        std_dev = variance ** 0.5
        
        # Return as percentage of mean
        return (std_dev / mean * 100) if mean > 0 else 0.0


if __name__ == "__main__":
    nodes = ['localhost:8001', 'localhost:8002', 'localhost:8003']
    ch = ConsistentHash(nodes)
    test_keys = [f"req_{i}" for i in range(10000)]
    distribution = ch.get_distribution(test_keys)
    
    print("Consistent Hashing Test")
    print("=" * 50)
    print(f"Nodes: {nodes}")
    print(f"Virtual nodes per physical node: {ch.virtual_nodes}")
    print(f"\nDistribution of {len(test_keys)} keys:")
    for node, count in sorted(distribution.items()):
        percentage = (count / len(test_keys)) * 100
        print(f"  {node}: {count} ({percentage:.2f}%)")
    
    variance = ch.get_load_balance_variance(10000)
    print(f"\nLoad balance variance: {variance:.2f}%")
    print("=" * 50)