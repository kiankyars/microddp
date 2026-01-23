"""
Performance Analysis for Distributed Data Parallelism

This module provides tools to analyze:
- Communication overhead
- Scaling efficiency
- Bottlenecks in DDP training
"""

import time
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple
from comms import DataParallelComms


class DDPPerformanceAnalyzer:
    """
    Analyzes performance characteristics of DDP training.
    """
    
    def __init__(self, comms: DataParallelComms, device):
        self.comms = comms
        self.device = device
        self.metrics = {
            'forward_times': [],
            'backward_times': [],
            'comm_times': [],
            'optimizer_times': [],
            'total_times': []
        }
    
    def time_step(self, model, input_chunk, target_chunk, optimizer, 
                  use_bucketing=False):
        """
        Time a single training step and break down the components.
        
        Returns:
            Dict with timing breakdown
        """
        # Forward pass
        self.comms.barrier()
        start = time.time()
        loss = model(input_chunk, target_chunk)
        self.comms.barrier()
        forward_time = time.time() - start
        
        # Backward pass
        self.comms.barrier()
        start = time.time()
        loss.backward()
        self.comms.barrier()
        backward_time = time.time() - start
        
        # Communication (if not using hooks/bucketing)
        self.comms.barrier()
        start = time.time()
        if not use_bucketing:
            for param in model.parameters():
                if param.grad is not None:
                    self.comms.all_reduce_mean(param.grad)
        self.comms.barrier()
        comm_time = time.time() - start
        
        # Optimizer step
        self.comms.barrier()
        start = time.time()
        optimizer.step()
        self.comms.barrier()
        optimizer_time = time.time() - start
        
        total_time = forward_time + backward_time + comm_time + optimizer_time
        
        timing = {
            'forward': forward_time,
            'backward': backward_time,
            'communication': comm_time,
            'optimizer': optimizer_time,
            'total': total_time
        }
        
        # Store metrics
        self.metrics['forward_times'].append(forward_time)
        self.metrics['backward_times'].append(backward_time)
        self.metrics['comm_times'].append(comm_time)
        self.metrics['optimizer_times'].append(optimizer_time)
        self.metrics['total_times'].append(total_time)
        
        return timing
    
    def analyze_communication_overhead(self, num_steps=10):
        """
        Analyze what fraction of time is spent on communication.
        """
        if not self.metrics['total_times']:
            return None
        
        avg_comm = sum(self.metrics['comm_times']) / len(self.metrics['comm_times'])
        avg_total = sum(self.metrics['total_times']) / len(self.metrics['total_times'])
        comm_overhead = (avg_comm / avg_total) * 100
        
        return {
            'avg_communication_time': avg_comm,
            'avg_total_time': avg_total,
            'communication_overhead_percent': comm_overhead
        }
    
    def get_summary(self):
        """Get summary statistics of all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
                }
        return summary


def analyze_scaling_efficiency(model_fn, data_fn, world_sizes: List[int], 
                               num_steps: int = 10):
    """
    Analyze how DDP scales with different world sizes.
    
    Measures:
    - Speedup vs single GPU
    - Efficiency (actual speedup / ideal speedup)
    - Communication overhead as function of world size
    """
    results = {}
    
    # Single GPU baseline
    if 1 in world_sizes:
        # Run single GPU baseline
        pass  # Would need to implement single GPU timing
    
    for world_size in world_sizes:
        # This would need to be run with different torchrun configurations
        # For now, return structure
        results[world_size] = {
            'time_per_step': None,
            'speedup': None,
            'efficiency': None
        }
    
    return results


def estimate_communication_time(tensor_size_bytes: int, world_size: int,
                                bandwidth_gbps: float = 10.0, latency_ms: float = 0.1):
    """
    Estimate communication time for all-reduce operation.
    
    Uses simplified model:
    - Ring all-reduce: 2*(n-1) messages
    - Each message: latency + size/bandwidth
    
    Args:
        tensor_size_bytes: Size of tensor to all-reduce
        world_size: Number of ranks
        bandwidth_gbps: Network bandwidth in Gbps
        latency_ms: Network latency in milliseconds
    
    Returns:
        Estimated communication time in seconds
    """
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    
    # Ring all-reduce: 2*(n-1) steps
    num_steps = 2 * (world_size - 1)
    
    # Each step: send and receive
    # Chunk size per step (simplified - actual ring splits data)
    chunk_size = tensor_size_bytes / world_size
    
    # Time per step: latency + transfer time
    time_per_step = (latency_ms / 1000) + (chunk_size / bandwidth_bytes_per_sec)
    
    total_time = num_steps * time_per_step
    
    return total_time


def communication_complexity_analysis():
    """
    Demonstrate communication complexity of different approaches.
    
    Shows why ring all-reduce is O(n) while naive is O(n²).
    """
    print("=== Communication Complexity Analysis ===\n")
    
    world_sizes = [2, 4, 8, 16, 32]
    tensor_size_mb = 100  # 100 MB tensor
    
    print(f"Tensor size: {tensor_size_mb} MB\n")
    print("World Size | Naive O(n²) | Ring O(n) | Speedup")
    print("-" * 50)
    
    for n in world_sizes:
        # Naive: each rank sends to rank 0, then broadcast
        # Messages: (n-1) sends + (n-1) broadcasts = 2*(n-1) but all ranks involved
        naive_messages = 2 * (n - 1) * n  # Simplified O(n²)
        
        # Ring: 2*(n-1) steps, each step is one send/receive per rank
        ring_messages = 2 * (n - 1)
        
        speedup = naive_messages / ring_messages if ring_messages > 0 else 0
        
        print(f"{n:10d} | {naive_messages:11d} | {ring_messages:8d} | {speedup:.2f}x")
    
    print("\nKey Insight: Ring all-reduce scales linearly with world size,")
    print("while naive approaches scale quadratically!")


def when_ddp_breaks_down():
    """
    Analyze scenarios where DDP becomes inefficient.
    
    DDP breaks down when:
    1. Model is too small (communication overhead > computation)
    2. Network is too slow (high latency, low bandwidth)
    3. Batch size per rank is too small (can't hide communication)
    """
    print("=== When DDP Breaks Down ===\n")
    
    scenarios = [
        {
            'name': 'Small Model',
            'model_size_mb': 1,
            'computation_time_ms': 1,
            'world_size': 4,
            'bandwidth_gbps': 10
        },
        {
            'name': 'Large Model',
            'model_size_mb': 1000,
            'computation_time_ms': 100,
            'world_size': 4,
            'bandwidth_gbps': 10
        },
        {
            'name': 'Slow Network',
            'model_size_mb': 100,
            'computation_time_ms': 50,
            'world_size': 4,
            'bandwidth_gbps': 1  # Slow network
        }
    ]
    
    for scenario in scenarios:
        comm_time = estimate_communication_time(
            scenario['model_size_mb'] * 1024 * 1024,
            scenario['world_size'],
            scenario['bandwidth_gbps']
        ) * 1000  # Convert to ms
        
        comp_time = scenario['computation_time_ms']
        overhead = (comm_time / comp_time) * 100
        
        print(f"{scenario['name']}:")
        print(f"  Computation: {comp_time:.1f} ms")
        print(f"  Communication: {comm_time:.1f} ms")
        print(f"  Overhead: {overhead:.1f}%")
        print(f"  {'Efficient' if overhead < 50 else 'Inefficient'}")
        print()
