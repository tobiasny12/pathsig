import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime

try:
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import keras_sig
except ImportError:
    raise ImportError(
        "This benchmarks requires keras_sig for comparison. "
        "Please install keras_sig to run performance comparisons."
    )

from pathsig import signature, Signature, sig_size

# Configurations for matplotlib
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def set_keras_dtype(dtype: torch.dtype):
    """Configure Keras to use the specified float type."""
    dtype_str = "float64" if dtype == torch.float64 else "float32"
    try:
        if hasattr(keras.config, "set_floatx"):
            keras.config.set_floatx(dtype_str)
        elif hasattr(keras.backend, "set_floatx"):
            keras.backend.set_floatx(dtype_str)
    except Exception:
        pass

def format_time(time_ms):
    """Format time in appropriate units (ns, μs, ms, or s)."""
    if isinstance(time_ms, str):
        return time_ms
    if time_ms is None or pd.isna(time_ms):
        return "—"
    if time_ms < 0.001:
        return f"{time_ms * 1000000:.2f} ns"
    elif time_ms < 1:
        return f"{time_ms * 1000:.2f} μs"
    elif time_ms < 1000:
        return f"{time_ms:.2f} ms"
    else:
        return f"{time_ms / 1000:.2f} s"


def format_signature_dim(dim):
    """Format signature dimension with K/M notation."""
    if dim is None or pd.isna(dim):
        return "—"
    if dim < 1000:
        return str(int(dim))
    elif dim < 1000000:
        return f"{dim/1000:.1f}K"
    else:
        return f"{dim/1000000:.2f}M"


def format_memory(mb):
    """Format memory in MB or GB."""
    if isinstance(mb, str):
        return mb
    if mb is None or pd.isna(mb):
        return "—"
    elif mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb/1024:.2f} GB"


def format_speedup(speedup):
    """Format speedup factor."""
    if speedup is None or pd.isna(speedup):
        return "—"
    return f"{speedup:.2f}×"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmarks run."""
    batch_size: int
    sequence_length: int
    path_dim: int
    truncation_level: int

    @property
    def signature_dim(self) -> int:
        """Calculate output dimension of signature."""
        return sig_size(self.path_dim, self.truncation_level)

    @property
    def memory_estimate_mb(self) -> float:
        """Estimate memory footprint in MB."""
        # Input + output tensors in float32
        input_size = self.batch_size * self.sequence_length * self.path_dim * 4
        output_size = self.batch_size * self.signature_dim * 4
        return (input_size + output_size) / (1024 * 1024)


class SignatureBenchmark:
    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 50, dtype: torch.dtype = torch.float32):
        """
        Initialize benchmarks framework.
        Args:
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of measurement iterations
            dtype: Data type for benchmarking
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.dtype = dtype
        self.results = []

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for benchmarking")

        # Print minimal system info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        print(f"Data type: {dtype}")
        print("-" * 60)

    def benchmark_forward_only(self, layer: nn.Module, input_tensor: torch.Tensor) -> Dict:
        """Benchmark forward pass only (forward mode)."""
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = layer(input_tensor)
                torch.cuda.synchronize()

        # Benchmark
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1e3
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                _ = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1e3
        mean_time = (end_time - start_time) / self.benchmark_runs
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

        return {
            'mean': mean_time,
            'memory_mb': memory_mb
        }

    def benchmark_forward_backward(self, layer: nn.Module, input_tensor: torch.Tensor) -> Dict:
        """Benchmark forward and backward passes (training mode)."""
        # Warmup
        for _ in range(self.warmup_runs):
            input_tensor.grad = None
            output = layer(input_tensor)
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()

        # Pre-allocate CUDA events
        f_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_runs)]
        f_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_runs)]
        b_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_runs)]
        b_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_runs)]

        # Benchmark
        torch.cuda.reset_peak_memory_stats()
        for i in range(self.benchmark_runs):
            input_tensor.grad = None
            f_start_events[i].record()
            output = layer(input_tensor)
            f_end_events[i].record()
            loss = output.sum()
            b_start_events[i].record()
            loss.backward()
            b_end_events[i].record()
        torch.cuda.synchronize()

        forward_times = [start.elapsed_time(end) for start, end in zip(f_start_events, f_end_events)]
        backward_times = [start.elapsed_time(end) for start, end in zip(b_start_events, b_end_events)]
        total_times = [f + b for f, b in zip(forward_times, backward_times)]

        return {
            'forward_mean': np.mean(forward_times),
            'backward_mean': np.mean(backward_times),
            'total_mean': np.mean(total_times),
            'memory_mb': torch.cuda.max_memory_allocated() / (1024**2)
        }

    def run_benchmark(self, config: BenchmarkConfig) -> Dict:
        """Run complete benchmarks for a configuration."""
        print(f"Testing: B={config.batch_size}, M={config.sequence_length}, "
              f"d={config.path_dim}, trunc={config.truncation_level}, "
              f"sig_dim={format_signature_dim(config.signature_dim)}")

        # Clean slate at start of each config
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        # Create input tensors
        input_forward = torch.randn(
            config.batch_size, config.sequence_length, config.path_dim,
            device='cuda', dtype=self.dtype, requires_grad=False
        )
        input_training = torch.randn(
            config.batch_size, config.sequence_length, config.path_dim,
            device='cuda', dtype=self.dtype, requires_grad=True
        )

        results = {
            'batch_size': config.batch_size,
            'sequence_length': config.sequence_length,
            'path_dim': config.path_dim,
            'truncation_level': config.truncation_level,
            'signature_dim': config.signature_dim,
            'memory_estimate': config.memory_estimate_mb
        }

        # Benchmark Keras implementation
        try:
            keras_layer = keras_sig.SigLayer(
                depth=config.truncation_level,
                stream=False,
                gpu_optimized=True
            )

            # forward benchmarks
            keras_forward = self.benchmark_forward_only(keras_layer, input_forward)
            results['keras_forward_ms'] = keras_forward['mean']
            results['keras_forward_memory'] = keras_forward['memory_mb']

            # Training benchmarks
            keras_training = self.benchmark_forward_backward(keras_layer, input_training)
            results['keras_training_forward_ms'] = keras_training['forward_mean']
            results['keras_backward_ms'] = keras_training['backward_mean']
            results['keras_total_ms'] = keras_training['total_mean']
            results['keras_training_memory'] = keras_training['memory_mb']

        except Exception as e:
            print(f" Keras failed: {e}")
            is_oom = 'out of memory' in str(e).lower()
            failed_value = 'OOM' if is_oom else 'ERR'
            results.update({
                'keras_forward_ms': failed_value,
                'keras_training_forward_ms': failed_value,
                'keras_backward_ms': None,
                'keras_total_ms': None,
                'keras_forward_memory': failed_value,
                'keras_training_memory': None
            })

        finally:
            # Cleanup
            if 'keras_layer' in locals():
                del keras_layer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

        # Benchmark pathsig implementation
        try:
            pathsig_layer = Signature(truncation_level=config.truncation_level)

            # forward benchmarks
            pathsig_forward = self.benchmark_forward_only(pathsig_layer, input_forward)
            results['pathsig_forward_ms'] = pathsig_forward['mean']
            results['pathsig_forward_memory'] = pathsig_forward['memory_mb']

            # Training benchmarks
            pathsig_training = self.benchmark_forward_backward(pathsig_layer, input_training)
            results['pathsig_training_forward_ms'] = pathsig_training['forward_mean']
            results['pathsig_backward_ms'] = pathsig_training['backward_mean']
            results['pathsig_total_ms'] = pathsig_training['total_mean']
            results['pathsig_training_memory'] = pathsig_training['memory_mb']

        except Exception as e:
            print(f" pathsig failed: {e}")
            is_oom = 'out of memory' in str(e).lower()
            failed_value = 'OOM' if is_oom else 'ERR'
            results.update({
                'pathsig_forward_ms': failed_value,
                'pathsig_training_forward_ms': failed_value,
                'pathsig_backward_ms': failed_value,
                'pathsig_total_ms': failed_value,
                'pathsig_forward_memory': None,
                'pathsig_training_memory': None
            })

        finally:
            # Cleanup
            if 'pathsig_layer' in locals():
                del pathsig_layer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

        # Calculate speedups
        keras_forward = results.get('keras_forward_ms')
        pathsig_forward = results.get('pathsig_forward_ms')
        if isinstance(keras_forward, (int, float)) and isinstance(pathsig_forward, (int, float)):
            results['forward_speedup'] = keras_forward / pathsig_forward
        else:
            results['forward_speedup'] = None

        keras_total = results.get('keras_total_ms')
        pathsig_total = results.get('pathsig_total_ms')
        if isinstance(keras_total, (int, float)) and isinstance(pathsig_total, (int, float)):
            results['training_speedup'] = keras_total / pathsig_total
        else:
            results['training_speedup'] = None

        return results

    def run_suite(self, configs: List[BenchmarkConfig]) -> pd.DataFrame:
        """Run benchmarks for all configurations."""
        print(f"Running {len(configs)} configurations...")
        print("=" * 60)

        self.results = []

        for config in configs:
            result = self.run_benchmark(config)
            self.results.append(result)

        return pd.DataFrame(self.results)

    def generate_markdown_tables(self, df: pd.DataFrame, output_dir: str = './tables'):
        """Generate Markdown tables for README inclusion with formatted values."""
        os.makedirs(output_dir, exist_ok=True)

        # Create a combined config column for all tables
        df['config'] = df.apply(lambda row: f"({int(row['batch_size'])}, {int(row['sequence_length'])}, {int(row['path_dim'])})", axis=1)

        # Table 1: Forward Performance
        forward_cols = ['config', 'truncation_level', 'signature_dim', 'keras_forward_ms', 'pathsig_forward_ms', 'forward_speedup']
        forward_df = df[forward_cols].copy()

        # Format columns
        forward_df['sig_dim_fmt'] = forward_df['signature_dim'].apply(format_signature_dim)
        forward_df['keras_time'] = forward_df['keras_forward_ms'].apply(format_time)
        forward_df['pathsig_time'] = forward_df['pathsig_forward_ms'].apply(format_time)
        forward_df['speedup'] = forward_df['forward_speedup'].apply(format_speedup)

        # Select formatted columns
        display_df = forward_df[['config', 'truncation_level', 'sig_dim_fmt', 'keras_time', 'pathsig_time', 'speedup']]
        display_df.columns = ['(B, M, d)', 'Trunc', 'Sig Dim', 'keras_sig', 'pathsig', 'Speedup']
        markdown_forward = f"# Forward Pass Benchmark (No Path Grads, dtype={str(self.dtype).split('.')[-1]})\n\n" + display_df.to_markdown(index=False)

        with open(os.path.join(output_dir, 'forward_performance.md'), 'w') as f:
            f.write(markdown_forward)

        # Table 2: Training Performance
        training_cols = ['config', 'truncation_level', 'signature_dim', 'keras_training_forward_ms', 'keras_backward_ms',
                         'pathsig_training_forward_ms', 'pathsig_backward_ms', 'training_speedup']
        training_df = df[training_cols].copy()

        # Format columns
        training_df['sig_dim_fmt'] = training_df['signature_dim'].apply(format_signature_dim)
        training_df['k_fwd'] = training_df['keras_training_forward_ms'].apply(format_time)
        training_df['k_bwd'] = training_df['keras_backward_ms'].apply(format_time)
        training_df['p_fwd'] = training_df['pathsig_training_forward_ms'].apply(format_time)
        training_df['p_bwd'] = training_df['pathsig_backward_ms'].apply(format_time)
        training_df['speedup'] = training_df['training_speedup'].apply(format_speedup)

        # Select formatted columns
        display_df = training_df[['config', 'truncation_level', 'sig_dim_fmt', 'k_fwd', 'k_bwd', 'p_fwd', 'p_bwd', 'speedup']]
        display_df.columns = ['(B, M, d)', 'Trunc', 'Sig Size', 'keras_sig-Fwd', 'keras_sig-Bwd', 'pathsig-Fwd', 'pathsig-Bwd', 'Speedup']
        markdown_training = f"# Training Benchmark (With Path Grads, dtype={str(self.dtype).split('.')[-1]})\n\n" + display_df.to_markdown(index=False)

        with open(os.path.join(output_dir, 'training_performance.md'), 'w') as f:
            f.write(markdown_training)

        # Table 3: Memory Usage
        memory_cols = ['config', 'truncation_level', 'signature_dim', 'memory_estimate', 'keras_forward_memory',
                       'pathsig_forward_memory', 'keras_training_memory', 'pathsig_training_memory']
        memory_df = df[memory_cols].copy()

        # Format columns
        memory_df['sig_dim_fmt'] = memory_df['signature_dim'].apply(format_signature_dim)
        memory_df['sig_mem'] = memory_df['memory_estimate'].apply(format_memory)
        memory_df['k_forward_mem'] = memory_df['keras_forward_memory'].apply(format_memory)
        memory_df['p_forward_mem'] = memory_df['pathsig_forward_memory'].apply(format_memory)
        memory_df['k_train_mem'] = memory_df['keras_training_memory'].apply(format_memory)
        memory_df['p_train_mem'] = memory_df['pathsig_training_memory'].apply(format_memory)

        # Select formatted columns
        display_df = memory_df[['config', 'truncation_level', 'sig_dim_fmt', 'sig_mem', 'k_forward_mem', 'k_train_mem', 'p_forward_mem', 'p_train_mem']]
        display_df.columns = ['(B, M, d)', 'Trunc', 'Sig Size', 'Sigs Mem', 'keras_sig-Fwd', 'keras_sig-Train', 'pathsig-Fwd', 'pathsig-Train']
        markdown_memory = f"# Memory Usage Comparison ({str(self.dtype).split('.')[-1]})\n\n" + display_df.to_markdown(index=False)

        with open(os.path.join(output_dir, 'memory_usage.md'), 'w') as f:
            f.write(markdown_memory)

        print(f"\nMarkdown tables saved to {output_dir}/")

    def plot_speedup_heatmaps(self, df: pd.DataFrame, output_dir: str = './figures'):
        """Generate speedup heatmaps for forward and training."""
        os.makedirs(output_dir, exist_ok=True)

        if df['forward_speedup'].isna().all():
            print("Cannot generate speedup heatmaps without comparison data")
            return

        # Setup style
        sns.set_style("whitegrid")

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Forward speedup heatmap
        ax = axes[0]
        pivot_forward = df.pivot_table(
            values='forward_speedup',
            index='sequence_length',
            columns='batch_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot_forward, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=1, ax=ax, cbar_kws={'label': 'Speedup Factor'})
        ax.set_title('Forward Speedup', fontsize=14)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Sequence Length')

        # Subplot 2: Training speedup heatmap
        ax = axes[1]
        pivot_training = df.pivot_table(
            values='training_speedup',
            index='sequence_length',
            columns='batch_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot_training, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=1, ax=ax, cbar_kws={'label': 'Speedup Factor'}
                    )
        ax.set_title('Training Speedup (Forward + Backward)', fontsize=14)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Sequence Length')

        plt.suptitle(f"pathsig vs keras_sig Speedup ({str(self.dtype).split('.')[-1]})", fontsize=16, y=1.02)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(output_dir, 'speedup_heatmaps.pdf'))
        plt.savefig(os.path.join(output_dir, 'speedup_heatmaps.png'), dpi=300)
        print(f"Speedup heatmaps saved to {output_dir}/")


def generate_configs() -> List[BenchmarkConfig]:
    """Generate configurations for benchmarking.
    Applies a shared grid of (path_dim, truncation_level) to every combination of batch_size and sequence_length,
    using the same set of sequence lengths for each batch size so results are directly comparable across batch sizes.
    """
    BATCH_SIZES = (1, 32, 64, 128, 256)

    SEQ_LENGTHS = (25, 50, 100, 500, 1000)

    # Shared grid for all (B, L)
    PATH_DIMS = (2, 3, 4, 6, 8, 10, 20, 30, 40)
    TRUNC_LEVELS = (3, 4, 5, 6)

    return [
        BenchmarkConfig(B, L, d, m)
        for B in BATCH_SIZES
        for L in SEQ_LENGTHS
        for m in TRUNC_LEVELS
        for d in PATH_DIMS
        if not (d >= 10 and m == 6)
        if not (d >= 20 and m == 5)
        if not (d >= 30 and m == 4)
    ]


def main():
    """Main benchmarks execution."""
    # Create output directory
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    float_type = torch.float32
    set_keras_dtype(float_type)
    output_dir = f"signature_benchmark_{gpu_name}_{str(float_type).split('.')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize benchmarks with desired dtype
    benchmark = SignatureBenchmark(
        warmup_runs=1,
        benchmark_runs=10,
        dtype=float_type
    )

    # Generate configurations
    configs = generate_configs()
    print(f"\nBenchmarking {len(configs)} configurations")
    print("=" * 60)

    # Run benchmarks
    results_df = benchmark.run_suite(configs)

    # Generate Markdown tables
    benchmark.generate_markdown_tables(results_df, os.path.join(output_dir, 'tables'))

    # Generate speedup heatmaps
    benchmark.plot_speedup_heatmaps(results_df, os.path.join(output_dir, 'figures'))

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()