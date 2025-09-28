#!/usr/bin/env python3
"""
Benchmark metrics and SOTA comparison utilities for Turkish LLM.
This script defines performance metrics, compares against state-of-the-art models,
and provides detailed analysis tools.
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from services.router import SectorRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics for LLM evaluation."""

    # Latency metrics (milliseconds)
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float

    # Throughput metrics
    requests_per_second: float
    tokens_per_second: float
    concurrent_users: int

    # Quality metrics
    success_rate: float
    error_rate: float
    accuracy_score: float

    # Resource utilization
    gpu_utilization: float
    memory_usage_gb: float
    cpu_utilization: float

    # Cost metrics
    cost_per_request: float
    cost_per_token: float

    # Sector-specific metrics
    sector_accuracy: Dict[str, float]
    adapter_utilization: Dict[str, float]


@dataclass
class SOTAMetrics:
    """State-of-the-art model metrics for comparison."""

    model_name: str
    model_size: str
    avg_latency: float
    p95_latency: float
    requests_per_second: float
    accuracy_score: float
    cost_per_request: float
    language_support: List[str]


class BenchmarkAnalyzer:
    """
    Comprehensive benchmark analyzer for Turkish LLM performance evaluation.
    """

    def __init__(self):
        self.router = SectorRouter()
        self.sota_models = self._load_sota_metrics()
        self.benchmark_results = []

    def _load_sota_metrics(self) -> Dict[str, SOTAMetrics]:
        """Load SOTA model metrics for comparison."""
        return {
            "gpt-4": SOTAMetrics(
                model_name="GPT-4",
                model_size="175B",
                avg_latency=200.0,
                p95_latency=500.0,
                requests_per_second=10.0,
                accuracy_score=0.95,
                cost_per_request=0.03,
                language_support=["English", "Turkish"],
            ),
            "gpt-3.5-turbo": SOTAMetrics(
                model_name="GPT-3.5 Turbo",
                model_size="175B",
                avg_latency=100.0,
                p95_latency=250.0,
                requests_per_second=25.0,
                accuracy_score=0.90,
                cost_per_request=0.002,
                language_support=["English", "Turkish"],
            ),
            "llama-2-7b": SOTAMetrics(
                model_name="LLaMA-2-7B",
                model_size="7B",
                avg_latency=50.0,
                p95_latency=150.0,
                requests_per_second=50.0,
                accuracy_score=0.85,
                cost_per_request=0.001,
                language_support=["English", "Multilingual"],
            ),
            "mixtral-7b": SOTAMetrics(
                model_name="Mixtral-7B",
                model_size="7B",
                avg_latency=60.0,
                p95_latency=180.0,
                requests_per_second=40.0,
                accuracy_score=0.88,
                cost_per_request=0.001,
                language_support=["English", "Multilingual"],
            ),
            "berturk": SOTAMetrics(
                model_name="BERTurk",
                model_size="110M",
                avg_latency=20.0,
                p95_latency=50.0,
                requests_per_second=100.0,
                accuracy_score=0.82,
                cost_per_request=0.0005,
                language_support=["Turkish"],
            ),
        }

    def calculate_metrics(
        self,
        latencies: List[float],
        success_count: int,
        total_count: int,
        sector_results: Dict[str, Dict],
        resource_usage: Dict[str, float],
    ) -> BenchmarkMetrics:
        """
        Calculate comprehensive benchmark metrics from raw data.

        Args:
            latencies: List of response latencies in milliseconds
            success_count: Number of successful requests
            total_count: Total number of requests
            sector_results: Dictionary of sector-specific results
            resource_usage: Dictionary of resource utilization metrics

        Returns:
            BenchmarkMetrics object with calculated metrics
        """
        # Latency metrics
        latencies_array = np.array(latencies)
        avg_latency = np.mean(latencies_array)
        p50_latency = np.percentile(latencies_array, 50)
        p95_latency = np.percentile(latencies_array, 95)
        p99_latency = np.percentile(latencies_array, 99)
        min_latency = np.min(latencies_array)
        max_latency = np.max(latencies_array)

        # Throughput metrics (assuming 1 second test duration)
        requests_per_second = (
            success_count / (max_latency / 1000) if max_latency > 0 else 0
        )
        tokens_per_second = (
            requests_per_second * 50
        )  # Assuming average 50 tokens per response

        # Quality metrics
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        error_rate = 100 - success_rate

        # Sector accuracy (simplified calculation)
        sector_accuracy = {}
        for sector, results in sector_results.items():
            if "correct" in results and "total" in results:
                sector_accuracy[sector] = (results["correct"] / results["total"]) * 100

        # Adapter utilization from router
        adapter_utilization = self.router.get_load_statistics()

        return BenchmarkMetrics(
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
            concurrent_users=resource_usage.get("concurrent_users", 1),
            success_rate=success_rate,
            error_rate=error_rate,
            accuracy_score=resource_usage.get("accuracy_score", 0.85),
            gpu_utilization=resource_usage.get("gpu_utilization", 0.0),
            memory_usage_gb=resource_usage.get("memory_usage_gb", 0.0),
            cpu_utilization=resource_usage.get("cpu_utilization", 0.0),
            cost_per_request=resource_usage.get("cost_per_request", 0.001),
            cost_per_token=resource_usage.get("cost_per_token", 0.00002),
            sector_accuracy=sector_accuracy,
            adapter_utilization=adapter_utilization,
        )

    def compare_with_sota(
        self, metrics: BenchmarkMetrics, target_model: str = "llama-2-7b"
    ) -> Dict[str, Any]:
        """
        Compare benchmark results with SOTA models.

        Args:
            metrics: Benchmark metrics to compare
            target_model: SOTA model to compare against

        Returns:
            Dictionary with comparison results
        """
        if target_model not in self.sota_models:
            raise ValueError(f"Unknown SOTA model: {target_model}")

        sota = self.sota_models[target_model]

        comparison = {
            "target_model": target_model,
            "latency_comparison": {
                "avg_latency_ratio": metrics.avg_latency / sota.avg_latency,
                "p95_latency_ratio": metrics.p95_latency / sota.p95_latency,
                "latency_improvement": (
                    (sota.avg_latency - metrics.avg_latency) / sota.avg_latency
                )
                * 100,
            },
            "throughput_comparison": {
                "rps_ratio": metrics.requests_per_second / sota.requests_per_second,
                "throughput_improvement": (
                    (metrics.requests_per_second - sota.requests_per_second)
                    / sota.requests_per_second
                )
                * 100,
            },
            "quality_comparison": {
                "accuracy_ratio": metrics.accuracy_score / sota.accuracy_score,
                "accuracy_improvement": (
                    (metrics.accuracy_score - sota.accuracy_score) / sota.accuracy_score
                )
                * 100,
            },
            "cost_comparison": {
                "cost_ratio": metrics.cost_per_request / sota.cost_per_request,
                "cost_savings": (
                    (sota.cost_per_request - metrics.cost_per_request)
                    / sota.cost_per_request
                )
                * 100,
            },
        }

        return comparison

    def generate_performance_report(
        self, metrics: BenchmarkMetrics, comparison: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            metrics: Benchmark metrics
            comparison: SOTA comparison results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("TURKISH LLM BENCHMARK REPORT")
        report.append("=" * 60)

        # Latency Section
        report.append("\n📊 LATENCY METRICS")
        report.append("-" * 30)
        report.append(f"Average Latency: {metrics.avg_latency:.2f}ms")
        report.append(f"P50 Latency: {metrics.p50_latency:.2f}ms")
        report.append(f"P95 Latency: {metrics.p95_latency:.2f}ms")
        report.append(f"P99 Latency: {metrics.p99_latency:.2f}ms")
        report.append(
            f"Min/Max Latency: {metrics.min_latency:.2f}ms / {metrics.max_latency:.2f}ms"
        )

        # Throughput Section
        report.append("\n🚀 THROUGHPUT METRICS")
        report.append("-" * 30)
        report.append(f"Requests per Second: {metrics.requests_per_second:.2f}")
        report.append(f"Tokens per Second: {metrics.tokens_per_second:.2f}")
        report.append(f"Concurrent Users: {metrics.concurrent_users}")

        # Quality Section
        report.append("\n✅ QUALITY METRICS")
        report.append("-" * 30)
        report.append(f"Success Rate: {metrics.success_rate:.2f}%")
        report.append(f"Error Rate: {metrics.error_rate:.2f}%")
        report.append(f"Accuracy Score: {metrics.accuracy_score:.3f}")

        # Resource Section
        report.append("\n💻 RESOURCE UTILIZATION")
        report.append("-" * 30)
        report.append(f"GPU Utilization: {metrics.gpu_utilization:.1f}%")
        report.append(f"Memory Usage: {metrics.memory_usage_gb:.2f}GB")
        report.append(f"CPU Utilization: {metrics.cpu_utilization:.1f}%")

        # Cost Section
        report.append("\n💰 COST METRICS")
        report.append("-" * 30)
        report.append(f"Cost per Request: ${metrics.cost_per_request:.4f}")
        report.append(f"Cost per Token: ${metrics.cost_per_token:.6f}")

        # SOTA Comparison Section
        report.append("\n🏆 SOTA COMPARISON")
        report.append("-" * 30)
        target_model = comparison["target_model"]
        report.append(f"Compared against: {target_model.upper()}")

        latency_comp = comparison["latency_comparison"]
        report.append(
            f"Latency Improvement: {latency_comp['latency_improvement']:+.1f}%"
        )

        throughput_comp = comparison["throughput_comparison"]
        report.append(
            f"Throughput Improvement: {throughput_comp['throughput_improvement']:+.1f}%"
        )

        quality_comp = comparison["quality_comparison"]
        report.append(
            f"Accuracy Improvement: {quality_comp['accuracy_improvement']:+.1f}%"
        )

        cost_comp = comparison["cost_comparison"]
        report.append(f"Cost Savings: {cost_comp['cost_savings']:+.1f}%")

        # Sector Performance
        if metrics.sector_accuracy:
            report.append("\n🎯 SECTOR PERFORMANCE")
            report.append("-" * 30)
            for sector, accuracy in metrics.sector_accuracy.items():
                report.append(f"{sector.capitalize()}: {accuracy:.1f}%")

        # Adapter Utilization
        if metrics.adapter_utilization:
            report.append("\n🔧 ADAPTER UTILIZATION")
            report.append("-" * 30)
            for adapter, count in metrics.adapter_utilization.items():
                report.append(f"{adapter}: {count} requests")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def create_visualizations(
        self, metrics: BenchmarkMetrics, output_dir: str = "benchmark_results"
    ):
        """
        Create visualization charts for benchmark results.

        Args:
            metrics: Benchmark metrics
            output_dir: Directory to save visualizations
        """
        Path(output_dir).mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Latency Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Latency comparison with SOTA
        sota_models = list(self.sota_models.keys())[:4]
        sota_latencies = [self.sota_models[model].avg_latency for model in sota_models]
        current_latency = metrics.avg_latency

        axes[0, 0].bar(sota_models + ["Our Model"], sota_latencies + [current_latency])
        axes[0, 0].set_title("Average Latency Comparison")
        axes[0, 0].set_ylabel("Latency (ms)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Throughput comparison
        sota_throughput = [
            self.sota_models[model].requests_per_second for model in sota_models
        ]
        current_throughput = metrics.requests_per_second

        axes[0, 1].bar(
            sota_models + ["Our Model"], sota_throughput + [current_throughput]
        )
        axes[0, 1].set_title("Throughput Comparison")
        axes[0, 1].set_ylabel("Requests per Second")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Resource utilization
        resources = ["GPU", "Memory", "CPU"]
        utilization = [
            metrics.gpu_utilization,
            metrics.memory_usage_gb,
            metrics.cpu_utilization,
        ]

        axes[1, 0].bar(resources, utilization)
        axes[1, 0].set_title("Resource Utilization")
        axes[1, 0].set_ylabel("Utilization (%)")

        # Sector accuracy
        if metrics.sector_accuracy:
            sectors = list(metrics.sector_accuracy.keys())
            accuracies = list(metrics.sector_accuracy.values())

            axes[1, 1].bar(sectors, accuracies)
            axes[1, 1].set_title("Sector-Specific Accuracy")
            axes[1, 1].set_ylabel("Accuracy (%)")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/benchmark_overview.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Detailed latency analysis
        fig, ax = plt.subplots(figsize=(10, 6))

        latency_metrics = ["Min", "P50", "Avg", "P95", "P99", "Max"]
        latency_values = [
            metrics.min_latency,
            metrics.p50_latency,
            metrics.avg_latency,
            metrics.p95_latency,
            metrics.p99_latency,
            metrics.max_latency,
        ]

        bars = ax.bar(latency_metrics, latency_values, color="skyblue")
        ax.set_title("Latency Distribution Analysis")
        ax.set_ylabel("Latency (ms)")

        # Add value labels on bars
        for bar, value in zip(bars, latency_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}/")

    def save_benchmark_results(
        self,
        metrics: BenchmarkMetrics,
        comparison: Dict[str, Any],
        output_file: str = "benchmark_results/results.json",
    ):
        """
        Save benchmark results to JSON file.

        Args:
            metrics: Benchmark metrics
            comparison: SOTA comparison results
            output_file: Output file path
        """
        Path(output_file).parent.mkdir(exist_ok=True)

        results = {
            "timestamp": time.time(),
            "metrics": asdict(metrics),
            "sota_comparison": comparison,
            "sota_models": {
                name: asdict(model) for name, model in self.sota_models.items()
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Benchmark results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer()

    # Example metrics (replace with actual benchmark data)
    example_latencies = [45, 52, 48, 61, 55, 49, 58, 63, 47, 51]
    example_sector_results = {
        "finance": {"correct": 85, "total": 100},
        "healthcare": {"correct": 78, "total": 100},
        "education": {"correct": 92, "total": 100},
    }
    example_resource_usage = {
        "gpu_utilization": 75.5,
        "memory_usage_gb": 6.2,
        "cpu_utilization": 45.0,
        "accuracy_score": 0.87,
        "cost_per_request": 0.0008,
        "cost_per_token": 0.000016,
    }

    # Calculate metrics
    metrics = analyzer.calculate_metrics(
        latencies=example_latencies,
        success_count=95,
        total_count=100,
        sector_results=example_sector_results,
        resource_usage=example_resource_usage,
    )

    # Compare with SOTA
    comparison = analyzer.compare_with_sota(metrics, "llama-2-7b")

    # Generate report
    report = analyzer.generate_performance_report(metrics, comparison)
    print(report)

    # Create visualizations
    analyzer.create_visualizations(metrics)

    # Save results
    analyzer.save_benchmark_results(metrics, comparison)
