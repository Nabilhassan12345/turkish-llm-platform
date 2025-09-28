#!/usr/bin/env python3
"""
Comprehensive benchmark runner for Turkish LLM system.
This script orchestrates the entire benchmarking process including:
- Load testing with Locust
- Metrics collection and analysis
- SOTA comparison
- Report generation
"""

import os
import sys
import time
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import GPUtil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_metrics import BenchmarkAnalyzer, BenchmarkMetrics
from services.router import SectorRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for Turkish LLM system.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.analyzer = BenchmarkAnalyzer()
        self.router = SectorRouter()
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

    def get_system_info(self) -> Dict:
        """Collect system information for benchmarking."""
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "architecture": psutil.machine(),
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
            },
            "gpu": {},
        }

        # Get GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                system_info["gpu"] = {
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_free_mb": gpu.memoryFree,
                    "temperature": gpu.temperature,
                    "load": gpu.load,
                }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")

        return system_info

    def run_load_test(
        self, host: str, users: int, spawn_rate: int, run_time: str, output_file: str
    ) -> bool:
        """
        Run Locust load test.

        Args:
            host: Target host URL
            users: Number of concurrent users
            spawn_rate: Users spawned per second
            run_time: Test duration (e.g., "5m", "300s")
            output_file: Output file for results

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(
                f"Starting load test: {users} users, {spawn_rate} spawn rate, {run_time} duration"
            )

            # Build Locust command
            cmd = [
                "locust",
                "-f",
                "scripts/benchmark_locust.py",
                "--host",
                host,
                "--users",
                str(users),
                "--spawn-rate",
                str(spawn_rate),
                "--run-time",
                run_time,
                "--headless",
                "--csv",
                output_file,
                "--html",
                output_file.replace(".csv", ".html"),
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(run_time.replace("s", "").replace("m", "")) * 60 + 300,
            )

            if result.returncode == 0:
                logger.info("Load test completed successfully")
                return True
            else:
                logger.error(f"Load test failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Load test timed out")
            return False
        except Exception as e:
            logger.error(f"Load test error: {e}")
            return False

    def parse_locust_results(self, csv_file: str) -> Dict:
        """
        Parse Locust CSV results file.

        Args:
            csv_file: Path to Locust CSV results file

        Returns:
            Dictionary with parsed results
        """
        try:
            import pandas as pd

            # Read CSV file
            df = pd.read_csv(csv_file)

            # Extract metrics
            results = {
                "total_requests": int(df["request_count"].sum()),
                "successful_requests": int(
                    df[df["response_code"] == 200]["request_count"].sum()
                ),
                "failed_requests": int(
                    df[df["response_code"] != 200]["request_count"].sum()
                ),
                "latencies": df["response_time"].tolist(),
                "avg_latency": float(df["response_time"].mean()),
                "p50_latency": float(df["response_time"].quantile(0.5)),
                "p95_latency": float(df["response_time"].quantile(0.95)),
                "p99_latency": float(df["response_time"].quantile(0.99)),
                "requests_per_second": float(df["requests_per_sec"].mean()),
                "min_latency": float(df["response_time"].min()),
                "max_latency": float(df["response_time"].max()),
            }

            return results

        except Exception as e:
            logger.error(f"Error parsing Locust results: {e}")
            return {}

    def collect_resource_metrics(self) -> Dict:
        """
        Collect real-time resource utilization metrics.

        Returns:
            Dictionary with resource metrics
        """
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)

            # GPU utilization
            gpu_utilization = 0.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilization = gpus[0].load * 100
            except:
                pass

            return {
                "cpu_utilization": cpu_percent,
                "memory_usage_gb": memory_gb,
                "gpu_utilization": gpu_utilization,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return {}

    def run_sector_accuracy_test(self) -> Dict:
        """
        Run sector-specific accuracy tests.

        Returns:
            Dictionary with sector accuracy results
        """
        logger.info("Running sector accuracy tests...")

        # Test cases for each sector
        test_cases = {
            "finance": [
                ("Banka kredisi almak istiyorum", "finance"),
                ("Döviz kuru tahminleri nasıl yapılır", "finance"),
                ("Sigorta poliçesi seçerken nelere dikkat etmeliyim", "finance"),
            ],
            "healthcare": [
                ("Hastane randevusu almak istiyorum", "healthcare"),
                ("Bu ilacın yan etkileri nelerdir", "healthcare"),
                ("Doktor muayenesi öncesi nasıl hazırlanmalıyım", "healthcare"),
            ],
            "education": [
                ("Üniversite sınavına hazırlanıyorum", "education"),
                ("Online eğitim platformları hangileri", "education"),
                ("Yabancı dil öğrenmek için en iyi yöntemler", "education"),
            ],
            "technology": [
                ("Yazılım geliştirme projesi için teknoloji çözümleri", "technology"),
                ("Siber güvenlik önlemleri nelerdir", "technology"),
                ("Cloud computing avantajları nelerdir", "technology"),
            ],
            "retail": [
                ("Mağazada ürün satışı yapıyorum", "retail"),
                ("E-ticaret sitesi kurulumu için adımlar", "retail"),
                ("Müşteri hizmetleri stratejileri nelerdir", "retail"),
            ],
            "manufacturing": [
                (
                    "Fabrikada üretim süreçlerini optimize etmek istiyorum",
                    "manufacturing",
                ),
                ("Kalite kontrol süreçleri nasıl iyileştirilir", "manufacturing"),
                ("Makine bakım planlaması nasıl yapılır", "manufacturing"),
            ],
        }

        sector_results = {}

        for sector, cases in test_cases.items():
            correct = 0
            total = len(cases)

            for query, expected_sector in cases:
                # Use router to classify
                sector_scores = self.router.classify_sector(query)

                if sector_scores and sector_scores[0][0] == expected_sector:
                    correct += 1

            sector_results[sector] = {
                "correct": correct,
                "total": total,
                "accuracy": (correct / total) * 100 if total > 0 else 0,
            }

        return sector_results

    def run_comprehensive_benchmark(self, host: str = "http://localhost:8000") -> Dict:
        """
        Run comprehensive benchmark suite.

        Args:
            host: Target host URL

        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Starting comprehensive benchmark suite...")

        # Collect system info
        system_info = self.get_system_info()
        logger.info(
            f"System info collected: {system_info['cpu']['count']} CPUs, "
            f"{system_info['memory']['total_gb']:.1f}GB RAM"
        )

        # Run sector accuracy test
        sector_results = self.run_sector_accuracy_test()
        logger.info("Sector accuracy test completed")

        # Run load tests with different configurations
        load_test_results = {}
        test_configs = [
            {"users": 10, "spawn_rate": 2, "duration": "2m", "name": "light_load"},
            {"users": 50, "spawn_rate": 5, "duration": "3m", "name": "medium_load"},
            {"users": 100, "spawn_rate": 10, "duration": "2m", "name": "heavy_load"},
        ]

        for config in test_configs:
            logger.info(f"Running {config['name']} test...")

            output_file = self.results_dir / f"locust_{config['name']}.csv"

            # Collect resource metrics before test
            resource_before = self.collect_resource_metrics()

            # Run load test
            success = self.run_load_test(
                host=host,
                users=config["users"],
                spawn_rate=config["spawn_rate"],
                run_time=config["duration"],
                output_file=str(output_file),
            )

            # Collect resource metrics after test
            resource_after = self.collect_resource_metrics()

            if success and output_file.exists():
                # Parse results
                parsed_results = self.parse_locust_results(str(output_file))
                parsed_results.update(
                    {
                        "resource_before": resource_before,
                        "resource_after": resource_after,
                        "test_config": config,
                    }
                )
                load_test_results[config["name"]] = parsed_results

                logger.info(
                    f"{config['name']} test completed: "
                    f"{parsed_results['requests_per_second']:.1f} RPS, "
                    f"{parsed_results['avg_latency']:.1f}ms avg latency"
                )
            else:
                logger.error(f"{config['name']} test failed")

        # Calculate comprehensive metrics
        all_latencies = []
        total_requests = 0
        total_successful = 0

        for test_name, results in load_test_results.items():
            all_latencies.extend(results.get("latencies", []))
            total_requests += results.get("total_requests", 0)
            total_successful += results.get("successful_requests", 0)

        # Calculate resource usage averages
        resource_usage = {
            "gpu_utilization": 0.0,
            "memory_usage_gb": 0.0,
            "cpu_utilization": 0.0,
            "accuracy_score": 0.85,  # Default accuracy
            "cost_per_request": 0.001,  # Estimated cost
            "cost_per_token": 0.00002,  # Estimated cost per token
            "concurrent_users": max(config["users"] for config in test_configs),
        }

        # Calculate average resource usage
        resource_samples = 0
        for test_name, results in load_test_results.items():
            if "resource_after" in results:
                resource_usage["gpu_utilization"] += results["resource_after"].get(
                    "gpu_utilization", 0
                )
                resource_usage["memory_usage_gb"] += results["resource_after"].get(
                    "memory_usage_gb", 0
                )
                resource_usage["cpu_utilization"] += results["resource_after"].get(
                    "cpu_utilization", 0
                )
                resource_samples += 1

        if resource_samples > 0:
            resource_usage["gpu_utilization"] /= resource_samples
            resource_usage["memory_usage_gb"] /= resource_samples
            resource_usage["cpu_utilization"] /= resource_samples

        # Calculate final metrics
        metrics = self.analyzer.calculate_metrics(
            latencies=all_latencies,
            success_count=total_successful,
            total_count=total_requests,
            sector_results=sector_results,
            resource_usage=resource_usage,
        )

        # Compare with SOTA
        comparison = self.analyzer.compare_with_sota(metrics, "llama-2-7b")

        # Generate comprehensive results
        comprehensive_results = {
            "system_info": system_info,
            "metrics": metrics,
            "sota_comparison": comparison,
            "load_test_results": load_test_results,
            "sector_results": sector_results,
            "timestamp": datetime.now().isoformat(),
        }

        return comprehensive_results

    def save_and_report(self, results: Dict, output_dir: str = "benchmark_results"):
        """
        Save results and generate reports.

        Args:
            results: Comprehensive benchmark results
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save raw results
        results_file = output_path / "comprehensive_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Generate performance report
        metrics = results["metrics"]
        comparison = results["sota_comparison"]
        report = self.analyzer.generate_performance_report(metrics, comparison)

        report_file = output_path / "performance_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        # Create visualizations
        self.analyzer.create_visualizations(metrics, str(output_path))

        # Save SOTA comparison
        self.analyzer.save_benchmark_results(
            metrics, comparison, str(output_path / "benchmark_results.json")
        )

        logger.info(f"Results saved to {output_path}/")
        logger.info(f"Performance report: {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Average Latency: {metrics.avg_latency:.2f}ms")
        print(f"P95 Latency: {metrics.p95_latency:.2f}ms")
        print(f"Requests per Second: {metrics.requests_per_second:.2f}")
        print(f"Success Rate: {metrics.success_rate:.2f}%")
        print(f"GPU Utilization: {metrics.gpu_utilization:.1f}%")
        print(f"Memory Usage: {metrics.memory_usage_gb:.2f}GB")

        target_model = comparison["target_model"]
        latency_improvement = comparison["latency_comparison"]["latency_improvement"]
        throughput_improvement = comparison["throughput_comparison"][
            "throughput_improvement"
        ]

        print(f"\nCompared to {target_model.upper()}:")
        print(f"Latency Improvement: {latency_improvement:+.1f}%")
        print(f"Throughput Improvement: {throughput_improvement:+.1f}%")
        print("=" * 60)


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Turkish LLM Benchmark Runner")
    parser.add_argument(
        "--host", default="http://localhost:8000", help="Target host URL"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--config", default="configs/sectors.yaml", help="Configuration file path"
    )

    args = parser.parse_args()

    # Load configuration
    config = {
        "host": args.host,
        "output_dir": args.output_dir,
        "config_file": args.config,
    }

    # Initialize runner
    runner = BenchmarkRunner(config)

    try:
        # Run comprehensive benchmark
        logger.info("Starting Turkish LLM benchmark suite...")
        results = runner.run_comprehensive_benchmark(args.host)

        # Save results and generate reports
        runner.save_and_report(results, args.output_dir)

        logger.info("Benchmark suite completed successfully!")

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
