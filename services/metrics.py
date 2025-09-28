"""
Prometheus metrics for Turkish AI Agent
Fulfills F5: Logging & Metrics task
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from prometheus_client.metrics import (
    CounterMetricFamily,
    HistogramMetricFamily,
    GaugeMetricFamily,
)
import time
import psutil
import GPUtil
from typing import Dict, Any

# Request metrics
requests_total = Counter(
    "turkish_ai_requests_total",
    "Total number of requests",
    ["sector", "endpoint", "method"],
)

errors_total = Counter(
    "turkish_ai_errors_total",
    "Total number of errors",
    ["sector", "endpoint", "error_type"],
)

# Response time metrics
response_time = Histogram(
    "turkish_ai_response_time_seconds",
    "Response time in seconds",
    ["sector", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Throughput metrics
requests_in_progress = Gauge(
    "turkish_ai_requests_in_progress",
    "Number of requests currently being processed",
    ["sector"],
)

# Audio processing metrics
audio_processing_duration = Histogram(
    "turkish_ai_audio_processing_duration_seconds",
    "Audio processing duration in seconds",
    ["sector", "processing_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

stt_accuracy = Gauge(
    "turkish_ai_stt_accuracy", "Speech-to-text accuracy score", ["sector"]
)

tts_quality_score = Gauge(
    "turkish_ai_tts_quality_score", "Text-to-speech quality score", ["sector"]
)

# WebSocket metrics
websocket_connections = Gauge(
    "turkish_ai_websocket_connections_total",
    "Total number of WebSocket connections",
    ["sector"],
)

# Adapter performance metrics
adapter_accuracy = Gauge(
    "turkish_ai_adapter_accuracy",
    "Adapter accuracy score by sector",
    ["sector", "adapter_version"],
)

router_accuracy = Gauge(
    "turkish_ai_router_accuracy", "Router accuracy score", ["sector"]
)

# System resource metrics
gpu_utilization = Gauge(
    "turkish_ai_gpu_utilization", "GPU utilization percentage", ["gpu_id"]
)

memory_usage = Gauge("turkish_ai_memory_usage_bytes", "Memory usage in bytes", ["type"])

cpu_usage = Gauge("turkish_ai_cpu_usage_percent", "CPU usage percentage")

# Cost metrics (if applicable)
cost_per_request = Summary("turkish_ai_cost_sum", "Total cost of requests", ["sector"])


# Custom metrics collector for system resources
class SystemMetricsCollector:
    def collect(self):
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_utilization.labels(gpu_id=gpu.id).set(gpu.load * 100)
        except:
            pass

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage.labels(type="virtual").set(memory.total)
        memory_usage.labels(type="available").set(memory.available)
        memory_usage.labels(type="used").set(memory.used)

        # CPU metrics
        cpu_usage.set(psutil.cpu_percent(interval=1))


# Metrics decorator for timing requests
def track_request_time(sector: str, endpoint: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                # Record successful request
                requests_total.labels(
                    sector=sector, endpoint=endpoint, method="POST"
                ).inc()
                return result
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                errors_total.labels(
                    sector=sector, endpoint=endpoint, error_type=error_type
                ).inc()
                raise
            finally:
                # Record response time
                duration = time.time() - start_time
                response_time.labels(sector=sector, endpoint=endpoint).observe(duration)

        return wrapper

    return decorator


# Audio processing decorator
def track_audio_processing(sector: str, processing_type: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                audio_processing_duration.labels(
                    sector=sector, processing_type=processing_type
                ).observe(duration)

        return wrapper

    return decorator


# WebSocket connection tracking
def track_websocket_connection(sector: str, connected: bool):
    if connected:
        websocket_connections.labels(sector=sector).inc()
    else:
        websocket_connections.labels(sector=sector).dec()


# Update adapter performance
def update_adapter_accuracy(
    sector: str, accuracy: float, adapter_version: str = "latest"
):
    adapter_accuracy.labels(sector=sector, adapter_version=adapter_version).set(
        accuracy
    )


# Update router accuracy
def update_router_accuracy(sector: str, accuracy: float):
    router_accuracy.labels(sector=sector).set(accuracy)


# Update STT accuracy
def update_stt_accuracy(sector: str, accuracy: float):
    stt_accuracy.labels(sector=sector).set(accuracy)


# Update TTS quality score
def update_tts_quality_score(sector: str, score: float):
    tts_quality_score.labels(sector=sector).set(score)


# Record cost
def record_cost(sector: str, cost: float):
    cost_per_request.labels(sector=sector).observe(cost)


# Get metrics endpoint
def get_metrics():
    return generate_latest()


# Initialize system metrics collection
system_collector = SystemMetricsCollector()
