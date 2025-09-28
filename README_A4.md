# Phase A4: Scalability Benchmarks & Router

This phase implements comprehensive scalability benchmarking and intelligent router functionality for the Turkish LLM system with **22 comprehensive Turkish business sectors**.

## 🎯 Overview

Phase A4 provides:
- **Sector-based Router**: Intelligent adapter selection based on text classification across 22 Turkish business sectors
- **Load Testing**: Comprehensive performance testing with Locust
- **SOTA Comparison**: Benchmark metrics vs. state-of-the-art models
- **Performance Analysis**: Detailed metrics and visualization tools

## 📁 File Structure

```
├── configs/
│   └── sectors.yaml              # Sector configuration and router settings (22 sectors)
├── services/
│   └── router.py                 # Sector router implementation
├── scripts/
│   ├── benchmark_locust.py       # Locust load testing script
│   ├── benchmark_metrics.py      # Metrics calculation and SOTA comparison
│   ├── run_benchmarks.py         # Comprehensive benchmark runner
│   └── test_a4_components.py     # Component testing with 22 sectors
├── requirements_benchmark.txt    # Benchmarking dependencies
└── README_A4.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_benchmark.txt
```

### 2. Test the Router

```bash
python services/router.py
```

### 3. Run Component Tests (22 Sectors)

```bash
python scripts/test_a4_components.py
```

### 4. Run Load Tests

```bash
# Basic load test
locust -f scripts/benchmark_locust.py --host=http://localhost:8000

# Comprehensive benchmark
python scripts/run_benchmarks.py --host=http://localhost:8000
```

## 🔧 Components

### 1. Sector Router (`services/router.py`)

The router intelligently selects adapters based on text classification across 22 Turkish business sectors:

```python
from services.router import SectorRouter

# Initialize router
router = SectorRouter()

# Classify text into sectors
sector_scores = router.classify_sector("Banka kredisi almak istiyorum")
print(sector_scores)  # [('finance_banking', 0.85), ('general', 0.15)]

# Select adapters for inference
adapters = router.select_adapters("Banka kredisi almak istiyorum")
print(adapters)  # ['adapters/finance_banking_adapter', 'adapters/general_adapter']
```

**Features:**
- **22 Turkish Business Sectors**: Comprehensive coverage of Turkish business domains
- Keyword-based sector classification with Turkish language optimization
- Confidence scoring with fallback thresholds
- Load balancing (round-robin)
- Thread-safe load statistics
- Configurable expert selection (up to 3 experts per token)

### 2. Load Testing (`scripts/benchmark_locust.py`)

Comprehensive load testing with sector-specific queries:

```bash
# Run with specific parameters
locust -f scripts/benchmark_locust.py \
  --host=http://localhost:8000 \
  --users=50 \
  --spawn-rate=5 \
  --run-time=5m \
  --headless \
  --csv=results.csv
```

**Test Types:**
- Sector-specific queries (75% of requests) across all 22 sectors
- General queries (25% of requests)
- Router endpoint testing
- Real-time performance monitoring

### 3. Metrics Analysis (`scripts/benchmark_metrics.py`)

Comprehensive metrics calculation and SOTA comparison:

```python
from scripts.benchmark_metrics import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer()

# Calculate metrics from raw data
metrics = analyzer.calculate_metrics(
    latencies=[45, 52, 48, 61, 55],
    success_count=95,
    total_count=100,
    sector_results=sector_results,
    resource_usage=resource_usage
)

# Compare with SOTA
comparison = analyzer.compare_with_sota(metrics, "llama-2-7b")

# Generate report
report = analyzer.generate_performance_report(metrics, comparison)
print(report)
```

**Metrics Tracked:**
- Latency (avg, P50, P95, P99, min, max)
- Throughput (requests/sec, tokens/sec)
- Quality (success rate, accuracy)
- Resource utilization (GPU, memory, CPU)
- Cost metrics (per request, per token)
- Sector-specific accuracy across all 22 sectors

### 4. Benchmark Runner (`scripts/run_benchmarks.py`)

Orchestrates the entire benchmarking process:

```bash
# Run comprehensive benchmark suite
python scripts/run_benchmarks.py \
  --host=http://localhost:8000 \
  --output-dir=benchmark_results
```

**Features:**
- Automated load testing with multiple configurations
- System resource monitoring
- Sector accuracy testing across all 22 sectors
- Results aggregation and analysis
- Report generation with visualizations

## 📊 Configuration

### Sector Configuration (`configs/sectors.yaml`)

Define 22 Turkish business sectors and router behavior:

```yaml
sectors:
  finance_banking:
    name: "Finans ve Bankacılık"
    description: "Bankacılık, finansal hizmetler, yatırım ve para yönetimi"
    keywords: ["banka", "kredi", "finans", "yatırım", "para", "döviz", "borsa"]
    adapter_path: "adapters/finance_banking_adapter"
    priority: 1

  healthcare:
    name: "Sağlık"
    description: "Tıp, eczacılık, hastane yönetimi ve sağlık hizmetleri"
    keywords: ["hastane", "doktor", "ilaç", "tedavi", "sağlık", "tıp", "eczane"]
    adapter_path: "adapters/healthcare_adapter"
    priority: 2

  # ... 20 more sectors

router:
  default_adapter: "adapters/general_adapter"
  fallback_threshold: 0.2
  max_experts_per_token: 3
  load_balancing: "round_robin"

benchmarks:
  latency_threshold_ms: 100
  throughput_target_rps: 50
  memory_limit_gb: 8
  gpu_utilization_target: 0.8
```

## 🏢 Comprehensive Turkish Business Sectors

The system supports **22 comprehensive Turkish business sectors**:

### Core Business Sectors
1. **Finans ve Bankacılık** - Banking and financial services
2. **Sağlık** - Healthcare and medical services
3. **Eğitim** - Education and training
4. **Medya ve Yayıncılık** - Media and publishing
5. **Hukuk** - Legal services
6. **Kamu Yönetimi** - Public administration

### Industrial & Manufacturing
7. **İmalat Endüstrisi** - Manufacturing industry
8. **Varlık Takibi** - Asset tracking
9. **Sigortacılık** - Insurance
10. **Enerji** - Energy
11. **Enerji Üretimi, Dağıtımı ve İletimi** - Energy production, distribution, and transmission
12. **Tarım** - Agriculture

### Transportation & Logistics
13. **Ulaşım** - Transportation
14. **Lojistik** - Logistics
15. **Telekomünikasyon** - Telecommunications

### Technology & Infrastructure
16. **İnşaat ve Mimarlık** - Construction and architecture
17. **Akıllı Şehirler, Kentleşme ve Altyapı** - Smart cities, urbanization, and infrastructure
18. **Mobilite** - Mobility
19. **Savunma ve Güvenlik** - Defense and security
20. **Acil Durum İletişimi ve Afet Yönetimi** - Emergency communication and disaster management

### Commerce & Services
21. **Turizm ve Otelcilik** - Tourism and hospitality
22. **E-ticaret** - E-commerce

## 📈 Benchmark Results

### Performance Metrics

The system tracks comprehensive metrics:

- **Latency**: Average, P50, P95, P99 percentiles
- **Throughput**: Requests per second, tokens per second
- **Quality**: Success rate, error rate, accuracy scores
- **Resources**: GPU utilization, memory usage, CPU usage
- **Cost**: Cost per request, cost per token

### SOTA Comparison

Compares against state-of-the-art models:

- GPT-4, GPT-3.5 Turbo
- LLaMA-2-7B, Mixtral-7B
- BERTurk (Turkish-specific)

### Visualizations

Automatically generates:
- Latency distribution charts
- Throughput comparison graphs
- Resource utilization plots
- Sector accuracy breakdowns across all 22 sectors

## 🎯 Use Cases

### 1. Production Load Testing

```bash
# Test production-like load
python scripts/run_benchmarks.py \
  --host=https://your-production-api.com \
  --output-dir=production_benchmarks
```

### 2. Development Performance Monitoring

```bash
# Quick performance check
locust -f scripts/benchmark_locust.py \
  --host=http://localhost:8000 \
  --users=10 \
  --spawn-rate=2 \
  --run-time=2m \
  --headless
```

### 3. Sector-Specific Testing

```python
# Test specific sector performance
router = SectorRouter()
finance_queries = [
    "Banka kredisi almak istiyorum",
    "Portföy yönetimi için yatırım danışmanlığı arıyorum",
    "Faiz oranları hakkında bilgi istiyorum"
]

for query in finance_queries:
    adapters = router.select_adapters(query)
    print(f"Query: {query}")
    print(f"Selected adapters: {adapters}")
```

## 🔍 Troubleshooting

### Common Issues

1. **Locust not found**
   ```bash
   pip install locust
   ```

2. **GPU monitoring fails**
   ```bash
   pip install GPUtil nvidia-ml-py
   ```

3. **Import errors**
   ```bash
   # Add project root to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Performance Tuning

1. **Adjust router thresholds** in `configs/sectors.yaml`
2. **Modify load test parameters** in `scripts/run_benchmarks.py`
3. **Update SOTA metrics** in `scripts/benchmark_metrics.py`

## 📋 Example Output

### Router Test Results

```
=== Sector Router Test Results ===

Test 1: Banka kredisi almak istiyorum, faiz oranları nedir?
Classification: [('finance_banking', 1.0)]
Selected adapters: ['adapters/finance_banking_adapter', 'adapters/general_adapter']

Test 2: Hastane randevusu almak istiyorum, hangi doktorlar müsait?
Classification: [('healthcare', 1.0)]
Selected adapters: ['adapters/healthcare_adapter', 'adapters/general_adapter']

Test 3: E-ticaret sitesi kurulumu için adımlar neler?
Classification: [('ecommerce', 1.0)]
Selected adapters: ['adapters/ecommerce_adapter', 'adapters/general_adapter']
```

### Benchmark Report

```
============================================================
TURKISH LLM BENCHMARK REPORT
============================================================

📊 LATENCY METRICS
------------------------------
Average Latency: 52.34ms
P50 Latency: 48.12ms
P95 Latency: 89.45ms
P99 Latency: 156.78ms

🚀 THROUGHPUT METRICS
------------------------------
Requests per Second: 45.67
Tokens per Second: 2283.50
Concurrent Users: 100

✅ QUALITY METRICS
------------------------------
Success Rate: 98.50%
Error Rate: 1.50%
Accuracy Score: 0.870

🏆 SOTA COMPARISON
------------------------------
Compared against: LLaMA-2-7B
Latency Improvement: +4.5%
Throughput Improvement: -8.7%
Accuracy Improvement: +2.4%
Cost Savings: +20.0%

🎯 SECTOR PERFORMANCE (Top 10)
------------------------------
 1. Education: 92.0%
 2. E-commerce: 89.0%
 3. Media Publishing: 88.0%
 4. Telecommunications: 88.0%
 5. Tourism Hospitality: 87.0%
 6. Mobility: 87.0%
 7. Manufacturing: 86.0%
 8. Logistics: 86.0%
 9. Finance Banking: 85.0%
10. Transportation: 85.0%
```

## 🚀 Next Steps

1. **Deploy to production** with the comprehensive router system
2. **Monitor performance** using the benchmark tools across all 22 sectors
3. **Optimize adapters** based on sector-specific performance data
4. **Scale horizontally** with load balancing
5. **Add more sectors** as needed for specific business domains

## 📚 Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Turkish NLP Resources](https://github.com/turkish-nlp-suite)

---

**Phase A4 Complete!** 🎉 The Turkish LLM system now has comprehensive benchmarking and intelligent routing capabilities across **22 Turkish business sectors**. 