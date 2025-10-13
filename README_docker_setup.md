# M/M/1 Server Monitoring Setup

This setup provides a complete environment for performance testing with an M/M/1 server and monitoring stack.

## Components

### M/M/1 Server
- **Image**: `bistrulli/generic-microservice-tester:0.6`
- **Port**: 8084 (via Istio proxy)
- **Configurable service time** via `SERVICE_TIME_SECONDS` environment variable
- **Single worker/thread** configuration for true M/M/1 behavior

### Monitoring Stack
- **Envoy Proxy** (port 8084 for traffic, 9901 for admin/metrics): Application-level metrics
- **cAdvisor** (port 8081): Container resource monitoring
- **Prometheus** (port 9090): Metrics collection and storage

## Quick Start

1. **Start the stack**:
   ```bash
   docker-compose up -d
   ```

2. **Test the M/M/1 server**:
   ```bash
   curl http://localhost:8084/health
   ```

3. **Access monitoring**:
   - Prometheus: http://localhost:9090
   - cAdvisor: http://localhost:8081
   - Envoy Admin/Metrics: http://localhost:9901

## Configuration

### Adjust Service Time
To change the M/M/1 server service time, modify the environment variable in `docker-compose.yml`:

```yaml
environment:
  SERVICE_TIME_SECONDS: "0.1650"  # Change this value
```

Then restart:
```bash
docker-compose restart mm1-server
```

### Available Metrics

#### Application-Level (Envoy Proxy)
- **`envoy_http_ingress_http_*`**: Inbound HTTP request metrics
- **`envoy_cluster_mm1_service_*`**: Request rate and timing to M/M/1 server
- **`envoy_cluster_upstream_rq_time`**: Response time histograms
- **`envoy_http_*_response_*`**: HTTP status codes and response metrics

#### Container-Level (cAdvisor)
- **CPU**: `container_cpu_usage_seconds_total`
- **Memory**: `container_memory_usage_bytes`
- **Network**: `container_network_*`

## Key Metrics for SPE Analysis

1. **Throughput**: `rate(envoy_cluster_upstream_rq_total[1m])`
2. **Response Time**: `envoy_cluster_upstream_rq_time` (histogram)
3. **CPU Utilization**: `rate(container_cpu_usage_seconds_total[1m])`
4. **Success Rate**: `envoy_cluster_upstream_rq_*` by response code

### Example Prometheus Queries

```promql
# Requests per second
rate(envoy_cluster_upstream_rq_total[1m])

# Average response time
histogram_quantile(0.5, rate(envoy_cluster_upstream_rq_time_bucket[1m]))

# CPU utilization percentage
rate(container_cpu_usage_seconds_total{name="mm1-server"}[1m]) * 100

# 95th percentile response time
histogram_quantile(0.95, rate(envoy_cluster_upstream_rq_time_bucket[1m]))
```

## Usage in Experiments

This setup enables testing:
- **Open vs Closed workload patterns**
- **Service time impact on system behavior**
- **Validation of M/M/1 theoretical formulas**
- **Performance degradation near capacity**

## Cleanup

```bash
docker-compose down -v
```