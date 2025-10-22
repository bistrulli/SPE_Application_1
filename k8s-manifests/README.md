# Kubernetes M/M/1 Setup with Istio

This directory contains Kubernetes manifests to deploy the M/M/1 SPE environment using Istio service mesh instead of standalone Envoy.

## Prerequisites

1. **Kubernetes cluster** with Istio installed
2. **Prometheus Operator** for monitoring
3. **kubectl** configured for your cluster

## Quick Deploy

```bash
# Apply all manifests
kubectl apply -f k8s-manifests/

# Check deployment status
kubectl get pods -n spe-system

# Get Istio gateway external IP
kubectl get svc -n istio-system istio-ingressgateway
```

## Architecture Changes from Docker Compose

| Component | Docker Compose | Kubernetes + Istio |
|-----------|----------------|-------------------|
| **Load Balancer** | Envoy proxy | Istio Envoy sidecar |
| **Service Discovery** | Docker DNS | Kubernetes DNS |
| **Metrics Collection** | Envoy admin endpoint | Istio telemetry v2 |
| **Container Monitoring** | cAdvisor | Native K8s metrics |
| **Traffic Management** | Static config | VirtualService/DestinationRule |

## Key Metrics Available

### Istio Metrics (replaces Envoy metrics)
```promql
# Request rate (λ)
rate(istio_requests_total{destination_service_name="mm1-server"}[1m])

# Response time percentiles
histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket{destination_service_name="mm1-server"}[1m]))

# Error rate
rate(istio_requests_total{destination_service_name="mm1-server",response_code!~"2.."}[1m])
```

### Native Kubernetes Metrics (replaces cAdvisor)
```promql
# CPU utilization
rate(container_cpu_usage_seconds_total{pod=~"mm1-server-.*"}[1m]) * 100

# Memory utilization
(container_memory_working_set_bytes{pod=~"mm1-server-.*"} / container_spec_memory_limit_bytes{pod=~"mm1-server-.*"}) * 100
```

## Configuration

### Adjust Service Time
Edit the `SERVICE_TIME_SECONDS` environment variable in `mm1-deployment.yaml`:

```yaml
env:
- name: SERVICE_TIME_SECONDS
  value: "6"  # Change this value
```

### Traffic Policies
Modify connection pooling and load balancing in `istio-config.yaml`:

```yaml
spec:
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100  # Adjust based on testing needs
```

## Access Points

- **Application**: `http://<GATEWAY_IP>/`
- **Prometheus**: Port-forward to access: `kubectl port-forward -n monitoring svc/prometheus 9090:9090`
- **Istio Dashboard**: `kubectl port-forward -n istio-system svc/kiali 20001:20001`

## Validation Commands

```bash
# Test M/M/1 endpoint
curl http://<GATEWAY_IP>/

# Check Istio proxy stats
kubectl exec -n spe-system deployment/mm1-server -c istio-proxy -- curl localhost:15000/stats

# View Prometheus targets
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets
```