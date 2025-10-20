# SPE Application: Poisson Process & M/M/1 Validation Platform

Educational platform for hands-on Systems Performance Engineering (SPE) combining theoretical foundations with practical M/M/1 server validation. Students learn Poisson processes through interactive experiments and real system measurements.

## Architecture
- **Docker Infrastructure**: M/M/1 server with Envoy proxy, Prometheus monitoring, cAdvisor metrics
- **Interactive Modules**: Three progressive Jupyter notebooks
- **Workload Generators**: Synchronous and asynchronous request generators
- **Metrics Collection**: Automated Prometheus-based performance monitoring

## Modules
1. **Module 1** (`poisson_spe_lecture.ipynb`): Poisson theory, statistical validation, exponential distributions
2. **Module 2** (`module2_workload_patterns.ipynb`): Open vs closed workload patterns, synchronous vs asynchronous generators
3. **Module 3** (`module3_mm1_validation.ipynb`): M/M/1 theoretical validation using calibration-based service rate estimation

## Core Components
- `workload_generator.py`: Synchronous and asynchronous workload generators
- `metrics_collector.py`: Prometheus integration with automatic container discovery
- `poisson_plots.py`: Statistical validation and visualization utilities
- `workload_analysis_plots.py`: Workload pattern analysis and comparison plots
- `docker-compose.yml`: Complete monitoring stack (Envoy, Prometheus, cAdvisor)
 - `docker-stack.yml`: Swarm stack file (same services; `mm1-server` scalable)
- `envoy-config-swarm.yaml`: Envoy configuration for Docker Swarm (uses `spe_mm1-server` service name)

## Quick Start
1. **Setup environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start M/M/1 system**:
   ```bash
   docker-compose up -d
   ```

3. **Run modules**:
   ```bash
   jupyter notebook
   # Open poisson_spe_lecture.ipynb and run sequentially
   ```

## System Requirements
- Docker and docker-compose
- Python 3.8+ with virtual environment
- Ports: 8084 (Envoy), 9090 (Prometheus), 8081 (cAdvisor), 9901 (Envoy admin)

## Usage Notes
- Modules build progressively: theory → workload patterns → system validation
- Real M/M/1 server allows comparison between theoretical predictions and measured performance
- Automatic container discovery handles service restarts transparently

## Docker Swarm Stack (Scaling Experiments)

This project also supports Docker Swarm to easily scale the `mm1-server` replicas without changing CPU core limits.

### Prerequisites
- Docker Engine with Swarm mode enabled
- Initialize Swarm (first time only):
  ```bash
  docker swarm init
  ```

### Deploy the stack
```bash
docker stack deploy -c docker-stack.yml spe
```

### Inspect services
```bash
docker service ls
docker service ps spe_mm1-server
```

### Scale the M/M/1 server
Change replica count at runtime (e.g., 3 replicas):
```bash
docker service scale spe_mm1-server=3
```

You can also set the initial replica count by editing `deploy.replicas` under `mm1-server` in `docker-stack.yml`.

### Logs and troubleshooting
```bash
# Envoy access/admin
docker service logs -f spe_envoy-proxy

# M/M/1 server instances (all tasks)
docker service logs -f spe_mm1-server

# Prometheus
docker service logs -f spe_prometheus

# cAdvisor (node-level permissions may vary)
docker service logs -f spe_cadvisor
```

### Remove the stack
```bash
docker stack rm spe
```

### Notes
- In Swarm, `deploy` keys are honored (unlike plain `docker compose up`).
- `depends_on` does not enforce start order in Swarm; components retry automatically.
- cAdvisor may require privileges that differ per host; running it as a standalone container per node can be more robust if issues arise.
