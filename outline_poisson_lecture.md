# Poisson Process & M/M/1 Validation
## First Hands-on SPE Lecture

---

### **Learning Objectives**
- Master **Poisson process** theory through statistical validation
- Understand **open vs closed workload** patterns
- Validate **M/M/1 theoretical predictions** against real system measurements

---

### **Module 1: Poisson Theory & Statistical Validation**
1. **Recall on Formal Definition**
   - Transition probability
   - Connection to exponential inter-arrivals

2. **Statistical Validation**
   - Poisson count distribution verification
   - Exponential inter-arrival time validation
   - Kolmogorov-Smirnov testing
   - Coefficient of variation analysis

---

### **BREAK** (15 min)

---

### **Module 2: Workload Pattern Analysis**
3. **Open vs Closed Workloads**
   - Synchronous generator (closed workload behavior)
   - Asynchronous generator (true open workload)
   - Inter-arrival time measurement methodology

4. **Practical Experiments**
   - Low service time: exponential patterns emerge
   - High service time: synchronous shows non-exponential behavior
   - Asynchronous maintains exponential regardless of service time

---

### **Module 3: M/M/1 System Validation**
5. **Real System Integration**
   - Docker-based M/M/1 server with monitoring stack
   - Prometheus metrics collection (Envoy + cAdvisor)
   - Automatic container discovery

6. **Calibration-Based Validation**
   - Service rate estimation
   - Systematic validation across utilization levels 
   - Theoretical vs measured comparison

---

### **Technical Stack**
- **Infrastructure**: Docker Compose, Envoy proxy, Prometheus, cAdvisor
- **Analysis**: Python (numpy, scipy, pandas, matplotlib)
- **Environment**: Jupyter notebooks with modular utilities

### **Prerequisites**
- Basic Poisson process knowledge
- M/M/1 queueing theory fundamentals
- Python programming and Docker basics

---

*Software Performance Engineering - A.Y. 2025/2026*