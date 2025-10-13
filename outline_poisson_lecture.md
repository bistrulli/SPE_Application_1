# Poisson Process in Software Performance Engineering
## Hands-on Lecture

---

### **Learning Objectives**
- Implement a **Poisson workload generator**
- Validate **statistical properties** of the process
- Verify **match with M/M/1 formulas** through practical experiments

---

### **Lecture Program**

#### **Module 1: Review and Workload Generator**
1. **Essential Review**
   - Parameter λ and practical meaning
   - Exponential inter-arrival times
   - Connection with M/M/1 systems

2. **Environment Setup**
   - Required Python libraries
   - Notebook structure

3. **Poisson Workload Generator**
   - Generator implementation
   - Inter-arrival times visualization
   - Exponential distribution verification
   - Actual vs theoretical rate validation

---

### **BREAK**

---

#### **Module 2: M/M/1 System Application**
4. **M/M/1 System as Target**
   - M/M/1 "black box" system provided
   - Request submission with controlled λ
   - Metrics collection: throughput, response time, queue length

5. **Theoretical-Practical Validation**
   - Comparison λ_generated vs λ_observed
   - Match with M/M/1 formulas: ρ = λ/μ, E[T], E[N]
   - Experiments with different λ values
   - Interpretation of deviations and practical limits

---

### **Tools Used**
- **Python**: numpy, matplotlib, scipy
- **Jupyter Notebook**: interactive environment
- **M/M/1 System**: implementation provided by instructor

---

### **Prerequisites**
- Theoretical knowledge of Poisson process
- Familiarity with M/M/1 system and related formulas
- Basic Python programming skills

---

*Software Performance Engineering - Software: Science and Technology - A.Y. 2025/2026*