# Fitness and Overfitness: Implicit Regularization in Evolutionary Dynamics

This repository contains code to reproduce simulations and generate plots from the manuscript:

**Fitness and Overfitness: Implicit Regularization in Evolutionary Dynamics**  
Hagai Rappeport and Mor Nitzan  


---

## Overview

This project implements a computational framework to study the evolution of biological complexity through the lens of **implicit regularization** in evolutionary dynamics. It leverages the mathematical analogy between the replicator equation and Bayesian inference to explore how organismal complexity evolves to match environmental complexity.


---

## Repository Contents

- `bayesian_evolution.py`  
  Contains the full implementation of the evolutionary simulation framework, including:  
  - Definition of various genotype-to-phenotype mappings of different complexity (currently supported linear functions, polynomials and neural networks)  
  - Environmental complexity modeling  
  - Replicator dynamics simulation  
  - Fitness calculations 
  - Plotting routines to generate figures in the style of those in the paper

---

## Requirements

- Python 3.8+  
- NumPy  
- Matplotlib  
- SciPy  


## License
This project is released under the MIT License. See LICENSE for details.

## Contact
For questions or collaboration inquiries, please contact:

Hagai Rappeport — [hagai.rappeport@huji.mail.ac.il]
Mor Nitzan — [mor.nitzan@huji.mail.ac.il]

