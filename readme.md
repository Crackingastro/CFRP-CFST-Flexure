# Physics-Informed Data Augmentation for Explainable ML-Based Multi-Objective Optimization in Flexural Design of CFRP-Strengthened CFST Beams

<p align="center">
  📄 <a href="" target="_blank">Paper</a> &nbsp; | &nbsp;
  🌐 <a href="https://huggingface.co/datasets/crackingastro/CFRP-strengthened-CFST-Under-flexure" target="_blank">Dataset</a> &nbsp; | &nbsp;
  🖥️ <a href="https://explainable-ml-based-moo-in-cfrp-strengthened-cfst-beam-designs.streamlit.app/" target="_blank">Website</a>
</p>

This repository contains the **computational framework and data** supporting the research article:
- "Physics-Informed Data Augmentation for Explainable ML-Based Multi-Objective Optimization in Flexural Design of CFRP-Strengthened CFST Beams"  
- *Muluken Bogale, Addisu Mengistu, Tariku Habtamu*  
- *Structures*, Elsevier (2025).  
The framework integrates a Physics-Informed Tabular Variational Autoencoder (PI-TVAE) for data augmentation with an **explainable ensemble Machine Learning** model for flexural strength prediction and multi-objective optimization with Monte Carlo uncertainty quantification.

## Repository Contents

1.streamlit_app
A deployment-ready Streamlit interface that provides:
- flexural strength prediction using trained ensemble ML models  
- visualization of prediction outputs
- Monte Carlo-based uncertainty simulation 
- Basic CFRP configuration exploration  

Quick Start
Prerequisites

-Python 3.8+
```bash
# Create and activate virtual environment
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
````

2.trained_models
Contains the saved ensemble ML models used in the manuscript:
- XGBoost
- Random Forest
- GraBoost
- AdaBoost
- Stacking Regressor
These .pkl files are directly used by the Streamlit interface to compute:
- flexural strength predictions
- optimization objectives
- Monte Carlo uncertainty estimations
This allows complete reproducibility without requiring retraining.

3.MOO
Scripts for performing:
- Bayesian optimization with Tree-structured Parzen Estimator (TPE) 
- multi-objective optimization for 
- maximizing flexural strength 
- minimizing CFRP usage 
- minimizing constructability demand
- Extraction and visualization of Pareto-optimal solutions 
 
4.sample_data
Contains experimental data used in the manuscript figures, validation examples, and demonstration of the workflow.
These datasets are sufficient for running the provided code and reproducing core results.

# About the PI-TVAE Code
The complete Physics-Informed Tabular VAE (PI-TVAE) implementation includes customized loss functions, constraint encoding, and internal architecture that constitute original methodological contributions.
To protect ongoing research and intellectual property:
- The full implementation is not included in this public repository.
- A lightweight demonstration version may be included for illustrative purposes.
- The full model is available upon reasonable academic request, subject to research use only.

## Contact
For access to the full PI-TVAE code, collaboration opportunities, or technical inquiries:

| Name                | Email                                                         | ORCID               |
| ------------------- | ------------------------------------------------------------- | ------------------- |
| **Muluken Bogale**  | [mulukenbogale67@gmail.com](mailto:mulukenbogale67@gmail.com) | 0009-0001-6640-4513 |
| **Addisu Mengistu** | [addisum443@gmail.com](mailto:addisum443@gmail.com)           | 0009-0004-1248-3422 |
