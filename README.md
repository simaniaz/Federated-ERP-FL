Project Summary & Documentation: A Federated Learning Pipeline for Healthcare
This document provides a complete overview of the project, from its initial conception to its final implementation as a research-level federated learning simulator.

Phase 1: Foundation & Environment Setup
Objective: The initial goal was to build a professional and reproducible federated learning (FL) project simulating drug inventory management in hospitals, suitable for a PhD application.

Actions Taken:

Tool Selection: We chose Docker to create an isolated and consistent environment, ensuring that anyone (e.g., a PhD supervisor) could run the project without dependency issues.

Project Structure: A standard, modular directory structure (client, server, data, model, etc.) was created for clean, extensible code.

Dockerization: We wrote the necessary Dockerfile and docker-compose.yml files to build the Docker image and orchestrate the container.

Challenges Overcome: This phase was defined by significant technical challenges that required perseverance to solve:

Docker on Windows/WSL: We resolved configuration issues related to Docker Desktop, such as the dockerDesktopLinuxEngine not found error.

Python Versioning: We corrected execution errors in the WSL environment where the python command did not alias to python3.

pip Installation Errors: Due to Ubuntu's security restrictions, we encountered the externally-managed-environment error, which led us to adopt a venv (virtual environment) workflow.

venv Issues: Even with venv, we faced pathing and activation problems that caused packages to be installed in incorrect locations.

Jupyter Permissions: Finally, we resolved a file permission conflict between the Windows (NTFS) and Linux (WSL) filesystems that caused Jupyter Notebooks to open in "read-only" mode.

Outcome: Despite the challenges, we successfully built a robust and stable technical infrastructure that served as the foundation for all subsequent development.

Phase 2: Building the Core Federated Learning Pipeline
Objective: To construct a simple but functional FL system where multiple clients (hospitals) could collaboratively train a machine learning model without sharing raw data.

Actions Taken:

Initial Data Generation: We began by creating a synthetic, homogeneous dataset where the drug consumption patterns were similar across all three hospitals. This was a pragmatic decision to first validate the FL pipeline's functionality before introducing complexity.

Baseline Model (MLP): Our first model was a Multi-Layer Perceptron (MLP) implemented in PyTorch, directly addressing the PhD advertisement's requirement for PyTorch experience.

FL Communication with Flower:

client.py: Handled local model training at each hospital.

server.py: Aggregated model updates from clients using the FedAvg algorithm.

run_fl.py: Orchestrated the simultaneous execution of the server and all clients.

Challenges Overcome:

Model Heterogeneity: When adding an LSTM model, we discovered a key constraint of FedAvg: all clients must share the exact same model architecture. Attempting to run MLP and LSTM clients simultaneously resulted in a size mismatch error, which was an important learning experience.

Outcome: We produced a complete, working FL pipeline capable of training a PyTorch model in a federated manner. At this stage, it was a solid portfolio piece but had yet to meet the research objectives of the proposal.

Phase 3: Elevating the Project to a Research Level
Objective: To directly align the project with the key themes of the PhD proposal and the requirements of a high-level research project: data heterogeneity, privacy, and fairness.

Actions Taken:

Simulating Heterogeneity: We replaced the simple, homogeneous dataset with a more realistic, non-IID (non-identically and independently distributed) dataset. We created three distinct hospital profiles (general, oncology, and community clinic), each with a unique drug consumption pattern. This directly addressed one of the most significant challenges in FL mentioned in the proposal.

Implementing Privacy: We integrated a Differential Privacy (DP) layer using the Opacus library. By adding controlled noise to the gradients during training, this provides a mathematical guarantee of privacy for each hospital's data. This was a major step that made the project scientifically more valuable and directly addressed the "Privacy and Security" section of the proposal. The main challenge was the incompatibility of nn.LSTM with Opacus, which we solved by replacing it with DPLSTM.

Implementing Fairness: Finally, we implemented the FedProx algorithm. By adding a "proximal term" to the client's loss function, this algorithm encourages local models to stay closer to the global model. This improves training stability on heterogeneous data and ensures a fairer performance distribution across all clients, another key theme from the proposal.

Outcome: The project evolved from a simple simulator into an advanced research tool that models and implements solutions for three core, real-world challenges in Federated Learning.

Phase 4: Analysis and Final Product
Objective: To create an automated process for running experiments, collecting results, and visualizing them for comparative analysis.

Actions Taken:

Results Logging System: We enhanced server.py to intelligently log the metrics from each round (Loss, MAE, R²) into a single results_log.csv file, along with the model name and a timestamp.

Full Automation: The run_fl.py script was converted into a master orchestrator that not only runs all experiments (for both MLP and LSTM) sequentially but also automatically executes the analysis.ipynb notebook at the end to generate and save updated comparison plots.

Final Result: The outcome is a complete, automated, advanced, and highly presentable project. It is ready to be featured on a GitHub profile and referenced in a PhD application as a demonstration of both practical engineering skills and a deep understanding of complex machine learning concepts.

Technical Documentation: File & Class Breakdown
Here is a detailed breakdown of each critical component and its role within the pipeline.

run_fl.py
This is the main entry point and master orchestrator for the entire project.

Purpose: To run a complete set of experiments automatically. It sequentially trains different models (e.g., MLP, LSTM) across all clients and then triggers the final analysis and plotting.

run_single_experiment(model_name): This function manages a single federated learning run for a specified model.

It sets an environment variable (SELECTED_MODEL) to inform the server and clients which model to use.

It uses Python's subprocess module to launch the server.py and multiple client.py instances as independent processes, simulating a real-world distributed environment.

execute_analysis_notebook(): After all experiments are complete, this function is called.

It uses jupyter nbconvert to automatically execute the analysis.ipynb notebook. This ensures that the comparison plots are always up-to-date with the latest results from the results_log.csv file.

server.py
This script represents the central aggregation server in the federated learning setup.

Purpose: To coordinate the training rounds, aggregate model updates from clients, and log the performance of the global model.

CustomStrategy(fl.server.strategy.FedAvg): This class inherits from Flower's default FedAvg but extends its functionality.

aggregate_evaluate(...): This method is overridden to intercept the results after each evaluation round. It calculates the network-wide average for key metrics (loss, MAE, R²) and appends these metrics, along with the model name and a timestamp, to the results/results_log.csv file.

client.py
This script simulates the behavior of a single hospital (client).

DrugDemandDataset(Dataset): A custom PyTorch Dataset class that loads a hospital's specific CSV file, performs feature engineering (e.g., extracting day_of_week), and applies standard preprocessing like LabelEncoder and StandardScaler.

FlowerClient(fl.client.NumPyClient): The core of the client logic.

__init__(...): Initializes the model, optimizer, and loss function. It also attaches an Opacus PrivacyEngine to the model and optimizer to enable Differential Privacy.

fit(...): Defines the local training process. It implements the FedProx algorithm by adding a proximal term to the standard loss function, encouraging the local model to stay close to the global model. After training, it calculates and prints the spent privacy budget (epsilon).

evaluate(...): Evaluates the global model on the client's local data and returns the performance metrics to the server.

model/ Directory
This directory contains the definitions for the neural network architectures.

mlp.py: Defines the DrugDemandMLP class, a standard Multi-Layer Perceptron.

lstm.py: Defines the LSTMNet class. It uses an opacus.layers.DPLSTM layer, which is a drop-in replacement for nn.LSTM required for compatibility with the PrivacyEngine.

__init__.py: Makes the model directory a proper Python package.

Notebooks (.ipynb)
These files separate data generation and analysis from the core FL pipeline.

generate_heterogeneous_data.ipynb: Creates realistic, non-IID datasets by simulating three distinct hospital profiles (general, oncology, community) to test the robustness and fairness of the FL algorithms.

analysis.ipynb: Executed automatically by run_fl.py, this notebook reads the results_log.csv and uses seaborn to generate and save professional comparison plots visualizing the performance of the different models.