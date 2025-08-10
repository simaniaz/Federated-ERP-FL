Federated Learning for Hospital Drug Inventory Management
This project is a research-level simulator for a federated learning (FL) pipeline designed to address the challenge of drug demand forecasting in a multi-hospital healthcare system. The primary goal is to collaboratively train accurate machine learning models without sharing sensitive, private patient data, directly addressing regulations like GDPR and HIPAA.

The pipeline simulates a network of hospitals (clients) and a central server. It trains both MLP and LSTM models on realistic, heterogeneous (non-IID) data while incorporating advanced techniques like Differential Privacy (via Opacus) and Fairness algorithms (via FedProx).

Visual Results: Model Performance Comparison
The following plots show a comparison between the MLP and LSTM models trained over 10 federated rounds on the heterogeneous dataset.

Loss (MSE) per Round
Mean Absolute Error (MAE) per Round
R² Score per Round
Key Features

<img width="200" height="200" alt="comparison_loss" src="https://github.com/user-attachments/assets/6492f659-571c-49af-8692-db3a73d82966" />
<img width="200" height="200" alt="comparison_mae" src="https://github.com/user-attachments/assets/6e065ac1-06df-4c27-a27c-275ea364d8ff" />
<img width="200" height="200" alt="comparison_r2" src="https://github.com/user-attachments/assets/47bc9bf9-3cc1-45a5-bfd7-8a8bb515b4c3" />



This project demonstrates a comprehensive understanding of modern federated learning challenges and solutions:

Statistical Heterogeneity: The models are trained on non-IID data, simulating realistic scenarios where different hospitals (e.g., general, oncology, community) have unique drug demand patterns.

Differential Privacy: Implements a strong privacy-preserving layer using PyTorch Opacus. This adds statistical noise during client-side training to provide mathematical guarantees against data leakage.

Fairness Algorithm: Uses the FedProx algorithm to ensure more stable and fair performance across all diverse clients, preventing the global model from being biased towards larger institutions.

Multiple Model Support: The pipeline is architected to support and compare different model types (MLP and LSTM) within the same federated learning framework.

Fully Automated Pipeline: A master script (run_fl.py) orchestrates the entire workflow, including running experiments for all models and automatically generating updated analysis plots.

How to Run
This project uses a local Python virtual environment.

Clone the repository:

Bash

git clone https://github.com/simaniaz/Federated-ERP-FL.git
cd Federated-ERP-FL
Create and activate the virtual environment (using Python 3.11):

Bash

python3.11 -m venv venv
source venv/bin/activate
Install dependencies:

Bash

pip install -r requirements.txt
pip install flwr-xgboost
Run the complete experiment pipeline:

Bash

python run_fl.py
This single command will run experiments for both MLP and LSTM models, save the results to results/results_log.csv, and automatically generate updated plots in the results/ folder.

Technologies Used
Python 3.11

PyTorch: For building and training neural network models.

Flower (flwr): As the core framework for the federated learning communication.

Opacus: For implementing Differential Privacy.

Pandas & NumPy: For data manipulation and processing.

Matplotlib & Seaborn: For data visualization and plotting results.

Scikit-learn: For data preprocessing and evaluation metrics.
