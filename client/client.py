# client.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import flwr as fl
from opacus import PrivacyEngine
import copy # برای کپی کردن مدل جهانی

from model.lstm import LSTMNet
from model import DrugDemandMLP

# ... کلاس DrugDemandDataset بدون تغییر باقی می‌ماند ...
class DrugDemandDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        all_drugs = ['Paracetamol', 'Ibuprofen', 'Amoxicillin', 'Metformin', 'Atorvastatin', 'Letrozole', 'Trastuzumab', 'Oseltamivir', 'Cough Syrup']
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        le_drug = LabelEncoder()
        le_drug.fit(all_drugs)
        df['drug_encoded'] = le_drug.transform(df['drug_name'])
        df = df.drop(columns=['date', 'hospital_id', 'drug_name'])
        X = df.drop(columns=['demand']).values
        y = df['demand'].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, model_type="MLP"):
        self.model = model
        self.train_loader = train_loader
        self.model_type = model_type
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # اتصال موتور حریم خصوصی
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )
        self.privacy_engine = privacy_engine

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model._module.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model._module.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model._module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1. دریافت مدل جهانی و ذخیره یک کپی از آن
        self.set_parameters(parameters)
        global_model = copy.deepcopy(self.model)

        self.model.train()
        for epoch in range(5):
            for X_batch, y_batch in self.train_loader:
                if self.model_type == "LSTM":
                    X_batch = X_batch.unsqueeze(1)
                
                # محاسبه loss اصلی
                preds = self.model(X_batch)
                loss = self.loss_fn(preds, y_batch)

                # 2. محاسبه و افزودن Proximal Term مربوط به FedProx
                proximal_term = 0.0
                mu = 0.01 # هایپرپارامتر FedProx
                for local_param, global_param in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += torch.sum(torch.pow(local_param - global_param, 2))
                
                loss += (mu / 2) * proximal_term

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"Privacy budget spent: Epsilon = {epsilon:.2f}")

        return self.get_parameters(), len(self.train_loader.dataset), {}

    # ... متد evaluate بدون تغییر باقی می‌ماند ...
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                if self.model_type == "LSTM":
                    X_batch = X_batch.unsqueeze(1)
                preds = self.model(X_batch)
                y_true.extend(y_batch.numpy().tolist())
                y_pred.extend(preds.numpy().tolist())
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
        return float(mse), len(self.train_loader.dataset), {"mae": mae, "r2": r2}

# ... بخش Main execution بدون تغییر باقی می‌ماند ...
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id> [MLP|LSTM]")
        sys.exit(1)
    client_id = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "MLP"
    csv_path = os.path.join(os.path.dirname(__file__), f"../data/inventory_{client_id}.csv")
    dataset = DrugDemandDataset(csv_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    input_dim = dataset.X.shape[1]
    if model_type.upper() == "LSTM":
        model = LSTMNet(input_size=input_dim)
    else:
        model = DrugDemandMLP(input_dim=input_dim)
    client = FlowerClient(model, train_loader, model_type).to_client()
    fl.client.start_client(server_address="0.0.0.0:8080", client=client)