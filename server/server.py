# server.py
import flwr as fl
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

# Read the model name from the environment variable
MODEL_NAME = os.getenv("SELECTED_MODEL", "Unknown")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RESULTS_PATH = "results/results_log.csv"

# Create the results directory and log file if they don't exist
os.makedirs("results", exist_ok=True)
if not os.path.exists(RESULTS_PATH):
    pd.DataFrame(columns=["timestamp", "model_name", "round", "loss", "mae", "r2"]).to_csv(RESULTS_PATH, index=False)

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_loss is not None:
            mean_mae = sum(res[1].metrics["mae"] for res in results) / len(results)
            mean_r2 = sum(res[1].metrics["r2"] for res in results) / len(results)

            print(f"Model: {MODEL_NAME} | Round {server_round} | Loss: {aggregated_loss:.4f} | MAE: {mean_mae:.4f} | R²: {mean_r2:.4f}")

            # Save results to the CSV file
            new_row = pd.DataFrame([{
                "timestamp": TIMESTAMP,
                "model_name": MODEL_NAME,
                "round": server_round,
                "loss": aggregated_loss,
                "mae": mean_mae,
                "r2": mean_r2
            }])
            new_row.to_csv(RESULTS_PATH, mode='a', header=False, index=False)

        return aggregated_loss, {}

strategy = CustomStrategy(
    fraction_fit=1.0, fraction_evaluate=1.0,
    min_fit_clients=3, min_available_clients=3
)

def main():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10), # Increased number of rounds for better learning
        strategy=strategy,
    )

if __name__ == "__main__":
    print(f"Starting server for model: {MODEL_NAME}")
    main()