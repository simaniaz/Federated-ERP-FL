import subprocess
import time
import sys
import os

def run_single_experiment(model_name):
    print(f"\n--- Starting experiment for model: {model_name} ---")
    
    PYTHON = sys.executable
    env = os.environ.copy()
    env["SELECTED_MODEL"] = model_name

    server_proc = subprocess.Popen([PYTHON, "server/server.py"], env=env)
    print("Server process started.")
    time.sleep(5) 

    # --- Use the new, full hospital names ---
    client_ids = ["Hosp_A_General", "Hosp_B_Oncology", "Hosp_C_Community"]
    # ----------------------------------------
    
    client_procs = []
    for client_id in client_ids:
        print(f"Starting client {client_id} with model {model_name}")
        proc = subprocess.Popen([PYTHON, "client/client.py", client_id, model_name])
        client_procs.append(proc)
        time.sleep(2)

    for proc in client_procs:
        proc.wait()
        
    server_proc.wait()
    print(f"--- Experiment for model: {model_name} is complete ---")

def execute_analysis_notebook():
    print("\n--- Executing analysis notebook to generate plots ---")
    
    notebook_path = "notebooks/analysis.ipynb"
    
    # Use jupyter nbconvert to execute the notebook
    # This will run the notebook and save it with the output
    command = [
        "jupyter", "nbconvert", 
        "--to", "notebook", 
        "--execute", notebook_path,
        "--inplace" # Overwrite the notebook with the one containing output
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Analysis notebook executed successfully.")
        print("Plots have been updated in the 'results/' folder.")
    else:
        print("--- Error executing analysis notebook ---")
        print(result.stderr)

if __name__ == "__main__":
    models_to_run = ["MLP", "LSTM"]
    
    for model in models_to_run:
        run_single_experiment(model)
        
    execute_analysis_notebook()