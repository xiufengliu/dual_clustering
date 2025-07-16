

import json
import numpy as np

file_path = "/work3/xiuli/dual_clustering/results/optimized/comprehensive_evaluation_20250715_204051.json"

with open(file_path, 'r') as f:
    data = json.load(f)

main_results = data['main_results']
ablation_studies = data['ablation_studies']

# Function to safely get metrics, returning None if not found or failed
def get_metrics(results, model_name):
    if model_name in results and not results[model_name].get('failed', False):
        return results[model_name].get('point_metrics'), results[model_name].get('interval_metrics')
    return None, None

# Prepare data for LaTeX tables
latex_data = {
    "solar": {},
    "wind": {},
    "pi": {},
    "ablation": {}
}

# Process main results for solar and wind
for dataset_key, dataset_results in main_results.items():
    if dataset_key == 'kaggle_solar_plant':
        target_data = latex_data['solar']
        pi_target = latex_data['pi']
    elif dataset_key == 'nrel_canada_wind':
        target_data = latex_data['wind']
        pi_target = latex_data['pi']
    else:
        continue

    for model_name, metrics_data in dataset_results['model_results'].items():
        if metrics_data.get('failed', False):
            target_data[model_name] = {"rmse": "N/A", "mae": "N/A"}
            if model_name == 'NDC-RF':
                pi_target[dataset_key.split('_')[0]] = {"PICP": "N/A", "MPIW": "N/A"}
        else:
            point_metrics = metrics_data.get('point_metrics', {})
            target_data[model_name] = {
                "rmse": point_metrics.get('rmse'),
                "mae": point_metrics.get('mae')
            }
            if model_name == 'NDC-RF':
                interval_metrics = metrics_data.get('interval_metrics', {})
                picp = interval_metrics.get('picp', "N/A")
                pinaw = interval_metrics.get('pinaw', "N/A")
                if picp != "N/A":
                    picp *= 100
                pi_target[dataset_key.split('_')[0]] = {"PICP": f"{picp:.2f}" if picp != "N/A" else "N/A", "MPIW": f"{pinaw:.2f}" if pinaw != "N/A" else "N/A"}

# Process ablation results
for config_name, metrics_data in ablation_studies.items():
    if metrics_data.get('failed', False):
        latex_data['ablation'][config_name] = {"rmse": "N/A", "mae": "N/A"}
    else:
        point_metrics = metrics_data.get('point_metrics', {})
        latex_data['ablation'][config_name] = {
            "rmse": point_metrics.get('rmse'),
            "mae": point_metrics.get('mae')
        }

# Calculate percentage improvements for solar
best_solar_baseline_rmse = np.inf
best_solar_baseline_mae = np.inf
for model_name, metrics in latex_data['solar'].items():
    if model_name != 'NDC-RF' and metrics['rmse'] != "N/A":
        if metrics['rmse'] < best_solar_baseline_rmse:
            best_solar_baseline_rmse = metrics['rmse']
        if metrics['mae'] < best_solar_baseline_mae:
            best_solar_baseline_mae = metrics['mae']

proposed_solar_rmse = latex_data['solar']['NDC-RF']['rmse']
proposed_solar_mae = latex_data['solar']['NDC-RF']['mae']

solar_rmse_impr = "-"
solar_mae_impr = "-"
if proposed_solar_rmse != "N/A" and best_solar_baseline_rmse != np.inf:
    solar_rmse_impr = ((best_solar_baseline_rmse - proposed_solar_rmse) / best_solar_baseline_rmse) * 100
    solar_mae_impr = ((best_solar_baseline_mae - proposed_solar_mae) / best_solar_baseline_mae) * 100

latex_data['solar']['NDC-RF']['rmse_impr'] = solar_rmse_impr
latex_data['solar']['NDC-RF']['mae_impr'] = solar_mae_impr

# Calculate percentage improvements for wind
best_wind_baseline_rmse = np.inf
best_wind_baseline_mae = np.inf
for model_name, metrics in latex_data['wind'].items():
    if model_name != 'NDC-RF' and metrics['rmse'] != "N/A":
        if metrics['rmse'] < best_wind_baseline_rmse:
            best_wind_baseline_rmse = metrics['rmse']
        if metrics['mae'] < best_wind_baseline_mae:
            best_wind_baseline_mae = metrics['mae']

proposed_wind_rmse = latex_data['wind']['NDC-RF']['rmse']
proposed_wind_mae = latex_data['wind']['NDC-RF']['mae']

wind_rmse_impr = "-"
wind_mae_impr = "-"
if proposed_wind_rmse != "N/A" and best_wind_baseline_rmse != np.inf:
    wind_rmse_impr = ((best_wind_baseline_rmse - proposed_wind_rmse) / best_wind_baseline_rmse) * 100
    wind_mae_impr = ((best_wind_baseline_mae - proposed_wind_mae) / best_wind_baseline_mae) * 100

latex_data['wind']['NDC-RF']['rmse_impr'] = wind_rmse_impr
latex_data['wind']['NDC-RF']['mae_impr'] = wind_mae_impr

# Format for LaTeX
def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def format_row(model, rmse, mae, rmse_impr=None, mae_impr=None):
    rmse_str = format_value(rmse)
    mae_str = format_value(mae)
    rmse_impr_str = format_value(rmse_impr)
    mae_impr_str = format_value(mae_impr)

    if rmse_impr is not None and mae_impr is not None and rmse_impr != "-":
        return f"        \\textbf{{{model}}} & \\textbf{{{rmse_str}}} & \\textbf{{{mae_str}}} & \\textbf{{{rmse_impr_str}}} & \\textbf{{{mae_impr_str}}} \\ \\hline"
    else:
        return f"        {model} & {rmse_str} & {mae_str} & - & - \\ \\hline"

solar_table_rows = []
# Define a consistent order for models as they appear in the paper's table
model_order_solar = ["Prophet", "Naive", "SES", "BATS", "arima", "SARIMA", "svr", "mlp", "vanilla_rf", "lightgbm", "lstm", "cnn_lstm", "NDC-RF"]
for model_name in model_order_solar:
    if model_name in latex_data['solar']:
        metrics = latex_data['solar'][model_name]
        display_name = model_name.replace("_", " ").title()
        if display_name == "Arima": display_name = "ARIMA"
        if display_name == "Sarima": display_name = "SARIMA"
        if display_name == "Svr": display_name = "SVM" # SVR is SVM in the paper
        if display_name == "Mlp": display_name = "MLP"
        if display_name == "Vanilla Rf": display_name = "Vanilla RF"
        if display_name == "Lightgbm": display_name = "LightGBM"
        if display_name == "Lstm": display_name = "LSTM"
        if display_name == "Cnn Lstm": display_name = "CNN-LSTM"

        if model_name == "NDC-RF":
            solar_table_rows.append(format_row("Neutrosophic (Proposed)", metrics['rmse'], metrics['mae'], metrics['rmse_impr'], metrics['mae_impr']))
        else:
            solar_table_rows.append(format_row(display_name, metrics['rmse'], metrics['mae']))

wind_table_rows = []
model_order_wind = ["Prophet", "Naive", "SES", "BATS", "arima", "SARIMA", "svr", "mlp", "vanilla_rf", "lightgbm", "lstm", "cnn_lstm", "nbeats", "NDC-RF"]
for model_name in model_order_wind:
    if model_name in latex_data['wind']:
        metrics = latex_data['wind'][model_name]
        display_name = model_name.replace("_", " ").title()
        if display_name == "Arima": display_name = "ARIMA"
        if display_name == "Sarima": display_name = "SARIMA"
        if display_name == "Svr": display_name = "SVM"
        if display_name == "Mlp": display_name = "MLP"
        if display_name == "Vanilla Rf": display_name = "Vanilla RF"
        if display_name == "Lightgbm": display_name = "LightGBM"
        if display_name == "Lstm": display_name = "LSTM"
        if display_name == "Cnn Lstm": display_name = "CNN-LSTM"
        if display_name == "Nbeats": display_name = "N-BEATS"

        if model_name == "NDC-RF":
            wind_table_rows.append(format_row("Neutrosophic (Proposed)", metrics['rmse'], metrics['mae'], metrics['rmse_impr'], metrics['mae_impr']))
        else:
            wind_table_rows.append(format_row(display_name, metrics['rmse'], metrics['mae']))

ablation_table_rows = []
# Define a consistent order for ablation configurations
ablation_order = [
    "full_model",
    "kmeans_only", # Corresponds to "Without FCM"
    "without_neutrosophic", # Corresponds to "Without Neutrosophic Transformation"
    "linear_model", # Corresponds to "Using Linear Regression"
    # "fcm_only", # Not present in the provided JSON
    # "without_indeterminacy", # Not present in the provided JSON
    # "distance_indeterminacy", # Not present in the provided JSON
    # "n_clusters_3", # Not present in the provided JSON
    # "n_clusters_7"  # Not present in the provided JSON
]

for config_name in ablation_order:
    if config_name in latex_data['ablation']:
        metrics = latex_data['ablation'][config_name]
        rmse_str = format_value(metrics['rmse'])
        mae_str = format_value(metrics['mae'])

        if config_name == "full_model":
            ablation_table_rows.append(f"Full Framework (Proposed) & \\textbf{{{rmse_str}}} & \\textbf{{{mae_str}}} \\ \\hline")
        elif config_name == "kmeans_only":
            ablation_table_rows.append(f"Without FCM (K-Means only + adapted Neutro) & {rmse_str} & {mae_str} \\ \\hline")
        elif config_name == "without_neutrosophic":
            ablation_table_rows.append(f"Without Neutrosophic Transformation (Cluster features only) & {rmse_str} & {mae_str} \\ \\hline")
        elif config_name == "linear_model":
            ablation_table_rows.append(f"Using Linear Regression (instead of RF) & {rmse_str} & {mae_str} \\ \\hline")


pi_table_rows = [
    f"        Solar Power & {latex_data['pi']['kaggle']['PICP']} & {latex_data['pi']['kaggle']['MPIW']} \\",
    f"        Wind Power & {latex_data['pi']['nrel']['PICP']} & {latex_data['pi']['nrel']['MPIW']} \\"
]

print("SOLAR_TABLE_ROWS_START")
for row in solar_table_rows:
    print(row)
print("SOLAR_TABLE_ROWS_END")

print("WIND_TABLE_ROWS_START")
for row in wind_table_rows:
    print(row)
print("WIND_TABLE_ROWS_END")

print("ABLATION_TABLE_ROWS_START")
for row in ablation_table_rows:
    print(row)
print("ABLATION_TABLE_ROWS_END")

print("PI_TABLE_ROWS_START")
for row in pi_table_rows:
    print(row)
print("PI_TABLE_ROWS_END")
