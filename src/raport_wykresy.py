import os
import math
import matplotlib.pyplot as plt
import numpy as np
from file_manager import FileManager

STATS_DIR = os.path.join('.', 'statystyki')
DANE_DIR = os.path.join('.', 'dane')

def main():
    activations = ['sigmoid', 'tanh', 'relu']
    colors = {'sigmoid': 'red', 'tanh': 'blue', 'relu': 'green'}
    
    baseline_data = FileManager.load_json(STATS_DIR, "baseline_info.json")
    baseline_val = baseline_data["baseline_mse"]
    
    plt.figure(figsize=(10, 5))
    max_epochs_1 = 0
    
    for act in activations:
        history_file = os.path.join(STATS_DIR, f"train_history_{act}.json")
        if not os.path.exists(history_file):
            continue
        history = FileManager.load_json(STATS_DIR, f"train_history_{act}.json")
        plt.plot(history, label=f"Sieć: {act.upper()}", color=colors[act])
        
        y_last = history[-1]
        x_last = len(history) - 1
        max_epochs_1 = max(max_epochs_1, len(history))
        
        plt.plot(x_last, y_last, marker='o', color=colors[act])
        plt.text(x_last + (max_epochs_1 * 0.015), y_last, f"{y_last:.6f}", color=colors[act], fontweight='bold', va='center')
        
    plt.title("Błąd MSE na zbiorze uczącym")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    # plt.yscale('log')
    plt.ylim(bottom=1e-4, top=baseline_val) 
    plt.xlim(left=0, right=max_epochs_1 * 1.12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(STATS_DIR, "wykres1_mse_train.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    max_epochs_2 = 0
    
    for act in activations:
        history_test_file = os.path.join(STATS_DIR, f"test_history_{act}.json")
        if not os.path.exists(history_test_file):
            continue
        history_test = FileManager.load_json(STATS_DIR, f"test_history_{act}.json")
        plt.plot(history_test, label=f"Sieć: {act.upper()}", color=colors[act])
        
        y_last = history_test[-1]
        x_last = len(history_test) - 1
        max_epochs_2 = max(max_epochs_2, len(history_test))
        
        plt.plot(x_last, y_last, marker='o', color=colors[act])
        plt.text(x_last + (max_epochs_2 * 0.015), y_last, f"{y_last:.6f}", color=colors[act], fontweight='bold', va='center')
        
    plt.axhline(y=baseline_val, color='black', linestyle='--', linewidth=2, label='Zmierzone UWB')
    plt.text(max_epochs_2, baseline_val * 0.9, f"Ref: {baseline_val:.6f}", color='black', fontweight='bold', va='top', ha='right')
    
    plt.title("Błąd MSE na zbiorze testowym")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    # plt.yscale('log')
    plt.ylim(bottom=1e-4, top=baseline_val+0.001) 
    plt.xlim(left=0, right=max_epochs_2 * 1.12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(STATS_DIR, "wykres2_mse_test.png"))
    plt.close()

    _, _, raw_test_inputs, raw_test_targets = FileManager.prepare_datasets(DANE_DIR)
    
    def calc_errors_mm(preds, reals):
        return [math.sqrt((preds[i][0] - reals[i][0])**2 + (preds[i][1] - reals[i][1])**2) for i in range(len(preds))]

    plt.figure(figsize=(10, 5))
    raw_errors = calc_errors_mm(raw_test_inputs, raw_test_targets)
    raw_errors.sort()
    y_vals = np.arange(1, len(raw_errors)+1) / len(raw_errors)
    plt.plot(raw_errors, y_vals, label="Surowe Pomiary UWB", color='black', linestyle='--')
    
    for act in activations:
        pred_file = os.path.join(STATS_DIR, f"predictions_{act}.json")
        if not os.path.exists(pred_file):
            continue
        preds = FileManager.load_json(STATS_DIR, f"predictions_{act}.json")
        net_errors = calc_errors_mm(preds, raw_test_targets)
        net_errors.sort()
        plt.plot(net_errors, y_vals, label=f"Skorygowane ({act.upper()})", color=colors[act])
        
    plt.title("Dystrybuanta błędów w pomiarach dynamicznych")
    plt.xlabel("Błąd odległości [mm]")
    plt.ylabel("Prawdopodobieństwo P(X <= x)")
    # plt.xscale('log')
    # plt.xlim(left=1, right=1000) 
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(STATS_DIR, "wykres3_cdf.png"))
    plt.close()

    global_best_info = FileManager.load_json(STATS_DIR, "global_best.json")
    best_act = global_best_info["best_activation"]
    best_preds = FileManager.load_json(STATS_DIR, f"predictions_{best_act}.json")
    
    real_x = [pt[0] for pt in raw_test_targets]
    real_y = [pt[1] for pt in raw_test_targets]
    raw_x = [pt[0] for pt in raw_test_inputs]
    raw_y = [pt[1] for pt in raw_test_inputs]
    corr_x = [pt[0] for pt in best_preds]
    corr_y = [pt[1] for pt in best_preds]

    plt.figure(figsize=(12, 8))
    plt.scatter(raw_x, raw_y, color='red', alpha=0.3, label="Zmierzone (Raw)", zorder=1)
    plt.scatter(corr_x, corr_y, color='blue', alpha=0.6, label=f"Skorygowane ({best_act.upper()})", zorder=2)
    plt.scatter(real_x, real_y, color='black', alpha=1.0, label="Rzeczywiste (Real)", zorder=3, marker='x')
    
    plt.title(f"Skorygowane wyniki (BEST: {best_act.upper()})")
    plt.xlabel("Współrzędna X [mm]")
    plt.ylabel("Współrzędna Y [mm]")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(STATS_DIR, "wykres4_scatter.png"))
    plt.close()

if __name__ == '__main__':
    main()