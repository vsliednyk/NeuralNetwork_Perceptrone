import itertools
import os
import concurrent.futures
import json
from file_manager import FileManager
from scaler import MinMaxScaler
from neural_network import NeuralNetwork

# Dolaczanie sciezek
DANE_DIR = os.path.join('.', 'dane')
KNOWLEDGE_DIR = os.path.join('.', 'knowledge')
STATS_DIR = os.path.join('.', 'statystyki')
CONFIG_FILE = 'config.json'
# funkcja do generowania wszystkich przxeplotow konfiguracji
def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"[BLAD] Brak pliku {CONFIG_FILE}. Uruchom najpierw: python configurator.py")
        exit(1)
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_configs(config_space):
    processed_space = {k: (v if isinstance(v, list) else [v]) for k, v in config_space.items()}
    keys = processed_space.keys()
    values = processed_space.values()
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def train_single_variant(config, train_inputs, train_targets, test_inputs, test_targets):
    variant_name = f"{config['activation']}_H{config['hidden_amount']}_E{config['epochs']}_T{config['trial']}"
    weight_range = config.get('weight_range', 0.5)
    
    network = NeuralNetwork(
        in_amount=2, hidden_amount=config['hidden_amount'], out_amount=2,
        activation=config['activation'], step=config['step'], momentum=config['momentum'],
        weight_range=weight_range
    )
    
    history_train, history_test = network.train(train_inputs, train_targets, test_inputs, test_targets, epochs=config['epochs'])
    
    return {
        'config': config,
        'variant_name': variant_name,
        'final_test_mse': history_test[-1],
        'history_train': history_train,
        'history_test': history_test,
        'model_state': network.export_model()
    }

def main():
    CONFIG_SPACE = load_config()
    print("=== START TRENINGU ===")
    
    if not os.path.exists(DANE_DIR):
        print(f"[BLAD] Nie znaleziono folderu z danymi: {DANE_DIR}")
        return

    print("[1/5] Wczytywanie danych z plikow CSV...")
    raw_train_inputs, raw_train_targets, raw_test_inputs, raw_test_targets = FileManager.prepare_datasets(DANE_DIR)
    # W ujeciu profesjonalnym scaler x to jest to co mamy a scaler y jest to dco oczekujemy
    print("[2/5] Skalowanie danych...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_inputs = scaler_x.fit_transform(raw_train_inputs)
    train_targets = scaler_y.fit_transform(raw_train_targets)
    
    test_inputs = scaler_x.transform(raw_test_inputs)
    test_targets = scaler_y.transform(raw_test_targets)
    
    print("[3/5] Zapisywanie skalerow i liczenie linii odniesienia (Baseline)...")
    FileManager.save_json(scaler_x.get_state(), KNOWLEDGE_DIR, "scaler_x.json")
    FileManager.save_json(scaler_y.get_state(), KNOWLEDGE_DIR, "scaler_y.json")
    
    raw_scaled_by_y = scaler_y.transform(raw_test_inputs)
    baseline_mse_sum = 0.0
    for i in range(len(test_targets)):
        baseline_mse_sum += (raw_scaled_by_y[i][0] - test_targets[i][0])**2 + (raw_scaled_by_y[i][1] - test_targets[i][1])**2
    baseline_mse = baseline_mse_sum / (len(test_targets) * 2.0)
    
    FileManager.save_json({"baseline_mse": baseline_mse}, STATS_DIR, "baseline_info.json")
    
    configs = generate_configs(CONFIG_SPACE)
    results = []
    
    print(f"\n[4/5] ROZPOCZYNAM UCZENIE {len(configs)} SIECI.")
    '''Do jednoczesnego uruchamiania kilku sieci wzgledem dostepnych jednostek przetwarzajacych'''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_single_variant, c, train_inputs, train_targets, test_inputs, test_targets) for c in configs]
        skonczone = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            skonczone += 1
            print(f"      Ukonczono sieci: {skonczone}/{len(futures)}", end="\r", flush=True)

    print("\n\n[5/5] Zakonczono obliczenia. Eksportowanie wynikow do plikow...")
    
    results.sort(key=lambda r: r['final_test_mse'])
    
    global_best = results[0]
    global_best_act = global_best['config']['activation']
    FileManager.save_json({"best_activation": global_best_act}, STATS_DIR, "global_best.json")
    print(f"      *** BEST!!!!!: {global_best_act.upper()} (Test MSE: {global_best['final_test_mse']:.6f}) ***")
    
    best_per_act = {}
    for act in CONFIG_SPACE['activation']:
        wyniki_act = [r for r in results if r['config']['activation'] == act]
        if wyniki_act:
            best_per_act[act] = wyniki_act[0]
            
    for act, res in best_per_act.items():
        if res is None:
            continue
            
        FileManager.save_json(res['model_state'], KNOWLEDGE_DIR, f"model_best_{act}.json")
        FileManager.save_json(res['history_train'], STATS_DIR, f"train_history_{act}.json")
        FileManager.save_json(res['history_test'], STATS_DIR, f"test_history_{act}.json")
        FileManager.save_json(res['config'], STATS_DIR, f"config_best_{act}.json")
        
        FileManager.save_csv_history(res['history_train'], res['history_test'], STATS_DIR, f"bledy_epoki_{act}.csv")
        
    print("\n=== GOTOWE! Trening zakonczony pomyslnie. ===")

if __name__ == '__main__':
    main()