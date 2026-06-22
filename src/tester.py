import os
from file_manager import FileManager
from scaler import MinMaxScaler
from neural_network import NeuralNetwork

DANE_DIR = os.path.join('.', 'dane')
KNOWLEDGE_DIR = os.path.join('.', 'knowledge')
STATS_DIR = os.path.join('.', 'statystyki')

def main():
    _, _, raw_test_inputs, raw_test_targets = FileManager.prepare_datasets(DANE_DIR)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.load_state(FileManager.load_json(KNOWLEDGE_DIR, "scaler_x.json"))
    scaler_y.load_state(FileManager.load_json(KNOWLEDGE_DIR, "scaler_y.json"))
    
    test_inputs = scaler_x.transform(raw_test_inputs)
    activations = ['sigmoid', 'tanh', 'relu']
    
    for act in activations:
        config_path = os.path.join(STATS_DIR, f"config_best_{act}.json")
        model_path = os.path.join(KNOWLEDGE_DIR, f"model_best_{act}.json")
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            continue
            
        config = FileManager.load_json(STATS_DIR, f"config_best_{act}.json")
        model_state = FileManager.load_json(KNOWLEDGE_DIR, f"model_best_{act}.json")
        
        network = NeuralNetwork(in_amount=2, hidden_amount=config['hidden_amount'], out_amount=2, activation=act)
        network.import_model(model_state)
        
        scaled_preds = [network.forward(x) for x in test_inputs]
        unscaled_preds = scaler_y.inverse_transform(scaled_preds)
        
        FileManager.save_json(unscaled_preds, STATS_DIR, f"predictions_{act}.json")
        FileManager.save_csv_predictions(unscaled_preds, STATS_DIR, f"odpowiedzi_oryginalna_skala_{act}.csv")

if __name__ == '__main__':
    main()