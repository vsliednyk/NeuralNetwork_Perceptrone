import csv
import os
import glob # przeszukiwanie plikow
import json

class FileManager:
    @staticmethod
    def load_directory_data(directory_path):
        inputs_measured = []
        targets_real = []
        search_pattern = os.path.join(directory_path, '*.csv')
        file_list = glob.glob(search_pattern)
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) >= 4:
                        try:
                            inputs_measured.append([float(row[0]), float(row[1])])
                            targets_real.append([float(row[2]), float(row[3])])
                        except ValueError:
                            continue
        return inputs_measured, targets_real

    @staticmethod
    def prepare_datasets(base_dane_dir):
        f8_stat_x, f8_stat_y = FileManager.load_directory_data(os.path.join(base_dane_dir, 'f8', 'stat'))
        f10_stat_x, f10_stat_y = FileManager.load_directory_data(os.path.join(base_dane_dir, 'f10', 'stat'))
        train_x = f8_stat_x + f10_stat_x
        train_y = f8_stat_y + f10_stat_y

        f8_dyn_x, f8_dyn_y = FileManager.load_directory_data(os.path.join(base_dane_dir, 'f8', 'dyn'))
        f10_dyn_x, f10_dyn_y = FileManager.load_directory_data(os.path.join(base_dane_dir, 'f10', 'dyn'))
        test_x = f8_dyn_x + f10_dyn_x
        test_y = f8_dyn_y + f10_dyn_y

        return train_x, train_y, test_x, test_y

    @staticmethod
    def save_json(data, dump_dir, file_name):
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, file_name), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(dump_dir, file_name):
        with open(os.path.join(dump_dir, file_name), 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_csv_history(train_hist, test_hist, dump_dir, file_name):
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, file_name), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for tr, te in zip(train_hist, test_hist):
                writer.writerow([tr, te])

    @staticmethod
    def save_csv_predictions(preds, dump_dir, file_name):
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, file_name), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for p in preds:
                writer.writerow([p[0], p[1]])