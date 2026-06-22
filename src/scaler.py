'''Klasa wzorowana scalerem MinMax'''
class MinMaxScaler:
    def __init__(self):
        self.min_values = []
        self.max_values = []
    '''Funkcja definiujaca punkty odniesienia'''
    def fit_transform(self, data):

        if not data:
            return []
        amount_of_columns = len(data[0])
        self.min_values = [min(row[c] for row in data) for c in range(amount_of_columns)]
        self.max_values = [max(row[c] for row in data) for c in range(amount_of_columns)]
        return self.transform(data)
    '''funkcja wlasciwej tranformacji wzgledem pivotow'''
    def transform(self, data):
        amount_of_columns = len(data[0])
        scaled_data = []
        for row in data:
            scaled_row = []
            for c in range(amount_of_columns):
                value_range = self.max_values[c] - self.min_values[c]
                if value_range == 0:
                    scaled_row.append(0.0)
                else:
                    scaled_row.append((row[c] - self.min_values[c]) / value_range)
            scaled_data.append(scaled_row)
        return scaled_data
    '''Funkcja odwrocenia transformacji wzgledem pivotow'''
    def inverse_transform(self, scaled_data):
        amount_of_columns = len(scaled_data[0])
        original_data = []
        for row in scaled_data:
            original_row = []
            for c in range(amount_of_columns):
                value_range = self.max_values[c] - self.min_values[c]
                original_row.append((row[c] * value_range) + self.min_values[c])
            original_data.append(original_row)
        return original_data

    '''Narzedziowe funkcje (Getter/Setter) do stanu calej klasy'''
    def get_state(self):
        return {'min_values': self.min_values, 'max_values': self.max_values}

    def load_state(self, state):
        self.min_values = state['min_values']
        self.max_values = state['max_values']