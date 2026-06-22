"""Modul odpowiadajacy za wygenerowanie pliku config.json w katalogu glownym"""
import json
import os

CONFIG_FILE = 'config.json'

'''Rozdzielanie wlasciwych opcji podawanych przy kazdym inpucie'''
def get_list_input(prompt, default, cast_type):
    user_input = input(f"{prompt} [domyslnie: {default}]: ").strip()
    if not user_input:
        return default
    try:
        return [cast_type(x.strip()) for x in user_input.split(',')]
    except ValueError:
        print("Blad formatu. Uzyto wartosci domyslnej.")
        return default

def get_single_input(prompt, default, cast_type):
    user_input = input(f"{prompt} [domyslnie: {default}]: ").strip()
    if not user_input:
        return default
    try:
        return cast_type(user_input)
    except ValueError:
        print("Blad formatu. Uzyto wartosci domyslnej.")
        return default

def main():
    print("=== KONFIGURATOR PRZESTRZENI BADAWCZEJ SIECI NEURONOWYCH ===")
    print("Instrukcja: Wprowadz wartosci oddzielone przecinkiem, aby przetestowac")
    print("wiele wariantow naraz. Wcisnij ENTER, aby uzyc wartosci domyslnej.\n")

    activations = input("Funkcje aktywacji (sigmoid, tanh, relu) [domyslnie: ['sigmoid', 'tanh', 'relu']]: ").strip()
    if not activations:
        activations = ['sigmoid', 'tanh', 'relu']
    else:
        activations = [x.strip().lower() for x in activations.split(',')]

    hidden_amount = get_list_input("Liczba neuronow w warstwie ukrytej (np. 8,16,32)", [8, 16, 32], int)
    epochs = get_list_input("Kryterium stopu - liczba epok (np. 50,100)", [100], int)
    step = get_list_input("Wspolczynnik uczenia (np. 0.01,0.001)", [0.001, 0.0001], float)
    momentum = get_list_input("Wspolczynnik momentum (np. 0.9)", [0.9], float)
    trials = get_single_input("Liczba niezaleznych realizacji procesu uczenia (np. 5)", 5, int)
    weight_range = get_single_input("Zakres inicjalizacji wag (np. 0.5)", 0.5, float)

    config = {
        'activation': activations,
        'hidden_amount': hidden_amount,
        'epochs': epochs,
        'step': step,
        'momentum': momentum,
        'trial': list(range(1, trials + 1)),
        'weight_range': weight_range
    }

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print(f"\nKonfiguracja zostala zapisana do pliku: {CONFIG_FILE}")

if __name__ == '__main__':
    main()