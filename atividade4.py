
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def classificar_dataset(nome, data_loader):
    # 1. Carregar o dataset
    dados = data_loader()
    X = dados.data
    y = dados.target
    
    # 2. Dividir entre treino (70%) e teste (30%) conforme exemplo [cite: 122]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Criar e treinar o modelo Naive Bayes [cite: 123, 124, 125, 126]
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    
    # 4. Fazer previsões e avaliar [cite: 127, 128, 129, 130, 131]
    y_pred = modelo.predict(X_test)
    
    print(f"\n--- Resultado: {nome} ---")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório:\n", classification_report(y_test, y_pred, target_names=dados.target_names))

# Executar para Wine e Breast Cancer (disponíveis no sklearn)
classificar_dataset("Wine", load_wine)
classificar_dataset("Breast Cancer", load_breast_cancer)
