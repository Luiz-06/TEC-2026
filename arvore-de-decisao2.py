import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_breast_cancer

# --- 1. breast cancer ---
bc = load_breast_cancer()
X_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
y_bc = bc.target

# cria a arvore (profundidade 3 pra não gerar regras demais)
clf_bc = DecisionTreeClassifier(random_state=42, max_depth=3)
clf_bc.fit(X_bc, y_bc)

print("--- regras breast cancer ---")
print(export_text(clf_bc, feature_names=list(bc.feature_names)))


# --- 2. seeds ---
# link direto pro txt da uci
url_seeds = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
colunas_seeds = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove', 'class']

# le o arquivo arrumando os espacos em branco zoados do txt
df_seeds = pd.read_csv(url_seeds, sep=r'\s+', names=colunas_seeds)

X_seeds = df_seeds.drop('class', axis=1)
# ajusta as classes pra 0, 1 e 2 (o padrao vem 1, 2 e 3)
y_seeds = df_seeds['class'] - 1 

# treina o modelo das sementes
clf_seeds = DecisionTreeClassifier(random_state=42, max_depth=3)
clf_seeds.fit(X_seeds, y_seeds)

print("\n--- regras seeds ---")
print(export_text(clf_seeds, feature_names=list(X_seeds.columns)))
