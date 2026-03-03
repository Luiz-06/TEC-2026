import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

dados = {
    'idade': ['jovem', 'jovem', 'média', 'idoso', 'idoso', 'idoso', 'média', 'jovem', 'jovem', 'idoso', 'jovem', 'média', 'média', 'idoso'],
    'salario': ['alto', 'alto', 'alto', 'médio', 'baixo', 'baixo', 'baixo', 'médio', 'baixo', 'médio', 'médio', 'médio', 'alto', 'médio'],
    'conta': ['não', 'não', 'não', 'não', 'sim', 'sim', 'sim', 'não', 'sim', 'sim', 'sim', 'não', 'sim', 'não'],
    'empréstimo': ['não', 'não', 'sim', 'sim', 'sim', 'não', 'sim', 'não', 'sim', 'sim', 'sim', 'sim', 'sim', 'não']
}

df = pd.DataFrame(dados)

label_encoders = {}
for coluna in df.columns:
    le = LabelEncoder()
    df[coluna] = le.fit_transform(df[coluna])
    label_encoders[coluna] = le

X = df.drop(columns=['empréstimo'])
y = df['empréstimo']

modelo = DecisionTreeClassifier(criterion='entropy', random_state=42)
modelo.fit(X, y)

plt.figure(figsize=(14,8))
tree.plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=label_encoders['empréstimo'].classes_,
    filled=True,
    rounded=True
)
plt.title("Árvore de Decisão (baseada em Entropia - ID3)")
plt.show()
