import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Cargar datos
df = pd.read_csv("../data/spam.csv", encoding="latin-1")

# Limpiar columnas
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convertir etiquetas
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Vectorización
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['message'])
y = df['label']

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modelo
modelo = MultinomialNB()

modelo.fit(X_train, y_train)

# Predicción
pred = modelo.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("RESULTADOS")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")