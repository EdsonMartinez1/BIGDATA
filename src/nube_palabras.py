import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud

# Cargar dataset
df = pd.read_csv("../data/spam.csv", encoding="latin-1")

# Mantener columnas necesarias
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Filtrar solo mensajes spam
spam_text = " ".join(
    df[df['label'] == 'spam']['message']
)

# Crear nube de palabras
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(spam_text)

# Mostrar gráfica
plt.figure(figsize=(10,5))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Nube de Palabras - Mensajes Spam")

# Guardar imagen
plt.savefig("../resultados/nube_spam.png")

plt.show()