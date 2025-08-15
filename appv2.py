from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
import requests

# Configuración de Flask
app = Flask(__name__)
conversation = [] # Guardara preguntas y respuestas

# Inicializar ChromaDB local
client = PersistentClient(path="./chroma")
collection = client.get_or_create_collection(name="documentos")

# Cargar el modelo de embedding (una sola vez)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cargar_text_y_embedding():
    if collection.count() > 0:
        return
    with open("info2.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    fragmentos = [p.strip() for p in raw_text.split("\n") if p.strip]
    embeddings = embedder.encode(fragmentos).tolist()

    for i, fragmento in enumerate(fragmentos):
        collection.add(documents=[fragmento], ids=[f"frag{i}"], embeddings=[embeddings[i]])

cargar_text_y_embedding()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

# Ruta Principal
@app.route("/", methods=["GET","POST"])
def index():
    global conversation
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Embbeding de la pregunta
        embedding_input = embedder.encode(user_input).tolist()

        # Buscar texto mas similar en la colección
        resultados = collection.query(query_embeddings=[embedding_input], n_results=30)
        fragmentos = resultados["documents"][0]
        contexto = "\n".join(fragmentos)

        prompt = f"""
            Eres un asistente amigable y profesional.

            Cuando el usuario te pregunte por cursos, responde de manera clara y natural, como si conversaras con él.

            Agrupa los cursos por tema si es posible, y no repitas el título exacto si puedes resumirlo. Usa un tono cercano, como si recomendaras personalmente.

            Puedes analizar, contar y filtrar cursos si se te pide. No inventes cursos que no estén presentes en el texto.
            Si no encuentras lo que se te pregunta, responde exactamente con: "No tengo datos sobre eso."

            Lista de Cursos Disponibles:
            \"\"\"{contexto}\"\"\"
            Pregunta: {user_input}"""

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()["response"]

        conversation.append(("Tú", user_input))
        conversation.append(("IA", result))

    return render_template("index.html", conversation=conversation)

if __name__ == "__main__":
    app.run(debug=True)