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

def cargar_desde_url_y_embeddings():
    # Siempre refresca los datos
    client.delete_collection("documentos")

    global collection
    collection = client.get_or_create_collection(name="documentos")

    #1. Obtener datos del json desde la URL
    url = "https://gist.githubusercontent.com/Jooochido/b9c2fbfe54a2cb8d112852cda0142cad/raw/propiedades_inmobiliaria.json"
    response = requests.get(url)
    data = response.json()

    fragmentos = []

    #2 Crear un fragmento de texto por propiedad
    for casa in data:
        detalles = "\n".join([
            f"- {d.get('nombre')}: {d.get('descripcion')}"
            for d in casa.get("detalles", [])
        ])
        contacto = casa.get("contacto", {})
        texto = f"""
        🏠 {casa.get("titulo")}
        📍 Dirección: {casa.get("direccion")}, {casa.get("distrito")}
        📐 Área: {casa.get("metros_cuadrados")} m²
        🛏️ Habitaciones: {casa.get("habitaciones")}
        🚿 Baños: {casa.get("banos")}
        💰 Precio: {casa.get("precio")} {casa.get("moneda")} 
        📦 Tipo: {casa.get("tipo")}
        📝 Descripción: {casa.get("descripcion")}
        📆 Publicado en: {casa.get("publicado_en")}

        📞 Contacto:
        - Nombre: {contacto.get("nombre")}
        - Teléfono: {contacto.get("telefono")}
        - Email: {contacto.get("email")}

        🔍 Detalles Adicionales:
        {detalles}
        """.strip()

        fragmentos.append(texto)

    #3 Embedding y carga en chroma
    embeddings = embedder.encode(fragmentos).tolist()

    for i, fragmento in enumerate(fragmentos):
        collection.add(documents=[fragmento], ids=[f"propiedad_{i}"], embeddings=[embeddings[i]])

cargar_desde_url_y_embeddings()

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
        resultados = collection.query(query_embeddings=[embedding_input], n_results=10)
        fragmentos = resultados["documents"][0]
        contexto = "\n".join(fragmentos)

        prompt = f"""
        Eres un asistente inmobiliario profesional, amable y claro. Tu tarea es ayudar a los usuarios a encontrar propiedades usando únicamente la información proporcionada a continuación.

        ✅ Puedes responder preguntas sobre:
        - Ubicación (distrito, dirección)
        - Precio y moneda
        - Tamaño (en m²), número de habitaciones y baños
        - Tipo de propiedad (venta, alquiler, etc.)
        - Fecha de publicación
        - Detalles especiales como piscina, jardín, sala de estar, etc.
        - Datos de contacto (nombre, teléfono, email)

        ❌ No combines datos de distintas propiedades, incluso si parecen similares.
        ❌ No inventes propiedades ni detalles que no estén en el texto.
        ❌ No Repitas propiedades al responder.
        ❌ Si no tienes información, responde exactamente: "No tengo datos sobre eso."

        💡 IMPORTANTE: Si hay muchas propiedades listadas, responde solo con las más relevantes y específicas a la pregunta del usuario.

        Usa un tono conversacional, como si estuvieras ayudando a un cliente. Puedes sugerir propiedades si coinciden con la pregunta del usuario.

        Lista de propiedades disponibles:
        
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