from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
import requests
import unicodedata

# Configuración de Flask
app = Flask(__name__)
conversation = [] # Guardara preguntas y respuestas

# Inicializar ChromaDB local
client = PersistentClient(path="./chroma")
collection = client.get_or_create_collection(name="documentos")

# Cargar el modelo de embedding (una sola vez)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Lista Global de Propiedades
propiedades = []

def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])
    return texto

def cargar_desde_url_y_embeddings():
    global collection, propiedades

    # Siempre refresca los datos
    client.delete_collection("documentos")
    collection = client.get_or_create_collection(name="documentos")

    # Obtener datos del json desde la URL
    url = "https://gist.githubusercontent.com/Jooochido/b9c2fbfe54a2cb8d112852cda0142cad/raw/propiedades_inmobiliaria.json"
    response = requests.get(url)
    propiedades = response.json()

    # Indexar solo los titulos con su ID real
    for casa in propiedades:
        id_real = str(casa["id"])
        titulo = normalizar(casa["titulo"])
        embedding = embedder.encode(titulo).tolist()

        collection.add(
            documents = [titulo],
            ids = [id_real],
            embeddings = [embedding]
        )

cargar_desde_url_y_embeddings()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

# Ruta Principal
@app.route("/", methods=["GET","POST"])
def index():
    global conversation
    if request.method == "POST":
        user_input = normalizar(request.form["user_input"])

        # Embbeding de la pregunta
        embedding_input = embedder.encode(user_input).tolist()

        # Buscar texto mas similar en la coleccion
        resultados = collection.query(query_embeddings=[embedding_input], n_results=1)
        ids_encontrados = resultados["ids"][0]

        if not ids_encontrados:
            result = "No tengo datos sobre eso."
        else:
            id_real = int(ids_encontrados[0])
            casa = next((c for c in propiedades if c["id"]  == id_real), None)

            if not casa:
                result = "No tengo datos sobre eso."
            else:
                detalles = "\n".join([
                    f"- {d.get('nombre')}: {d.get('descripcion')}"
                    for d in casa.get("detalles", [])
                ])
                contacto = casa.get("contacto", {})
                contexto = f"""
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

                promt = f"""
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

                💡 IMPORTANTE: Responde solo con la información proporcionada.

                Usa un tono conversacional, como si estuvieras ayudando a un cliente. Puedes sugerir propiedades si coinciden con la pregunta del usuario.

                Propiedad Encontrada:

                \"\"\"{contexto}\"\"\"

                Pregunta: {user_input}
                """

                payload ={
                    "model": MODEL_NAME,
                    "prompt": promt,
                    "stream": False
                }

                response = requests.post(OLLAMA_URL, json=payload)
                result = response.json()["response"]

        conversation.append(("Tú", user_input))
        conversation.append(("IA", result))

    return render_template("index.html", conversation=conversation)

if __name__ == "__main__":
    app.run(debug=True)