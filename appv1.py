from flask import Flask, render_template, request
import requests

app = Flask(__name__)
conversation = [] # Guardara preguntas y respuestas

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

@app.route("/", methods=["GET","POST"])
def index():
    global conversation
    if request.method == "POST":
        user_input = request.form["user_input"]

        contexto = cargar_contenido()

        payload = {
            "model": MODEL_NAME,
            "prompt": f"""
            Responde solo basandote en el siguiente texto. Si no sabes la respuesta di que no esta en el documento:
            \"\"\"{contexto}\"\"\"
            Pregunta: {user_input}""",
            #"prompt": user_input,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()

        conversation.append(("Tú", user_input))
        conversation.append(("IA", result["response"]))

    return render_template("index.html", conversation=conversation)

def cargar_contenido():
    texto_txt1 = open("info.txt", "r", encoding="utf-8").read()
    texto_txt2 = open("info2.txt", "r", encoding="utf-8").read()
    return f"{texto_txt1}\n{texto_txt2}"

if __name__ == "__main__":
    app.run(debug=True)