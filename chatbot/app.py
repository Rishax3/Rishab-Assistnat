from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import os

app = Flask(__name__)

# --- API KEY ---
API_KEY = "AIzaSyD1VjuLl9PQ6YDEMCFZCjpRVv0JuVbX4T8"  # Replace with your real Gemini API key
client = genai.Client(api_key=API_KEY)


# --- Function to call Gemini API ---
def chatbot_response(history):
    # Convert history into Gemini format
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # System prompt / instruction
    generate_config = types.GenerateContentConfig
    temperature=0.7,
    max_output_tokens=200,  # limit
    system_instruction = [
        types.Part.from_text(text="""
You are a helpful assistant.

TONE & STYLE:
- Speak concisely and clearly.
- Keep replies short (2–4 sentences max) unless a step-by-step answer is required.
- Keep casual, friendly, and guiding tone.

KNOWLEDGE & PURPOSE:
- You teach and guide users on any topic they ask.
- You know your creator is Rishab Kumar, he is the founder of FoundPreneur (use naturally if relevant). he is passionate about business & technology.

BEHAVIOR:
- Focus on giving actionable, easy-to-follow guidance.
- Recall previous messages if user asks.
- Apologize politely if answer is unclear.
""")
    ]

    try:
        # Call Gemini model
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_instruction
            )
        ):
            response_text += chunk.text
        return response_text.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    history = data.get("history", [])

    # Add new user message to history
    history.append({"role": "user", "content": user_message})

    # Get bot response
    reply = chatbot_response(history)

    # Add bot reply to history
    history.append({"role": "bot", "content": reply})

    return jsonify({"reply": reply, "history": history})


if __name__ == "__main__":
    app.run(debug=True)
