# Face-Recognition-WebApp
This project is a web-based face recognition system built with Streamlit
. It leverages InsightFace for face detection and embedding extraction, and uses a database to store and manage user profiles (name, gender, age, and face embedding).

✨ Key Features

👤 Face detection and recognition in real time

📝 Add and manage face data in a local database

📊 Display stored profiles (name, gender, age)

⚡ Powered by InsightFace + ONNXRuntime for fast and accurate embeddings

🌐 Simple, interactive web interface with Streamlit

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔧 Tech Stack

Python 3.9+

Streamlit – web app framework

InsightFace – face recognition and embedding

ONNXRuntime – inference engine

SQLite / custom DB – for storing face profiles

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Getting Started

Clone the repo: git clone https://github.com/your-username/face-rec-webapp.git
cd face-rec-webapp

Create virtual environment: 1. python -m venv .venv
2.  | .venv\Scripts\activate   for Windows
2. | source .venv/bin/activate for Linux/Mac


Install dependencies: pip install -r requirements.txt

Run the app: streamlit run app.py


⚙️ Configuration

Adjust recognition similarity threshold in face_engine.py via SIM_THRESHOLD.

Add new profiles via the Streamlit UI or by inserting directly into the database.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Notes

Make sure Microsoft C++ Build Tools are installed (for Windows).

Requires installing insightface with onnxruntime.
