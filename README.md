# RAG Assistant with Streamlit

## 📌 Overview
This project is a **Retrieval-Augmented Generation (RAG) Assistant** built using **LangChain**, **ChromaDB**, and **Google Generative AI (Gemini)**.  
It allows users to ask questions over a document dataset and get intelligent, contextual answers.  

The project also includes a **Streamlit UI** for easy interaction.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ashuuu2409/rag-assistant.git
   cd rag-assistant


2.Create a virtual environment:

python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows


3.Install dependencies:

pip install -r requirements.txt


4.Set your API key (Google Gemini):

export GOOGLE_API_KEY="your_api_key_here"      # Mac/Linux
setx GOOGLE_API_KEY "your_api_key_here"        # Windows


🚀 Usage

Run the Streamlit app:

streamlit run project1.py


📂 Project Structure
rag-assistant/
│── project1.py          # Main RAG + Streamlit code
│── requirements.txt     # Dependencies
│── README.md            # Documentation