# 🎧 Voice RAG Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A modern voice-based document interaction system powered by RAG (Retrieval-Augmented Generation)

<p align="center">
  <img src="assets/microphone.svg" alt="Voice RAG Assistant Logo" width="150"/>
</p>

</div>

## ✨ Features

- 🎤 **Voice Interaction**: Natural voice-based queries for document interaction
- 📄 **Document Processing**: Support for PDF and TXT files
- 🤖 **Intelligent Responses**: Powered by Google's Gemini AI
- 💡 **RAG Architecture**: Enhanced responses using document context
- 🎨 **Modern UI**: Beautiful, responsive design with glass morphism effects
- 🔊 **Text-to-Speech**: Natural voice responses for better interaction

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAG_Voice_QA.git
   cd RAG_Voice_QA
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Add your Google API key to .env file
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python 3.8+
- **AI Model**: Google Gemini
- **Vector Store**: Chroma DB
- **Speech Processing**: 
  - Speech Recognition
  - Google Text-to-Speech (gTTS)
- **Document Processing**: LangChain

## 💡 How It Works

1. **Document Upload**: Upload your PDF or TXT document
2. **Processing**: Document is processed and stored in Chroma DB
3. **Voice Query**: Click the microphone button and ask your question
4. **RAG Processing**: 
   - Retrieves relevant context from the document
   - Generates response using Gemini AI
5. **Voice Response**: Converts the answer to speech

## 📁 Project Structure

```
RAG_Voice_QA/
├── app.py              # Main application file
├── static/
│   └── style.css      # Custom styling
├── assets/
│   └── microphone.svg # UI assets
├── tmp/               # Temporary files
├── chroma_db/         # Vector database
└── requirement.txt    # Dependencies
```

## ⚙️ Configuration

Create a `.env` file with the following:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini AI for powerful language processing
- Streamlit for the amazing web framework
- LangChain for RAG implementation
- All contributors and users of this project

---

<div align="center">
Made with ❤️ for the AI community
</div> 