import streamlit as st
import re
import subprocess
import os
import tempfile
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from langchain.prompts import PromptTemplate
from playsound import playsound  
import google.generativeai as genai
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup temp directory
custom_tmp = os.path.join(os.getcwd(), "tmp")
os.makedirs(custom_tmp, exist_ok=True)
tempfile.tempdir = custom_tmp

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    genai.GenerativeModel('gemini-2.0-flash').generate_content("Test")
except Exception as e:
    st.error(f"Error configuring Gemini API: {str(e)}")
    st.stop()

# Globals
recognizer = sr.Recognizer()
trainData = (
    "Remember: Do not include any punctuation or special characters in your responses. "
    "Use only the text content from the document provided. "
    "If you don't know the answer, say 'I don't know'."
)

# Voice Output
def text_to_speech(text):
    # Clean text a bit
    cleaned_text = re.sub(r"[\*\–\-\(\)\[\]\{\}:;]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    tts = gTTS(cleaned_text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)
    playsound(temp_path)
    os.remove(temp_path)

# Voice Input
def speech_to_text():
    with sr.Microphone() as source:
        st.toast("Listening...", icon="🎧")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            return None

# Document Processing
def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path) if file.name.endswith('.pdf') else TextLoader(tmp_file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        return None
    finally:
        os.unlink(tmp_file_path)

# QA Chain with system prompt
def get_qa_chain(vectorstore):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            verbose=False,
            condense_question_prompt=PromptTemplate.from_template(trainData + "\n\n{question}")
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Load CSS from external file
def load_css():
    with open('static/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Streamlit UI Config
st.set_page_config(
    page_title="Voice RAG Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load external CSS
load_css()

# Main UI Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header Section
st.markdown("""
    <h1 class="app-title">🎧 Voice-Based RAG Assistant</h1>
    <p class="app-description">Your intelligent voice companion for document interaction. Upload a document and start a conversation!</p>
""", unsafe_allow_html=True)

# Upload Section
st.markdown("""
    <div class="card">
        <h2 class="section-header1">📄 Document Upload</h2>
        <p class="upload-description">Support for PDF and TXT files. Your documents are processed securely.</p>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=['pdf', 'txt'])
if uploaded_file:
    with st.spinner("Processing document..."):
        st.session_state.vectorstore = process_document(uploaded_file)
        if st.session_state.vectorstore:
            st.markdown('<div class="success">✨ Document processed successfully! Ready for your questions.</div>', unsafe_allow_html=True)

# Session State Setup
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# QA Flow
if st.session_state.vectorstore:
    # Voice Interaction Section
    st.markdown("""
        <div class="card">
            <h2 class="section-header">🎙️ Voice Interaction</h2>
            <p class="interaction-description">Click the button below and speak your question clearly.</p>
            <div class="voice-button-container">
                <img src="assets/microphone.svg" alt="Microphone" class="mic-icon" />
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start Voice Query", type="primary", key="voice_button"):
            question = speech_to_text()
            if question:
                st.session_state.last_question = question

    if 'last_question' in st.session_state:
        question = st.session_state.last_question
        qa_chain = get_qa_chain(st.session_state.vectorstore)
        if qa_chain:
            with st.spinner("🤔 Processing your question..."):
                result = qa_chain({"question": question, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            text_to_speech(answer)
            st.session_state.chat_history.append((question, answer))

    # Chat History Section
    if st.session_state.chat_history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">📜 Conversation History</h2>', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for q, a in st.session_state.chat_history:
            st.markdown(f"""
                <div class="chat-row">
                    <div class="user-message">
                        <strong>You:</strong> {q}
                    </div>
                </div>
                <div class="chat-row">
                    <div class="bot-message">
                        <strong>Assistant:</strong> {a}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
else:
    # Welcome Message
    st.markdown("""
        <div class="info-message">
            <h3 class="welc-header">👋 Welcome!</h3>
            <p>To get started, please upload a document using the file uploader above.</p>
            <p>I can help you analyze and answer questions about your documents using voice interaction.</p>
        </div>
    """, unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
