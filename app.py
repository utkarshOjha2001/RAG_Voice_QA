import streamlit as st
import pyttsx3
import re
import os
import tempfile
from dotenv import load_dotenv
import speech_recognition as sr
from langchain.prompts import PromptTemplate  
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

custom_tmp = os.path.join(os.getcwd(), "tmp")
os.makedirs(custom_tmp, exist_ok=True)
tempfile.tempdir = custom_tmp

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
    "Your task is to ask the questions based on the document provided by the user. "
    "You are a voice-based RAG assistant that can ask or if asked should answer questions based on the document provided. "
    "If you don't know the answer, say 'I don't know'."

)

def custom_spinner(text="Processing..."):
    """Creates a custom spinner with blur effect"""
    spinner_html = f"""
        <div class="custom-spinner-overlay">
            <div class="spinner-container">
                <div class="custom-spinner"></div>
                <div class="spinner-inner"></div>
                <div class="spinner-text">{text}</div>
            </div>
        </div>
    """
    spinner_placeholder = st.empty()
    return spinner_placeholder, spinner_html

def simple_spinner(text="Processing..."):
    """Creates a simple spinner without blur effect"""
    spinner_html = f"""
        <div class="simple-spinner-container">
            <div class="simple-spinner"></div>
            <div class="spinner-text">{text}</div>
        </div>
    """
    spinner_placeholder = st.empty()
    return spinner_placeholder, spinner_html

# Voice Output
# Voice Output
def text_to_speech(text):
    cleaned_text = re.sub(r"[\*\–\-\(\)\[\]\{\}:;]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  
    engine.setProperty('volume', 1.0)
    
    engine.say(cleaned_text)
    engine.runAndWait()


def speech_to_text():
    with sr.Microphone() as source:
        st.toast("Listening...", icon="🎧")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            if st.button("Try Again", key="retry_button"):
                return speech_to_text()
            return None

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
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
        vectorstore.persist() 

        return vectorstore
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        return None
    finally:
        os.unlink(tmp_file_path)

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

def load_css():
    with open('static/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Voice RAG Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_css()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

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

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

persist_dir = "./chroma_db"
if st.session_state.vectorstore is None and os.path.exists(persist_dir):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)  # ✅ ADD THIS LINE
        st.session_state.document_processed = True
    except Exception as e:
        st.error(f"Failed to load existing vectorstore: {str(e)}")



uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=['pdf', 'txt'])

if uploaded_file and not st.session_state.document_processed:
    spinner_placeholder, spinner_html = custom_spinner("Processing document...")
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
    try:
        st.session_state.vectorstore = process_document(uploaded_file)
        if st.session_state.vectorstore:
            st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)  # <--- Store chain
            st.session_state.document_processed = True
            st.markdown('<div class="success">✨ Document processed successfully! Ready for your questions.</div>', unsafe_allow_html=True)
    finally:
        spinner_placeholder.empty()


if st.session_state.vectorstore and st.session_state.document_processed:
    st.markdown("""
        <div class="card">
            <h2 class="section-header">🎙️ Voice Interaction</h2>
            <p class="interaction-description">Click the button below and speak your question clearly.</p>
            <div class="button-container">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button(" Start Voice Query ", type="primary", key="voice_button"):
            question = speech_to_text()
            if question:
                st.session_state.last_question = question
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    if 'last_question' in st.session_state:
        question = st.session_state.last_question
        qa_chain = st.session_state.qa_chain
        if qa_chain:
            with st.spinner("🤔 Processing your question..."):
                result = qa_chain({"question": question, "chat_history": st.session_state.chat_history})
                answer = result["answer"]
                text_to_speech(answer)
                st.session_state.chat_history.append((question, answer))

    if st.session_state.chat_history:
        st.markdown('<div class="card1">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Conversation History</h2>', unsafe_allow_html=True)
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
    st.markdown("""
        <div class="info-message">
            <h3 class="welc-header">👋 Welcome!</h3>
            <p>To get started, please upload a document using the file uploader above.</p>
            <p>I can help you analyze and answer questions about your documents using voice interaction.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)