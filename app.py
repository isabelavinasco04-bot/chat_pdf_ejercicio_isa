import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# ---- LangChain imports ----
from langchain.text_splitter import CharacterTextSplitter
# Si da error con estos imports, usa: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    from langchain_openai import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

st.set_page_config(page_title="RAG Directioner 💬", page_icon="🎤", layout="wide")

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬 — Experto en One Direction')
st.caption("Let’s go, directioners 🔥  |  Python " + platform.python_version())

# Sidebar
with st.sidebar:
    st.subheader("Tu Agente Directioner")
    st.write("Este Agente te ayudará a analizar tu PDF como si fuera un **experto en One Direction**.")
    st.markdown("- Prioriza el PDF cargado.\n- Si no está en el PDF, te avisa y complementa con conocimiento general.\n- Tono amigable y un pelín fandom 😉.")

# API Key
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Controles
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    expert_mode = st.checkbox(
        "Modo Experto One Direction",
        value=True,
        help="Activa el prompt de sistema estilo ‘Directioner senior’."
    )
with col_b:
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.1, 0.1,
                            help="Más bajo = más preciso; más alto = más creativo.")
with col_c:
    top_k = st.slider("Fragmentos relevantes (k)", 1, 8, 4, 1,
                      help="Cuántos fragmentos del PDF usar para contestar.")

# Subida de PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Prompt builder
def build_prompt(expert=True):
    if expert:
        system_text = (
            "Eres ‘Asistente Directioner’, un experto en One Direction (miembros: "
            "Harry Styles, Louis Tomlinson, Liam Payne, Niall Horan y Zayn Malik). "
            "Conoces su historia (The X Factor 2010, álbumes Up All Night, Take Me Home, "
            "Midnight Memories, Four, Made in the A.M.), giras, carreras solistas y momentos icónicos. "
            "Tu misión es responder en español, con tono cercano y fandom. "
            "Reglas:\n"
            "1) Prioriza SIEMPRE el contenido del PDF provisto en {context}. "
            "2) Si lo que pregunta el usuario NO está en el PDF, dilo claramente y complementa con conocimiento general. "
            "3) Sé preciso con nombres, fechas y discografía; si no estás seguro, decláralo."
        )
    else:
        system_text = (
            "Eres un asistente experto en análisis de documentos. "
            "Respondes de forma clara y concisa, priorizando el contexto del PDF en {context}."
        )

    system_msg = SystemMessagePromptTemplate.from_template(system_text)
    human_text = (
        "Contexto del PDF (fragmentos seleccionados):\n"
        "{context}\n\n"
        "Pregunta del usuario:\n"
        "{question}\n\n"
        "Entrega una respuesta directa y concreta."
    )
    human_msg = HumanMessagePromptTemplate.from_template(human_text)

    return ChatPromptTemplate.from_messages([system_msg, human_msg])

# Procesar PDF
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text

        if not text.strip():
            st.error("No se pudo extraer texto del PDF. Puede que sea un PDF escaneado sin OCR.")
        else:
            st.info(f"Texto extraído: {len(text)} caracteres")

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=120,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.success(f"Documento dividido en {len(chunks)} fragmentos")

            with st.spinner("Creando base de conocimiento..."):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.subheader("¿Qué quieres saber del documento (versión Directioner)?")
            user_question = st.text_area(" ", placeholder="Ej.: ¿Qué dice el PDF sobre la era ‘Midnight Memories’?")

            if user_question:
                docs = knowledge_base.similarity_search(user_question, k=top_k)
                retrieved_context = "\n\n".join([d.page_content.strip() for d in docs])

                prompt = build_prompt(expert=expert_mode)

                llm = ChatOpenAI(
                    temperature=temperature,
                    model="gpt-4o",
                )

                messages = prompt.format_messages(
                    context=retrieved_context,
                    question=user_question
                )

                with st.spinner("Pensando como Directioner…"):
                    response = llm(messages).content

                st.markdown("### Respuesta:")
                st.markdown(response)

                with st.expander("Ver fragmentos del PDF usados (contexto)"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Fragmento {i}:**\n\n{d.page_content}")

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
