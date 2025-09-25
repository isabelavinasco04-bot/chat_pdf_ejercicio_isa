import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# ---- LangChain imports ----
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # si falla, usar: from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Chat models y prompting
try:
    # Ramas clásicas de LangChain
    from langchain.chat_models import ChatOpenAI
except Exception:
    # Ramas nuevas de LangChain
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

# Load and display image
with st.sidebar:
    st.subheader("Tu Agente Directioner")
    st.write("Este Agente te ayudará a analizar tu PDF como si fuera un **experto en One Direction**.")
    st.markdown("- Prioriza el PDF cargado.\n- Si no está en el PDF, te avisa y complementa con conocimiento general.\n- Tono amigable y un pelín fandom 😉.")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Controls
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    expert_mode = st.toggle("Modo Experto One Direction", value=True, help="Activa el prompt de sistema estilo ‘Directioner senior’.")
with col_b:
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.1, 0.1, help="Más bajo = más preciso; más alto = más creativo.")
with col_c:
    top_k = st.slider("Fragmentos relevantes (k)", 1, 8, 4, 1, help="Cuántos fragmentos del PDF usar para contestar.")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

def build_prompt(expert=True):
    """Crea el ChatPromptTemplate con o sin ‘modo experto’."""
    if expert:
        system_text = (
            "Eres ‘Asistente Directioner’, un experto en One Direction (miembros: "
            "Harry Styles, Louis Tomlinson, Liam Payne, Niall Horan y Zayn Malik). "
            "Conoces su historia (The X Factor 2010, eras y giras, álbumes Up All Night, "
            "Take Me Home, Midnight Memories, Four, Made in the A.M.), fechas clave, "
            "carreras solistas, colaboraciones y momentos icónicos del fandom. "
            "Tu misión es responder en español, con tono cercano, claro y respetuoso, "
            "usando un toque ligero de humor Gen Z cuando sea apropiado. "
            "Reglas:\n"
            "1) Prioriza SIEMPRE el contenido del PDF provisto en {context}. "
            "2) Si lo que pregunta el usuario NO está en el PDF, dilo de forma explícita y, "
            "si procede, complementa con conocimiento general. "
            "3) Sé preciso con nombres, fechas y discografía; si no estás seguro, decláralo. "
            "4) Cuando corresponda, explica de dónde sale la respuesta (del PDF o conocimiento general)."
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
        "Entrega una respuesta directa, con detalles concretos y, si aplica, una breve justificación."
    )
    human_msg = HumanMessagePromptTemplate.from_template(human_text)

    return ChatPromptTemplate.from_messages([system_msg, human_msg])


# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            # Algunos PDFs pueden devolver None en extract_text()
            page_text = page.extract_text() or ""
            text += page_text

        if not text.strip():
            st.error("No se pudo extraer texto del PDF. Asegúrate de que no sea un PDF escaneado sin OCR.")
        else:
            st.info(f"Texto extraído: {len(text)} caracteres")

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=120,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.success(f"Documento dividido en {len(chunks)} fragmentos")

            # Create embeddings and knowledge base
            with st.spinner("Creando base de conocimiento (FAISS + embeddings)..."):
                embeddings = OpenAIEmbeddings()  # si falla, usa: OpenAIEmbeddings(model="text-embedding-3-large")
                knowledge_base = FAISS.from_texts(chunks, embeddings)

            # User question interface
            st.subheader("¿Qué quieres saber del documento (versión Directioner)?")
            user_question = st.text_area(" ", placeholder="Ej.: ¿Qué dice el PDF sobre la época ‘Midnight Memories’?")

            if user_question:
                # Retrieve top-k docs
                docs = knowledge_base.similarity_search(user_question, k=top_k)
                retrieved_context = "\n\n".join([d.page_content.strip() for d in docs])

                # Build prompt
                prompt = build_prompt(expert=expert_mode)

                # LLM
                llm = ChatOpenAI(
                    temperature=temperature,
                    model="gpt-4o",  # puedes cambiar a "gpt-4.1" o el que tengas acceso
                )

                # Compose final input
                messages = prompt.format_messages(
                    context=retrieved_context,
                    question=user_question
                )

                with st.spinner("Pensando como Directioner…"):
                    response = llm(messages).content

                # Display the response
                st.markdown("### Respuesta:")
                st.markdown(response)

                with st.expander("Ver fragmentos del PDF usados (contexto)"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Fragmento {i}:**\n\n{d.page_content}")

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
