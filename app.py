```python
import os
import re
import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HR Policy RAG Assistant",
    page_icon="📘",
    layout="wide"
)


# =========================================================
# SETTINGS
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(EMBEDDING_MODEL)

    return model


# =========================================================
# GROQ CLIENT
# =========================================================

@st.cache_resource
def get_groq_client(api_key):

    return Groq(api_key=api_key)


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text")

        text = text.strip()

        if text:

            text = re.sub(
                r"\s+",
                " ",
                text
            )

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    pdf.close()

    return pages


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        if end == len(words):

            break

        start = end - overlap

    return chunks


# =========================================================
# CREATE DOCUMENT CHUNKS
# =========================================================

def process_pdf(uploaded_file):

    pages = extract_text_from_pdf(
        uploaded_file
    )

    all_chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        chunks = create_chunks(text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number,
                    "source": uploaded_file.name
                }
            )

    return all_chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(chunks, model):

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_documents(
    question,
    index,
    chunks,
    model
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    k = min(
        TOP_K,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        result = chunks[index_number].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# =========================================================
# CREATE GROQ PROMPT
# =========================================================

def create_prompt(
    question,
    retrieved_documents
):

    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""
SOURCE {i}
File: {document["source"]}
Page: {document["page"]}

{document["text"]}

-------------------------
"""

    prompt = f"""
You are an HR Policy Assistant.

You must answer the user's question using ONLY the
information provided in the HR policy context below.

IMPORTANT RULES:

1. Do not invent HR policies.
2. Do not use outside knowledge.
3. If the answer is not available in the provided context,
   say:

   "I could not find this information in the uploaded HR policy."

4. Give a clear and simple answer.
5. Mention the relevant page number when possible.
6. Include important conditions or exceptions.
7. Do not make assumptions about company policy.

HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return prompt


# =========================================================
# GENERATE ANSWER USING GROQ
# =========================================================

def generate_answer(
    question,
    retrieved_documents,
    client
):

    prompt = create_prompt(
        question,
        retrieved_documents
    )

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role": "system",
                "content":
                "You are an accurate HR policy assistant. "
                "Answer only from the provided policy context."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0.1,

        max_tokens=1000
    )

    answer = response.choices[0].message.content

    return answer


# =========================================================
# SESSION STATE
# =========================================================

if "faiss_index" not in st.session_state:

    st.session_state.faiss_index = None


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "file_name" not in st.session_state:

    st.session_state.file_name = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write(
        "**Embedding Model:**"
    )

    st.code(
        EMBEDDING_MODEL
    )

    st.write(
        "**LLM Model:**"
    )

    st.code(
        GROQ_MODEL
    )

    st.write(
        f"**Retrieved Documents:** {TOP_K}"
    )

    st.divider()

    groq_api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY", "")
    )

    if groq_api_key:

        st.success(
            "Groq API key detected ✅"
        )

    else:

        st.error(
            "Groq API key not found ❌"
        )

        st.info(
            "Add GROQ_API_KEY in "
            "Streamlit Cloud → Settings → Secrets."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Document"
    ):

        st.session_state.faiss_index = None

        st.session_state.chunks = []

        st.session_state.file_name = None

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN TITLE
# =========================================================

st.title(
    "📘 HR Policy RAG Assistant"
)

st.write(
    "Upload an HR policy PDF and ask questions about it."
)

st.caption(
    "PyMuPDF + Sentence Transformers + FAISS + Groq"
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload HR Policy PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                model = load_embedding_model()

                chunks = process_pdf(
                    uploaded_file
                )

                if not chunks:

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                    st.info(
                        "The PDF may be scanned/image-based. "
                        "OCR may be required."
                    )

                    st.stop()

                faiss_index = create_faiss_index(
                    chunks,
                    model
                )

                st.session_state.faiss_index = (
                    faiss_index
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.messages = []

                st.success(
                    f"PDF processed successfully! "
                    f"Created {len(chunks)} chunks."
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )

    else:

        st.success(
            f"Current document: "
            f"{st.session_state.file_name}"
        )


# =========================================================
# SHOW CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and "sources" in message
        ):

            with st.expander(
                "📎 View Retrieved Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"**Page {source['page']}** "
                        f"| Similarity: "
                        f"{source['score']:.3f}"
                    )

                    st.caption(
                        source["text"]
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the HR policy..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    if (
        st.session_state.faiss_index
        is None
    ):

        st.warning(
            "Please upload an HR policy PDF first."
        )

        st.stop()

    if not groq_api_key:

        st.error(
            "Groq API key is not configured."
        )

        st.stop()

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching HR policy..."
        ):

            try:

                model = load_embedding_model()

                retrieved_documents = retrieve_documents(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    model
                )

                client = get_groq_client(
                    groq_api_key
                )

                answer = generate_answer(
                    question,
                    retrieved_documents,
                    client
                )

                st.markdown(
                    answer
                )


                # -------------------------------------------------
                # SOURCES
                # -------------------------------------------------

                with st.expander(
                    "📎 View Retrieved Sources"
                ):

                    for source in retrieved_documents:

                        st.write(
                            f"**{source['source']}** "
                            f"| Page **{source['page']}** "
                            f"| Similarity: "
                            f"{source['score']:.3f}"
                        )

                        st.caption(
                            source["text"]
                        )


                # -------------------------------------------------
                # SAVE ASSISTANT MESSAGE
                # -------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_documents
                    }
                )


            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )
```
