import streamlit as st
import requests
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from PIL import Image


try:
 from google import genai
 GENAI_AVAILABLE = True
except Exception:
 GENAI_AVAILABLE = False


CACHE_PATH = "faiss_cache.pkl"
GENAI_KEY = st.secrets["GENAI_API_KEY"]
TOP_K = 5
SYNTHESIS_TOP_K = 3
SIMILARITY_THRESHOLD = 0.6

icon= Image.open("babe_logo_shadow-1.png")
st.set_page_config(page_title="B.a.B.e Chatbot", page_icon=icon)


st.title("B.a.B.e Chatbot")

st.info("""
Dobrodošli 💜

Postavite pitanje ili odaberite jednu od ponuđenih opcija ispod.
""")



if "messages" not in st.session_state:
    st.session_state.messages = [
       {"role": "system", "content": "Odgovaraj isključivo na hrvatskom jeziku i "
       "odgovaraj toplim, prijateljskim, ali i dalje službenim tonom. "
       "Na pitanja u kojima korisnik koristi ti, vi, vaši i slično, "
       "odgovaraj kao zaposlenik udruge B.a.Be. s relevatnim informacijama i šalji potrebne linkove,"
       "znaš sve informacije!"}
    ]

@st.cache_resource
def load_cache(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    chunks = payload["chunks"]
    metadatas = payload["metadatas"]
    index_bytes = payload["index_bytes"]
    model_name = payload.get("model_name", "paraphrase-multilingualMiniLM-L12-v2")
    index = faiss.deserialize_index(index_bytes)
    embed_model = SentenceTransformer(model_name)
    return {"chunks": chunks, "metadatas": metadatas, "index": index,
    "embed_model": embed_model}

cache= None
try:
 cache = load_cache(CACHE_PATH)
 st.session_state.cache_loaded = True
except FileNotFoundError:
 st.warning(f"Cache datoteka {CACHE_PATH} nije pronađena. Najprije pokreni embeddings.py.")
except Exception as e:
 st.error(f"Greška pri učitavanju cache-a: {e}")


def retrieve(query, top_k=TOP_K):
    if not cache:
        return []
    model = cache["embed_model"]
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = cache["index"].search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({"score": float(score), "chunk":
cache["chunks"][idx], "meta": cache["metadatas"][idx]})
    return results

def call_gemini(prompt_text):
    if not GENAI_AVAILABLE:
        return "Gemini (google-genai) nije instaliran u ovom okruženju."
    if not GENAI_KEY:
        return "Gemini API ključ nije postavljen (GENAI_API_KEY)."
    try:
        # konfiguriraj biblioteku (ako podržava configure)
        try:
            genai.configure(api_key=GENAI_KEY)
            client = genai.Client()
        except Exception:
        # neke verzije koriste Client() bez configure
            client = genai.Client(api_key=GENAI_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )

        text = ""
        if hasattr(response, "text"):
            text = response.text
        elif isinstance(response, dict) and "text" in response:
            text = response["text"]
        else:
        
            text = str(response)
        return text
    except Exception as e:
        return f"Greška pri pozivu Gemini API-ja: {e}"
        

st.markdown("### Kako vam možemo pomoći?")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

selected_prompt = None

with col1:
    if st.button("Trebam pomoć!"):
        selected_prompt = """
        Trebam pomoć vezanu uz pravnu ili psihološku podršku,
        siguran smještaj ili savjetovanje.
        """

with col2:
    if st.button("Želim podržati udrugu!"):
        selected_prompt = """
        Kako mogu podržati udrugu B.a.B.e.?
        Zanimaju me donacije i načini uključivanja.
        """

with col3:
    if st.button("Zanimaju me projekti koje udruga provodi."):
        selected_prompt = """
        Koje projekte provodi udruga B.a.B.e.?
        """

with col4:
    if st.button("Zanimaju me aktivnosti udruge."):
        selected_prompt = """
        Koje aktivnosti provodi udruga B.a.B.e.?
        """



user_text = st.chat_input("Napiši poruku...", key="main_chat_input")

if selected_prompt:
    user_input = selected_prompt
elif user_text:
    user_input = user_text
else:
    user_input = None

if user_input:

    
    st.session_state.messages.append({"role": "user", "content":
    user_input})

    retrieved = retrieve(user_input, top_k=TOP_K)
    best = retrieved[0] if retrieved else None

    if best and best["score"] >= SIMILARITY_THRESHOLD:

        top_chunks = retrieved[:SYNTHESIS_TOP_K]
        combined_text = "\n\n---\n\n".join([r["chunk"] for r in
        top_chunks])

    # RAG synthesis prompt
        synthesis_prompt = f"""
Korisnik je postavio pitanje: "{user_input}"

U nastavku se nalaze izdvojeni dijelovi članka.
Na temelju NJIH (i ničega drugog), napiši prirodan, koherentan i jasan
odgovor na hrvatskom jeziku.
Nemoj dodavati ništa što se ne nalazi u tekstu.
Ako su dijelovi nepotpuni, svejedno napravi uredan i logičan sažetak.
--- POČETAK TEKSTA ---
{combined_text}
--- KRAJ TEKSTA ---
Odgovor:
"""
        answer = call_gemini(synthesis_prompt)
        # dodaj izvor
        answer += f"\n\n*(Izvor: {best['meta']['url']})*"

    else:
    
        convo = []
        for m in st.session_state.messages:
            role = m["role"].upper()
            content = m["content"]
            convo.append(f"{role}: {content}")
        prompt_text = "\n".join(convo) + + "\n\nOdgovori na hrvatskom jeziku. "
        + "Koristi samo i isključivo informacije koje se mogu pronaći na web stranici udruge B.a.B.e. "
        + "Dodaj linkove na kojima se može pronaći više informacija."
        
        answer = call_gemini(prompt_text)


    
    st.session_state.messages.append({"role": "assistant", "content":
    answer})

for msg in st.session_state.messages:

      # NE prikazuj system prompt korisniku
    if msg["role"] == "system":
        continue

    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

