import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time

from docx import Document


URLS = [
 "https://babe.hr/pravna-i-psiholoska-pomoc/", 
 "https://babe.hr/smjestaj-u-sigurnoj-kuci/",
 "https://babe.hr/projekt/uskladivanje-privatnog-i-poslovnog-zivota/",
 "https://babe.hr/projekt/surf-sound-2-0-support-unite-respond-fight-to-stop-online-violence/",
  "https://babe.hr/ostali-projekti/",
  "https://babe.hr/predavanje-na-filozofskom-fakultetu-u-rijeci-tematska-sustavna-podrska/",
  "https://babe.hr/okrugli-stol-nevidljivi-teret-rodne-dimenzije-kognitivno-emocionalnog-rada/",
 "https://babe.hr/donacije/",
 "https://babe.hr/kontakt/"
]

CHUNK_SIZE = 500 
CHUNK_OVERLAP = 100 
CACHE_PATH = "faiss_cache.pkl"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2" 
DOCS = ["babe_kontakt.docx"]

def fetch_docx_text(path):
    try:
        doc = Document(path)
        full_text = []

        for p in doc.paragraphs:
            if p.text.strip():
                full_text.append(p.text.strip())

        return "\n".join(full_text)

    except Exception as e:
        print(f"Greška pri čitanju {path}: {e}")
        return ""
    
def fetch_article_text(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        article_tag = soup.find("article")
        if article_tag:
            text = article_tag.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"Greška pri dohvaćanju {url}: {e}")
    return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def main():
    
  
    all_chunks = []
    metadatas = []

    print(" Dohvaćanje lokalnih dokumenata...")

    for doc_path in DOCS:

        print(f" - dohvaćam {doc_path}")

        text = fetch_docx_text(doc_path)

        if not text:
            continue

        chunks = chunk_text(text)

        base_idx = len(all_chunks)

        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadatas.append({
                "url": doc_path,
                "chunk_id": base_idx + i,
                "excerpt": c[:200]
            })
        print(f" > Iz {doc_path} dobiveno {len(chunks)} chunkova.")

        
    print(" Dohvaćanje članaka...")
    for url in URLS:
        print(f" - dohvaćam {url}")
        text = fetch_article_text(url)
        if not text:
            print(f" ! {url} ne sadrži tekst, preskačem.")
            continue
        chunks = chunk_text(text)
        base_idx = len(all_chunks)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadatas.append({
            "url": url,
            "chunk_id": base_idx + i,
            "excerpt": c[:200]
            })
        print(f" > Iz {url} dobiveno {len(chunks)} chunkova.")
        time.sleep(1)

    if not all_chunks:
        print("Nema chunkova za indeksiranje. Izlazim.")
        return
    
    print("2) Učitavam embedding model i izračunavam embeddinge (može potrajati)...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(all_chunks, convert_to_numpy=True,
    show_progress_bar=True)

    # Normaliziramo embeddings za cosine-sličnost
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
 
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings)
    
    index_bytes = faiss.serialize_index(index)
    payload = {
        "chunks": all_chunks,
        "metadatas": metadatas,
        "embeddings": embeddings,
        "index_bytes": index_bytes,
        "model_name": EMBED_MODEL_NAME,
        "urls": URLS
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)
    

if __name__ == "__main__":
 main()

