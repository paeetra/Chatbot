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
        error_text = str(e)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            return (
                "Trenutno je dosegnut privremeni limit za broj upita. "
                "Pokušajte ponovno za otprilike minutu."
            )

        return f"Greška pri pozivu Gemini API-ja: {e}" 
        

st.markdown("### Kako vam možemo pomoći?")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

selected_prompt = None
predefined_answer=""
user_input=""


with col1:
   # if st.button("Trebam pomoć!"):
    #    selected_prompt = """
     #   Trebam pomoć vezanu uz pravnu ili psihološku podršku,
     #   siguran smještaj ili savjetovanje.
     #   """
     if st.button("Trebam pomoć!"):
        selected_prompt = """
        Trebam pomoć vezanu uz pravnu ili psihološku podršku,
        siguran smještaj ili savjetovanje. """

        predefined_answer = """
            Drago nam je što ste nam se obratili. 💜

            Ako prolazite kroz teško razdoblje, trebate pravnu ili psihološku pomoć, savjetovanje ili siguran smještaj, niste sami.

            Udruga B.a.B.e. pruža podršku ženama i djeci žrtvama nasilja te osobama koje se suočavaju s različitim oblicima diskriminacije i kršenja ljudskih prava.

            Naša podrška uključuje:

            1. Pravnu pomoć i savjetovanje

            • Pravno zastupanje pred institucijama i sudovima

            • Pravne informacije i stručne savjete

            • Pomoć pri razumijevanju vaših prava i mogućnosti zaštite

            2. Psihološku i psihosocijalnu podršku

            • Psihološko savjetovanje

            • Emocionalnu podršku tijekom teških situacija

            • Psihosocijalnu pomoć kroz rad stručnog tima

            3. Savjetovalište

            • Godišnje pružamo oko 2000 pravnih i psiholoških savjetovanja osobama u potrebi.

            4. Pratnju od povjerenja

            • Pružamo podršku i pratnju tijekom prijava nasilja i drugih postupaka kako biste kroz proces prošli uz podršku osobe od povjerenja.

            5. Siguran smještaj

            • Sigurna kuća udruge B.a.B.e. pruža utočište ženama i djeci žrtvama obiteljskog nasilja.

            Kontakt podaci:

            📞 SOS linija:
            0800 200 144

            📧 E-mail:
            babe@babe.hr

            📞 Sigurna kuća:
            032 414 910
            098 9824 641
            📧 E-mail: 
            sigurnakucavsz@babe.hr


            Više informacija:
            https://babe.hr/pravna-i-psiholoska-pomoc/

            Sigurna kuća:
            https://babe.hr/smjestaj-u-sigurnoj-kuci/

            Nemojte čekati! Javite nam se. Tu smo kako bismo Vas saslušali, pružili podršku i pomogli Vam pronaći siguran put dalje.
            """


with col2:
    # if st.button("Želim podržati udrugu!"):
     #  selected_prompt = """
     #   Kako mogu podržati udrugu B.a.B.e.?
      #  Zanimaju me donacije i načini uključivanja.
       # """
        if st.button("Želim podržati udrugu!"):

            selected_prompt = """
       Kako mogu podržati udrugu B.a.B.e.?
       Zanimaju me donacije i načini uključivanja.
        """
            predefined_answer = """
          Udrugu B.a.B.e. možete podržati na nekoliko načina, prvenstveno kroz donacije i uključivanje kao pojedinac ili poduzeće.

            Donacije:
            Kao pojedinac, vašom donacijom možete pomoći žrtvama obiteljskog nasilja i drugih oblika kršenja ljudskih prava. Donacije omogućavaju kontinuirano pružanje zaštite, savjeta i podrške.

            Postoje dva načina za pružanje podrške jednokratnom donacijom:

            1. Skeniranjem 2D koda:
            Sami možete odabrati iznos svoje donacije.

            2. Uplatom na IBAN račun udruge B.a.B.e.:

            • Primatelj:
            B.a.B.e., Budi aktivna. Budi emancipiran.

            • Adresa:
            Selska cesta 112a, 10000 Zagreb

            • Banka:
            Raiffeisen Bank Austria

            • IBAN:
            HR8324840081100866793

            Vašom donacijom pomažete u radu savjetovališta koje na godišnjoj razini pruža 2000 pravnih i psiholoških savjetovanja osoba u potrebi. Također, podržavate Sigurnu kuću udruge B.a.B.e. koja pruža utočište i zaštitu ženama i djeci u najtežim situacijama kada im je potrebno skloniti se od nasilnika.

            Više informacija:
            https://babe.hr/donacije/
            """
       

with col3:
   # if st.button("Zanimaju me projekti koje udruga provodi."):
    #    selected_prompt = """
    #    Koje projekte provodi udruga B.a.B.e.?
    #    """
  
    if st.button("Zanimaju me projekti koje udruga provodi."):
        selected_prompt= """
    #    Koje projekte provodi udruga B.a.B.e.?
    #    """
        predefined_answer = """

    Hvala Vam na zanimanju za rad naše udruge! Vrlo nam je drago što ste se obratili s ovim pitanjem, jer volimo dijeliti informacije o našim aktivnostima.

    Udruga B.a.B.e. provodi niz važnih projekata usmjerenih na zaštitu ženskih prava, borbu protiv rodno uvjetovanog nasilja i promicanje rodne ravnopravnosti u svim sferama društva. Naši projekti obuhvaćaju širok spektar aktivnosti, od izravne podrške žrtvama nasilja do zagovaranja promjena u zakonodavstvu i podizanja svijesti javnosti.

    Evo nekih od naših trenutnih projekata koje biste mogli pronaći zanimljivima, a koji su detaljnije opisani na našoj web stranici:

    • Putokaz za poštivanje ženskih ljudska prava
    Više informacija: https://babe.hr/projekt/putokaz-za-postivanje-zenskih-ljudska-prava/

    • Surf & Sound 2.0 (Support, Unite, Respond, Fight to Stop Online violence)
    Više informacija: https://babe.hr/projekt/surf-sound-2-0-support-unite-respond-fight-to-stop-online-violence/
   
     • Savjetovalište za prevenciju i suzbijanje svih oblika nasilnog ponašanja u obitelji
    Više informacija: https://babe.hr/projekt/savjetovaliste-za-prevenciju-i-suzbijanje-svih-oblika-nasilnog-ponasanja-u-obitelji-2/

    • Mi to možemo! - mladi ljudi kao nosioci promjena za bolju budućnost i zdravije društvo
    Više informacija: https://babe.hr/projekt/mi-to-mozemo-mladi-ljudi-kao-nosioci-promjena-za-bolju-buducnost-i-zdravije-drustvo/


    Osim navedenih projekata, udruga B.a.B.e. kontinuirano pruža i druge oblike podrške, poput rada SOS linije za žene i djecu žrtve nasilja, pružanje besplatne pravne i psihosocijalne pomoći, te provođenje edukacija i zagovaračkih aktivnosti s ciljem podizanja svijesti o važnosti ravnopravnosti i borbe protiv svih oblika diskriminacije.

    Za detaljnije informacije o svim našim trenutnim projektima, kao i o arhivi završenih projekata, pozivamo vas da posjetite našu web stranicu, točnije stranicu posvećenu projektima:

    https://www.babe.hr/projekti
        """

with col4:
   # if st.button("Zanimaju me aktivnosti udruge."):
   #     selected_prompt = """
   #     Koje aktivnosti provodi udruga B.a.B.e.?
    #    """
         if st.button("Zanimaju me aktivnosti udruge."):
            selected_prompt = """
        Koje aktivnosti provodi udruga B.a.B.e.?
        """
            
            predefined_answer = """
            Udruga B.a.B.e. provodi širok spektar aktivnosti usmjerenih na zaštitu i promicanje ljudskih prava žena, prevenciju rodno uvjetovanog nasilja i podršku žrtvama.

            Naše aktivnosti obuhvaćaju:

            1. Pravna i psihološka podrška i savjetovanje

            • Pravno zastupanje:
            Pružamo pravno zastupanje pred javnopravnim tijelima, uključujući sudove, te pred Europskim sudom za ljudska prava u Strasbourgu.

            • Pravni savjeti i informacije:
            Pružamo stručne pravne savjete i informacije osobama u potrebi.

            • Psihološka pomoć:
            Dostupna je psihološka pomoć i savjetovanje za žrtve nasilja i kršenja ljudskih prava.

            • Savjetovalište:
            Naše savjetovalište godišnje pruža oko 2000 pravnih i psiholoških savjetovanja.

            • Pratnja od povjerenja:
            Od 2020. godine pružamo uslugu pratnje osobe od povjerenja žrtvama kaznenih djela i prekršaja, osiguravajući emocionalnu i psihološku potporu tijekom prijave nasilja i drugih postupaka.

            • Psihosocijalna podrška:
            Stručni tim (socijalna radnica, psihologinja, sociologinja) pruža psihosocijalnu podršku žrtvama rodno uvjetovanog nasilja radi egzistencijalne zaštite i oporavka.

            2. Siguran smještaj

            • Sigurna kuća:
            Osiguravamo smještaj u Sigurnoj kući koja je utočište i zaštita ženama žrtvama obiteljskog nasilja i njihovoj djeci, pružajući im sigurno sklonište od nasilnika. Boravak u Sigurnoj kući može trajati do godinu dana.

            3. Kampanje i inicijative

            • Aktivno sudjelujemo u javnim kampanjama i inicijativama usmjerenim na podizanje svijesti o problemu rodno uvjetovanog nasilja, diskriminaciji i potrebi za ravnopravnošću spolova.

            • Zagovaramo promjene zakona i javnih politika kako bismo osigurali bolju zaštitu i prava žena.

            4. Edukacija i prevencija

            • Organiziramo seminare, radionice i edukativne programe za stručnjake i širu javnost s ciljem prevencije nasilja, edukacije o ljudskim pravima i promicanja rodne ravnopravnosti.

            5. Projekti

            • Udruga provodi brojne projekte, često u suradnji s domaćim i međunarodnim partnerima, koji su usmjereni na specifične aspekte borbe protiv diskriminacije, promicanja ravnopravnosti, podrške žrtvama i jačanja kapaciteta civilnog društva.

            Sve ove aktivnosti usmjerene su na ostvarenje naše misije – izgradnje društva u kojem su žene sigurne, ravnopravne i slobodne od svih oblika diskriminacije i nasilja.

            Više informacija o našim aktivnostima i projektima možete pronaći na našoj web stranici:

            • Pravna i psihološka pomoć:
            https://babe.hr/pravna-i-psiholoska-pomoc/

            Više informacija o radu udruge:
            https://babe.hr/
            """

user_text = st.chat_input("Napiši poruku...", key="main_chat_input")



#if selected_prompt:
    #user_input = selected_prompt
if predefined_answer:

    # prikaži pitanje korisnika
    st.session_state.messages.append({
        "role": "user",
        "content": selected_prompt
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": predefined_answer
    })

elif user_text:
    user_input = user_text
else:
    user_input = None

if user_input:

    import time

    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    now = time.time()
    cooldown_seconds = 15

    if now - st.session_state.last_request_time < cooldown_seconds:
        st.warning("Molimo pričekajte nekoliko sekundi prije slanja novog pitanja.")
        st.stop()

    st.session_state.last_request_time = now
    
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
        for m in st.session_state.messages [-6:]:
            if m["role"] == "system":
                continue

            role = m["role"].upper()
            content = m["content"]
            convo.append(f"{role}: {content}")

        prompt_text = "\n".join(convo) + "Odgovori na hrvatskom jeziku.Koristi samo i isključivo informacije koje se mogu pronaći na web stranici udruge B.a.B.e. Ako nisi siguran u odgovor, reci korisniku da provjeri informacije na službenoj stranici https://babe.hr/  Dodaj linkove na kojima se može pronaći više informacija."
        
        answer = call_gemini(prompt_text)


### ovo ostaje i u novom kodu    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer})


for msg in st.session_state.messages:

      # NE prikazuj system prompt korisniku
    if msg["role"] == "system":
        continue

    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

