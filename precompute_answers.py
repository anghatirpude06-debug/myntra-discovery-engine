import json
import os
import google.generativeai as genai
from sentence_transformers import util
from utils import load_data, load_embedder, get_corpus_embeddings

def main():
    try:
        import tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomllib.load(f)
    except ImportError:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")

    api_key = secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    queries = [
        "Why do users hesitate before buying wishlisted items?",
        "What causes the most return and refund complaints?",
        "Do users trust Myntra's product authenticity?",
        "What delivery issues do users report most often?",
        "What UX problems affect the wishlist feature?"
    ]

    tagged, clustered, ranking = load_data()
    embedder = load_embedder()
    texts = clustered['text'].astype(str).tolist()
    corpus_embeddings = get_corpus_embeddings(embedder, texts)

    os.makedirs("data", exist_ok=True)
    precomputed_file = "data/precomputed_answers.json"
    precomputed = {}
    if os.path.exists(precomputed_file):
        with open(precomputed_file, "r", encoding="utf-8") as f:
            try:
                precomputed = json.load(f)
            except Exception:
                pass

    import time
    for q in queries:
        if q in precomputed:
            print(f"Skipping (already computed): {q}")
            continue
            
        print(f"Processing: {q}")
        query_embedding = embedder.encode(q, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=8)[0]

        context_texts = []
        for i, hit in enumerate(hits):
            idx = hit['corpus_id']
            row = clustered.iloc[idx]
            context_texts.append(f"Source {i+1} ({row.get('source', 'N/A')}): {row.get('text', 'N/A')}")
        context_str = "\n\n".join(context_texts)

        prompt = f"""
You are a lead UX researcher. Answer the following question synthesizing only the provided user feedback.
Make sure to reference your sources (e.g., "Based on Source 1...").

Question: {q}

User Feedback:
{context_str}
"""
        models_to_try = ["gemini-2.5-flash-lite", "gemini-3-flash", "gemini-3.6-flash"]
        ans = None
        used_model = None
        max_retries = 3

        for m in models_to_try:
            if ans: break
            model = genai.GenerativeModel(m)
            for attempt in range(max_retries):
                try:
                    resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=300))
                    ans = resp.text
                    used_model = m
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                        wait_time = (2 ** attempt) * 5
                        print(f"Quota error for {m} on attempt {attempt+1}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error with {m}: {e}")
                        break
        
        if ans:
            precomputed[q] = {
                "answer": ans,
                "model": used_model,
                "sources": context_texts
            }
            with open(precomputed_file, "w", encoding="utf-8") as f:
                json.dump(precomputed, f, indent=4)
            print(f"Saved incremental progress for: {q}")
        else:
            print(f"Failed for query: {q}")

    print(f"Done! Fully computed: {len(precomputed)}/{len(queries)} questions.")

if __name__ == "__main__":
    main()
