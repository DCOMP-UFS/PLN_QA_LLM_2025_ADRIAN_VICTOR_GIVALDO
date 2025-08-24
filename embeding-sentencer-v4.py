import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text().strip()
    return text


def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_relevant_chunks(query, model, document_embeddings, chunks, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# pierreguillou/bert-large-cased-squad-v1.1-portuguese
# timpal0l/mdeberta-v3-base-squad2
#rachen/matscibert-squad-accelerate
# monologg/koelectra-small-v2-distilled-korquad-384
# rachen/matscibert-squad-accelerate
# mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
if __name__ == "__main__":
    # 1. Carregar o modelo de QA
    qa_pipeline = pipeline(
        "question-answering",
        model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
        tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
    )
    # paraphrase-multilingual-mpnet-base-v2	
    # multi-qa-MiniLM-L6-cos-v1
    # 2. Carregar o modelo para embeddings
    embedding_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

    pdf_file = "./datasets/doencas-respiratorias.pdf"
    pdf_text = extract_text_from_pdf(pdf_file)
    chunks = split_text(pdf_text)
    document_embeddings = embedding_model.encode(chunks)

    while True:
        query = input("\nPergunta: ")
        if query.lower() in ["sair", "exit"]:
            break

        relevant_chunks = retrieve_relevant_chunks(
            query, embedding_model, document_embeddings, chunks)

        context = " ".join(relevant_chunks)[:1000]

        answer = qa_pipeline(question=query, context=context)

        print(f"\nResposta: {answer['answer']}")
        print(f"Confiança: {answer['score']:.2%}")
        print(f"\nTrecho de referência: {relevant_chunks[0][:200]}...")
