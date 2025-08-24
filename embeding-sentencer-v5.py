import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re
import nltk
import time


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


def preprocess_text(text):
    """Corrige problemas comuns de extração de PDF"""
    # 1. Remove quebras de linha entre palavras minúsculas (palavras divididas)
    text = re.sub(r'(?<=[a-z])\n(?=[a-z])', '', text)

    # 2. Remove quebras de linha após hífens (palavras hifenizadas)
    text = re.sub(r'-\n', '', text)

    # 3. Preserva quebras de parágrafo (mantém múltiplas quebras)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. Corrige espaçamento entre palavras e números
    text = re.sub(r'(\D)\n(\d)', r'\1 \2', text)

    # 5. Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def split_sentences(text):
    """Divide o texto preservando estrutura de listas e tabelas"""
    # Pré-processamento básico
    text = preprocess_text(text)

    # Divisão inicial em blocos (parágrafos/tabelas)
    blocks = re.split(r'(\n\s*\n)', text)

    chunks = []
    for block in blocks:
        if block.strip():
            # Divide blocos longos em sentenças
            sentences = nltk.sent_tokenize(block)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < 500:
                    current_chunk += " " + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks


# pierreguillou/bert-base-cased-squad-v1.1-portuguese
# timpal0l/mdeberta-v3-base-squad2
# mrm8488/bert-base-portuguese-cased-finetuned-squad-v1-pt
if __name__ == "__main__":
  
    # 1. Carregar o modelo de QA
    qa_pipeline = pipeline(
        "question-answering",
        model="pierreguillou/bert-base-cased-squad-v1.1-portuguese",
        tokenizer="pierreguillou/bert-base-cased-squad-v1.1-portuguese",
        device=0
    )

    embedding_model = SentenceTransformer(
        'multi-qa-MiniLM-L6-cos-v1', device='cuda')

    pdf_file = "./datasets/doencas-respiratorias.pdf"
    raw_text = extract_text_from_pdf(pdf_file)
    # pdf_text = preprocess_text(raw_text)
    chunks = split_sentences(raw_text)
    document_embeddings = embedding_model.encode(chunks)
    question = ''
    while True:
        if (question != ''):
            query = question
        else:
            query = input("\nPergunta: ")
            if query.lower() in ["sair", "exit"]:
                break
        start = time.time()
        relevant_chunks = retrieve_relevant_chunks(
            query, embedding_model, document_embeddings, chunks)

        context = " ".join(relevant_chunks)[:500]
        print(relevant_chunks)
        answer = qa_pipeline(question=query, context=context)

        print(f"\nResposta: {answer['answer']}")
        print(f"Score: {answer['score']:.2%}")
        print(f"\nTrecho de referência: {relevant_chunks[0][:500]}...")
        end = time.time()
        print(f"Tempo de execução: {end - start:.4f} segundos")
        if (question != ''):
            break
