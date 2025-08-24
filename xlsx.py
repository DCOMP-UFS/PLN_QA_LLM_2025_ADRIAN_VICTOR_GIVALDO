import pandas as pd
from docx import Document
from transformers import pipeline
import torch

def prepare_metadata(tables_content):
    data = []
    
    # Processar tabela índice
    index_table = tables_content[0]
    table_descriptions = {row[0]: row[2] for row in index_table[1:]}
    
    current_table = None
    
    # Processar tabelas subsequentes
    for table in tables_content[1:]:
        # Identificar tabelas de nome
        if len(table) == 1 and len(table[0]) == 1:
            current_table = table[0][0]
            continue
        
        # Processar tabelas de campos
        if len(table) > 1 and table[0][0] == 'NOME DO CAMPO':
            for row in table[1:]:
                if any(cell.strip() for cell in row):
                    data.append({
                        'table_name': current_table,
                        'description': table_descriptions.get(current_table, ''),
                        'field_name_local': row[0],
                        'data_type_local': row[1],
                        'foreign_key_local': row[2],
                        'field_name_federal': row[3],
                        'data_type_federal': row[4],
                        'foreign_key_federal': row[5],
                        'is_primary': row[6],
                        'is_nullable': row[7],
                        'description_field': row[8],
                        'domains': row[9]
                    })
    
    return pd.DataFrame(data)

# Preparar o DataFrame

def extract_table_data(file_path):
    document = Document(file_path)
    all_tables_data = []

    for table_index, table in enumerate(document.tables):
        current_table_data = []
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                row_cells.append(cell.text)
            current_table_data.append(row_cells)
        all_tables_data.append(current_table_data)
    return all_tables_data


# Usage example:
file_path = './datasets/DICIONARIO_DE_DADOS.docx'
tables_content = extract_table_data(file_path)

index_table = tables_content[0]

# Converter para DataFrame
index_df = pd.DataFrame(index_table[1:], columns=index_table[0])

# Criar dicionário de mapeamento nome local → descrição
table_descriptions = dict(zip(
    index_df['NOME DA TABELA (BANCO LOCAL)'],
    index_df['DESCRIÇÃO DA TABELA']
))

all_tables_data = []
current_table_name = None

# Iterar pelas tabelas a partir da segunda (índice 1)
for i, table_data in enumerate(tables_content[1:]):
    # Se a tabela tem apenas uma célula com o nome
    if len(table_data) == 1 and len(table_data[0]) == 1:
        current_table_name = table_data[0][0]
        continue

    # Se encontrou cabeçalho de campos
    if table_data[0] == ['NOME DO CAMPO', 'TIPO', 'FOREIGN KEY', 'NOME DO CAMPO', 'TIPO',
                         'FOREIGN KEY', 'PRIMARY KEY', 'NULL', 'DESCRIÇÃO', 'DOMÍNIOS']:

        # Processar linhas de campos
        for row in table_data[1:]:
            # Ignorar linhas vazias
            if not any(cell.strip() for cell in row):
                continue

            all_tables_data.append({
                'table_name': current_table_name,
                'description': table_descriptions.get(current_table_name, ''),
                'field_name': row[0] or row[3],  # Usar campo local ou federal
                'data_type': row[1] or row[4],    # Usar tipo local ou federal
                'is_primary': row[6],
                'is_nullable': row[7],
                'description_field': row[8],
                'domains': row[9]
            })

df = pd.DataFrame(all_tables_data)

# Exemplo de resultado
print(df[['table_name', 'field_name', 'data_type', 'description_field']].head())

# Criar representação textual para embeddings
df['text_representation'] = df.apply(
    lambda x: f"Tabela: {x['table_name']} ({x['description']})\n"
    f"Campo: {x['field_name']} ({x['data_type']})\n"
    f"Descrição: {x['description_field']}\n"
    f"Domínios: {x['domains']}\n"
    f"Chave primária: {x['is_primary']}, Nulo: {x['is_nullable']}",
    axis=1
)
metadata_df = prepare_metadata(tables_content)

# 1. Carregar o modelo (escolha um modelo adequado ao seu hardware)
# Alternativas: "codellama/CodeLlama-7b-hf", "bigcode/starcoderbase"
#  "bigcode/starcoder2-3b"
model_name = "bigcode/starcoder2-3b"

# Inicializar o pipeline de geração de texto
generator = pipeline(
    "text-generation",
    model=model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16
)


def query_metadata(question, df):
    # 2. Preparar o prompt
    prompt = f"""<|system|>
Você é um especialista em Python e pandas. Dado um DataFrame `df` com a seguinte estrutura:
Colunas: {', '.join(df.columns)}
Primeiras linhas:
{df.head(3).to_string()}

Gere APENAS código Python executável para responder à pergunta.
Atribua o resultado à variável `result`.
</s>
<|user|>
Pergunta: {question}
Código:</s>
<|assistant|>
```python
"""

    # 3. Gerar o código
    response = generator(
        prompt,
        max_new_tokens=300,
        temperature=0.1,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    # 4. Extrair o código da resposta
    generated_text = response[0]['generated_text']
    code_start = generated_text.find("```python") + len("```python")
    code_end = generated_text.find("```", code_start)
    code = generated_text[code_start:code_end].strip()

    print("Código gerado:")
    print(code)

    # 5. Executar o código
    local_vars = {'df': df.copy()}
    try:
        exec(code, globals(), local_vars)
        return local_vars.get('result', "Nenhum resultado encontrado")
    except Exception as e:
        return f"Erro na execução: {str(e)}"


# Exemplo de uso
question = "Qual o nome do campo que armazena o nome do profissional na tabela LFCES018?"
result = query_metadata(question, metadata_df)
print("\nResposta:")
print(result)
# Juntar todas as representações
# context_text = "\n\n".join(df['text_representation'].tolist())


# for i, table_data in enumerate(tables_content):
#     print(f"\nContent of Table {i+1}:")
#     for row in table_data:
#         print(row)

# table_data = tables_content[];
# main_table = table_data[0]


# print(table_data)
# for row in first_table:
#     print(row, '\n')
# print(first_table)
# import numpy as np
# import PyPDF2
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import pipeline


# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         for page in reader.pages:
#             text += page.extract_text().strip()
#     return text


# def split_text(text, chunk_size=500):
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# def retrieve_relevant_chunks(query, model, document_embeddings, chunks, top_k=3):
#     query_embedding = model.encode([query])
#     similarities = cosine_similarity(query_embedding, document_embeddings)[0]
#     top_indices = similarities.argsort()[-top_k:][::-1]
#     return [chunks[i] for i in top_indices]

# # pierreguillou/bert-large-cased-squad-v1.1-portuguese
# # timpal0l/mdeberta-v3-base-squad2
# #rachen/matscibert-squad-accelerate
# # monologg/koelectra-small-v2-distilled-korquad-384
# # rachen/matscibert-squad-accelerate
# # mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
# if __name__ == "__main__":
#     # 1. Carregar o modelo de QA
#     qa_pipeline = pipeline(
#         "question-answering",
#         model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
#         tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
#     )
#     # paraphrase-multilingual-mpnet-base-v2
#     # multi-qa-MiniLM-L6-cos-v1
#     # 2. Carregar o modelo para embeddings
#     embedding_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

#     pdf_file = "./datasets/doencas-respiratorias.pdf"
#     pdf_text = extract_text_from_pdf(pdf_file)
#     chunks = split_text(pdf_text)
#     document_embeddings = embedding_model.encode(chunks)

#     while True:
#         query = input("\nPergunta: ")
#         if query.lower() in ["sair", "exit"]:
#             break

#         relevant_chunks = retrieve_relevant_chunks(
#             query, embedding_model, document_embeddings, chunks)

#         context = " ".join(relevant_chunks)[:1000]

#         answer = qa_pipeline(question=query, context=context)

#         print(f"\nResposta: {answer['answer']}")
#         print(f"Confiança: {answer['score']:.2%}")
#         print(f"\nTrecho de referência: {relevant_chunks[0][:200]}...")
