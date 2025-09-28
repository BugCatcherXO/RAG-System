# RAG básico con OpenAI + Chroma

Pipeline para ingestar documentos .txt, segmentarlos, embebedarlos con OpenAI, almacenarlos en Chroma y hacer QA por retrieval-augmented generation.

## Requisitos

- Python 3.10+
- OpenAI API Key
- (Opcional) Chroma Cloud tenant y token

```txt
openai>=1.40.0
chromadb>=0.5.0
numpy>=1.24
python-dotenv>=1.0.1
```

## Configuración

Crear `.env`:

```bash
OPENAI_API=sk-...
CHROMA_TENANT=your-tenant-id
CHROMADB_TOKEN=your-chroma-token
```

## Estructura

```
RAG1/
├─ rag.ipynb
├─ requirements.txt
├─ news_articles/*.txt
└─ chroma_db/
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Parámetros

- **Chunking**: 1000 caracteres, solapamiento 30
- **Embeddings**: `text-embedding-3-small`
- **LLM**: `gpt-4.1-nano`
- **Retrieval**: 3 resultados por consulta

## Uso

### Búsqueda

```python
question = "What are the main findings?"
results = query_documents(question, n_results=3)
```

### QA

```python
retrieved = query_documents(question, n_results=3)
answer = generate_response(question, retrieved)
```

## Troubleshooting

- **Missing OPENAI_API**: Verificar `.env`
- **Chroma credenciales**: Comprobar `CHROMA_TENANT` y `CHROMADB_TOKEN`
- **No documentos**: Verificar `news_articles/` con archivos `.txt`
- **IDs duplicados**: Limpiar colección o cambiar IDs

## Extensiones

- Metadatos en chunks
- Re-ranking (MMR)
- API REST con FastAPI
- Embeddings más grandes (`text-embedding-3-large`)