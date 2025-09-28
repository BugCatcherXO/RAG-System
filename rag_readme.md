# RAG básico con OpenAI + Chroma (Cloud & Local)

Pipeline mínimo para ingestar documentos `.txt`, segmentarlos, embebedarlos con OpenAI, almacenarlos en Chroma (Cloud y local) y hacer QA por retrieval-augmented generation.

El código incluye carga de ficheros, chunking con solapamiento, cálculo de embeddings, alta en Chroma, búsqueda por similitud, y generación de respuesta condicionada al contexto recuperado.

## Arquitectura

**Carga de documentos** desde `news_articles/*.txt`

**Chunking**: ventanas de 1000 caracteres con solape de 30

**Embeddings** (OpenAI `text-embedding-3-small`)

**Vector store**:
- Chroma Cloud (`chromadb.CloudClient`) — multi-tenant
- Persistencia local (`chromadb.PersistentClient`) en `./chroma_db`

**Búsqueda** por similitud (`n_results=3`) y QA con `gpt-4.1-nano`

## Requisitos

Python 3.10+

Cuenta y API Key de OpenAI

(Opcional) Proyecto en Chroma Cloud con tenant y token

### requirements.txt (mínimo recomendado)

```txt
openai>=1.40.0
chromadb>=0.5.0
numpy>=1.24
python-dotenv>=1.0.1  # opcional, para manejar .env
```

## Variables de entorno

Crea un `.env` (o exporta en tu shell):

```bash
OPENAI_API=sk-...
CHROMA_TENANT=your-tenant-id
CHROMADB_TOKEN=your-chroma-token
```

**Nota:** `CHROMA_TENANT` y `CHROMADB_TOKEN` son necesarios para usar Chroma Cloud. Si no los tienes, puedes comentar el bloque de `CloudClient` y trabajar solo con `PersistentClient` local.

## Estructura de proyecto

```
RAG1/
├─ rag.ipynb                     # Notebook principal (o script equivalente)
├─ requirements.txt
├─ news_articles/                # Coloca aquí tus .txt
│  ├─ 01_noticia.txt
│  └─ ...
└─ chroma_db/                    # Se crea automáticamente (persistencia local)
```

## Instalación

```bash
# 1) Crear entorno (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) Instalar dependencias
pip install -r requirements.txt
```

## Datos de ejemplo

Coloca uno o más `.txt` en `news_articles/`. El loader solo lee archivos con extensión `.txt`.

## Ejecución

### Opción A) VS Code / Jupyter

1. Abre `rag.ipynb` en VS Code
2. Ejecuta las celdas en orden (verás mensajes de progreso)

### Opción B) Script (si lo migras a .py)

Asegúrate de mover el código tal cual a un script y ejecútalo:

```bash
python rag.py
```

## Parámetros clave

### Modelos
- **LLM**: `gpt-4.1-nano` (rápido y barato para QA breve)
- **Embeddings**: `text-embedding-3-small`

### Decodificación
- `TEMPERATUER = 0.7` *(nota: en el código hay un pequeño typo en el nombre de la variable; puedes renombrar a `TEMPERATURE` por claridad)*
- `MAX_TOKENS = 100`

### Chunking
- `chunk_size = 1000`
- `chunk_overlap = 30`

### Retrieval
- `n_results = 3` (ajústalo según corpus)

## Qué hace cada bloque

**OpenAIEmbeddingFunction**: adaptador que llama a la API de embeddings y devuelve `np.float32`

**Carga y chunking**: `load_documents_from_directory` + `split_text`

**Embeddings**: `get_openai_embedding(text)` y bucle que añade embedding a cada chunk

**Ingesta en Chroma**:
- Cloud: `chroma.CloudClient(...).get_or_create_collection(...)`
- Local: `chromadb.PersistentClient(path="./chroma_db")`

**Consulta**: `query_documents(question, n_results)` devuelve top-k con ids, distances, text

**Generación**: `generate_response(question, retrieved_chunks)` construye un prompt estricto: "Responde solo con el contexto, en ≤3 frases."

## Ejemplos de uso

### 1) Smoke test del LLM

Imprime la respuesta a "¿Cuál es la capital de Francia?" para validar credenciales/modelo.

### 2) Búsqueda + QA (en inglés)

```python
question = "What threat do Google and OpenAI face according to the internal memo, and why might open-source LLMs erode their competitive moat?"
retrieved = query_documents(question, n_results=3)
answer = generate_response(question, retrieved)
print(answer)
```

### 3) Búsqueda en español

```python
question = "¿Qué menciona el artículo sobre Databricks?"
results = query_documents(question, n_results=3)
for r in results:
    print(r["id"], r["distance"], r["text"][:200])
```

## Buenas prácticas y notas

**Idempotencia**: `get_or_create_collection` evita duplicaciones por nombre, pero añadir los mismos ids dos veces fallará. Asegúrate de que los ids sean únicos o limpia la colección antes de reingestar.

**Dimensiones de embedding**: `text-embedding-3-small` → 1536 dims (se maneja automáticamente).

**Métricas**: Chroma retorna distancia (menor = más similar) por defecto. Ajusta score según tu preferencia.

**Coste**: El chunking fino + muchas consultas de embedding puede elevar coste. Evalúa `chunk_size` y deduplicación de chunks.

**Seguridad**: No subas llaves al repo. Usa `.env`/variables de entorno.

**Cloud vs Local**: Puedes trabajar solo con local si no tienes credenciales de Chroma Cloud.

## Troubleshooting

**`AssertionError: Missing OPENAI_API`**  
→ Exporta `OPENAI_API` o crea `.env`.

**`chromadb.CloudClient` credenciales inválidas**  
→ Verifica `CHROMA_TENANT` y `CHROMADB_TOKEN`. Si no usas Cloud, comenta ese bloque y usa solo `PersistentClient`.

**No se cargan documentos**  
→ Comprueba que `news_articles/` existe y contiene `.txt`.

**Duplicados al añadir**  
→ Limpia la colección o cambia ids (e.g., añade hash del contenido).

## Extensiones rápidas

Añadir metadatos (título, fecha, fuente) y guardarlos en Chroma para filtros

Cambiar a embeddings más grandes (`text-embedding-3-large`) si necesitas mayor recall

Implementar re-ranking (e.g., por MMR) antes de pasar al LLM

Exponer una CLI o FastAPI con endpoints `/ingest` y `/ask`

## Licencia

MIT (o la que prefieras).

## Checklist antes de correr

- [ ] `.env` con `OPENAI_API` (y, si usas Cloud, `CHROMA_TENANT` + `CHROMADB_TOKEN`)
- [ ] `pip install -r requirements.txt`
- [ ] `news_articles/*.txt` creados
- [ ] Ejecutar el notebook en orden

---

Si quieres, puedo arreglar el typo de `TEMPERATUER` → `TEMPERATURE`, añadir un script `.py` ejecutable, o crear una API mínima.