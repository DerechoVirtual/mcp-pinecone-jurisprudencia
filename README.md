# Pinecone Semantic Search — Servidor MCP

Servidor [MCP](https://modelcontextprotocol.io) de **búsqueda semántica de solo
lectura** sobre **cualquier índice de [Pinecone](https://www.pinecone.io)**.
Conéctalo a Claude Desktop (u otro cliente MCP) y pregúntale en lenguaje natural;
el servidor embebe tu consulta y devuelve los registros más parecidos de tu
índice, con su puntuación de similitud y sus metadatos.

No está atado a ningún dominio: sirve para jurisprudencia, documentación técnica,
base de conocimiento, soporte, RAG… **Tú pones tu propia API key y tu propio
índice** mediante un archivo `.env`. Este repositorio **no contiene credenciales
ni datos**.

## Cómo funciona

1. Recibe una consulta de texto.
2. La convierte en un vector con OpenAI (`text-embedding-3-large` por defecto).
3. Consulta tu índice de Pinecone (`query`, `top_k`, `include_metadata`).
4. Devuelve los resultados formateados (texto + metadatos + similitud).

> El modelo de embeddings debe producir vectores de la **misma dimensión** que
> tu índice (`text-embedding-3-large` = 3072, `text-embedding-3-small` = 1536).

## Requisitos

- Python 3.10+
- Una cuenta de [Pinecone](https://app.pinecone.io) con un índice ya poblado.
- Una API key de [OpenAI](https://platform.openai.com) (para los embeddings).

## Instalación

```bash
git clone https://github.com/DerechoVirtual/mcp-pinecone-jurisprudencia.git
cd mcp-pinecone-jurisprudencia

# Dependencias (recomendado con uv; también vale pip + venv)
uv venv
uv pip install mcp openai pinecone python-dotenv

# Configura tus claves
cp .env.example .env   # en Windows: copy .env.example .env
# edita .env y rellena PINECONE_API_KEY, OPENAI_API_KEY y PINECONE_INDEX
```

## Configuración (`.env`)

| Variable | Obligatoria | Descripción |
|---|---|---|
| `PINECONE_API_KEY` | ✅ | Tu API key de Pinecone |
| `OPENAI_API_KEY` | ✅ | Tu API key de OpenAI (embeddings) |
| `PINECONE_INDEX` | ✅ | Índice por defecto donde buscar |
| `EMBED_MODEL` | ❌ | Modelo de embeddings (def. `text-embedding-3-large`) |
| `PINECONE_NAMESPACE` | ❌ | Namespace por defecto |
| `TEXT_FIELD` | ❌ | Campo de metadata con el texto (si no, se autodetecta) |
| `PINECONE_INDEX_ALIASES` | ❌ | JSON de alias → índice, p. ej. `{"docs":"mi-indice-docs"}` |

## Conectar a Claude Desktop

Edita `claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`,
macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "pinecone-search": {
      "command": "/ruta/al/.venv/Scripts/python.exe",
      "args": ["/ruta/al/server.py"]
    }
  }
}
```

Reinicia Claude Desktop por completo. Aparecerá la herramienta `pinecone_search`.

> **Windows:** Claude Desktop reescribe este archivo al guardar preferencias, así
> que añade la entrada con la app **cerrada** y luego ábrela (si no, puede
> sobrescribir tu cambio).

## La herramienta

`pinecone_search(query, index="", top_k=5, namespace="")`

- `query` — texto a buscar (lenguaje natural).
- `index` — vacío = índice por defecto; acepta el nombre real o un alias.
- `top_k` — número de resultados (1–50, def. 5).
- `namespace` — namespace de Pinecone (opcional).

## Probar sin Claude

```bash
python -c "import server; print(server.pinecone_search('tu consulta', '', 3))"
```

## Seguridad

- `.env` y `.venv/` están en `.gitignore`: **nunca** subas tus claves.
- El servidor es de **solo lectura**: solo hace `query`; no inserta ni borra.

## Licencia

MIT (ver `LICENSE`).
