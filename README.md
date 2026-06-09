# MCP Pinecone — Búsqueda semántica de jurisprudencia (Derecho Virtual)

Servidor MCP de **solo lectura** que conecta los índices de Pinecone con
**Claude Desktop** para hacer búsqueda semántica de jurisprudencia y normativa
española ya indexada.

## Herramienta

`buscar_jurisprudencia(consulta, indice="familia", top_k=5)` — embebe la consulta
con OpenAI `text-embedding-3-large` (3072 dim) y devuelve los fragmentos más
parecidos del índice, con cita, enlace a vLex y puntuación de similitud.

### Índices disponibles (parámetro `indice`)

| clave | índice Pinecone | contenido |
|---|---|---|
| `familia` (def.) | `jurisprudencia-derecho-familia` | ~592 STS de derecho de familia |
| `lec` | `lec-espana` | Ley de Enjuiciamiento Civil (Ley 1/2000) |
| `temario-justicia` | `rag-temario-justicia` | temario oposiciones Justicia |
| `temario-justicia-openai` | `rag-temario-justicia-openai` | variante del temario |

## Archivos

- `server.py` — el servidor MCP (FastMCP, stdio).
- `.env` — claves `PINECONE_API_KEY`, `OPENAI_API_KEY`, `EMBED_MODEL`. **Secreto.**
- `pyproject.toml` — dependencias.
- `.venv/` — entorno virtual (creado con `uv`).

## Cómo está conectado a Claude Desktop

Registrado en `%APPDATA%\Claude\claude_desktop_config.json` como
`pinecone-jurisprudencia`, llamando al Python del `.venv` con `server.py`.

**Para activarlo: cerrar y volver a abrir Claude Desktop por completo.** Luego
aparecerá la herramienta `buscar_jurisprudencia`.

## Probar manualmente

```powershell
cd "C:\Users\carlo\OneDrive\Documentos\antigravity\pinecone y jurisprudencia"
.venv\Scripts\python.exe -c "import server; print(server.buscar_jurisprudencia('pensión compensatoria temporal', 'familia', 3))"
```

## Ampliar

- Añadir un índice nuevo: añade una entrada al dict `INDICES` en `server.py`.
- Es de solo lectura por diseño (solo busca; no inserta ni borra).
