"""
Servidor MCP — Búsqueda semántica genérica en Pinecone.

Conecta CUALQUIER índice de Pinecone con un cliente MCP (p. ej. Claude Desktop)
para hacer búsqueda semántica de SOLO LECTURA. No está atado a ningún dominio.

>> SOLO necesitas tu clave de Pinecone. <<
Por defecto, el embebido de la consulta lo hace el propio Pinecone (Pinecone
Inference), así que no requiere ninguna otra clave.

Embebido (cómo se convierte la consulta en vector):
  - Por defecto: Pinecone Inference (modelo `multilingual-e5-large`). Solo Pinecone.
  - Opcional: OpenAI, SOLO si defines OPENAI_API_KEY y un EMBED_MODEL de OpenAI
    (p. ej. text-embedding-3-large). Útil si tu índice se creó con embeddings
    de OpenAI.

El modelo de embebido DEBE coincidir con el que se usó para crear el índice
(mismo modelo y misma dimensión).

Configuración (.env — ver .env.example):
  PINECONE_API_KEY        (obligatoria)
  PINECONE_INDEX          (obligatoria) índice por defecto donde buscar
  EMBED_MODEL             (opcional) modelo de embebido
  OPENAI_API_KEY          (opcional)  solo para índices embebidos con OpenAI
  PINECONE_NAMESPACE      (opcional) namespace por defecto
  TEXT_FIELD              (opcional) campo de metadata con el texto a mostrar
  PINECONE_INDEX_ALIASES  (opcional) JSON {"alias":"nombre-real-indice", ...}
"""

import json
import logging
import os
import sys

# Silenciar logs de librerías para no ensuciar el canal MCP (stdio)
for _name in ("pinecone", "httpx", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from dotenv import load_dotenv
from pinecone import Pinecone
from mcp.server.fastmcp import FastMCP

# Cargar variables del .env que está junto a este archivo
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
DEFAULT_INDEX = os.environ.get("PINECONE_INDEX", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "").strip()
DEFAULT_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "").strip()
TEXT_FIELD = os.environ.get("TEXT_FIELD", "").strip()

try:
    INDEX_ALIASES = json.loads(os.environ.get("PINECONE_INDEX_ALIASES", "") or "{}")
    if not isinstance(INDEX_ALIASES, dict):
        INDEX_ALIASES = {}
except json.JSONDecodeError:
    INDEX_ALIASES = {}

_TEXT_CANDIDATES = [
    "text", "texto", "content", "contenido", "chunk_text",
    "page_content", "body", "passage",
]

if not PINECONE_API_KEY:
    sys.stderr.write("ERROR: falta PINECONE_API_KEY en el .env\n")
    sys.exit(1)
if not DEFAULT_INDEX:
    sys.stderr.write("ERROR: falta PINECONE_INDEX en el .env (índice por defecto)\n")
    sys.exit(1)

# --- Selección del proveedor de embebido ---------------------------------
# OpenAI solo si hay clave Y (no se fija modelo o el modelo es de OpenAI).
_model_is_openai = EMBED_MODEL.startswith("text-embedding")
if OPENAI_API_KEY and (not EMBED_MODEL or _model_is_openai):
    EMBED_PROVIDER = "openai"
    EMBED_MODEL = EMBED_MODEL or "text-embedding-3-large"
else:
    EMBED_PROVIDER = "pinecone"
    EMBED_MODEL = EMBED_MODEL or "multilingual-e5-large"

_pc = Pinecone(api_key=PINECONE_API_KEY)
_index_cache: dict[str, object] = {}
_openai_client = None  # se crea de forma perezosa solo si se usa OpenAI

mcp = FastMCP("pinecone-search")


def _resolver_indice(indice: str) -> str:
    indice = (indice or "").strip()
    if not indice:
        return DEFAULT_INDEX
    if indice in INDEX_ALIASES:
        return INDEX_ALIASES[indice]
    return indice


def _get_index(nombre_real: str):
    if nombre_real not in _index_cache:
        _index_cache[nombre_real] = _pc.Index(nombre_real)
    return _index_cache[nombre_real]


def _embed(texto: str) -> list[float]:
    """Convierte el texto en vector. Pinecone por defecto; OpenAI si está activo."""
    if EMBED_PROVIDER == "openai":
        global _openai_client
        if _openai_client is None:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        resp = _openai_client.embeddings.create(model=EMBED_MODEL, input=texto)
        return resp.data[0].embedding
    # Pinecone Inference (solo clave de Pinecone)
    r = _pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[texto],
        parameters={"input_type": "query"},
    )
    d = r.data[0]
    return d["values"] if isinstance(d, dict) else d.values


def _extraer_texto(meta: dict) -> tuple[str | None, str | None]:
    if TEXT_FIELD and TEXT_FIELD in meta:
        return TEXT_FIELD, str(meta[TEXT_FIELD])
    for c in _TEXT_CANDIDATES:
        if c in meta and meta[c]:
            return c, str(meta[c])
    return None, None


@mcp.tool()
def pinecone_search(
    query: str,
    index: str = "",
    top_k: int = 5,
    namespace: str = "",
) -> str:
    """Búsqueda semántica en un índice de Pinecone. Embebe la consulta y
    devuelve los registros más parecidos con su puntuación de similitud y sus
    metadatos.

    Args:
        query: Texto a buscar en lenguaje natural.
        index: Índice donde buscar. Vacío = índice por defecto (PINECONE_INDEX).
            Acepta el nombre real del índice o un alias de PINECONE_INDEX_ALIASES.
        top_k: Número de resultados a devolver (1-50, por defecto 5).
        namespace: Namespace de Pinecone (vacío = el del .env o el por defecto).

    Returns:
        Texto con los resultados: puntuación, texto principal (si existe en los
        metadatos) y el resto de metadatos de cada coincidencia.
    """
    query = (query or "").strip()
    if not query:
        return "Error: la consulta está vacía."

    top_k = max(1, min(int(top_k), 50))
    nombre_real = _resolver_indice(index)
    ns = (namespace or DEFAULT_NAMESPACE or "").strip()

    try:
        vector = _embed(query)
    except Exception as e:  # noqa: BLE001
        return f"Error al generar el embebido ({EMBED_PROVIDER}/{EMBED_MODEL}): {e}"

    try:
        idx = _get_index(nombre_real)
        kwargs = {"vector": vector, "top_k": top_k, "include_metadata": True}
        if ns:
            kwargs["namespace"] = ns
        res = idx.query(**kwargs)
    except Exception as e:  # noqa: BLE001
        return f"Error al consultar Pinecone (índice '{nombre_real}'): {e}"

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    if not matches:
        return f"Sin resultados en el índice '{nombre_real}' para: {query!r}"

    lineas = [
        f"Resultados de '{nombre_real}'"
        + (f" (namespace: {ns})" if ns else "")
        + f" para: {query!r}",
        f"({len(matches)} coincidencias, ordenadas por relevancia)\n",
    ]

    for i, m in enumerate(matches, 1):
        if isinstance(m, dict):
            score = m.get("score", 0.0) or 0.0
            meta = m.get("metadata", {}) or {}
            _id = m.get("id", "")
        else:
            score = getattr(m, "score", 0.0) or 0.0
            meta = getattr(m, "metadata", {}) or {}
            _id = getattr(m, "id", "")

        meta = dict(meta)
        text_key, texto = _extraer_texto(meta)

        lineas.append(f"--- Resultado {i} | similitud {score:.3f} | id: {_id} ---")

        if texto is not None:
            if len(texto) > 1500:
                texto = texto[:1500] + " […]"
            lineas.append(texto)

        otros = []
        for k, v in meta.items():
            if k == text_key:
                continue
            sval = str(v)
            if len(sval) > 300:
                sval = sval[:300] + " […]"
            otros.append(f"  {k}: {sval}")
        if otros:
            lineas.append("metadatos:")
            lineas.extend(otros)

        lineas.append("")

    return "\n".join(lineas)


if __name__ == "__main__":
    mcp.run()
