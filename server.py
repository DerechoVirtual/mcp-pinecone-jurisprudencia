"""
Servidor MCP — Búsqueda semántica en Pinecone (jurisprudencia y normativa).

Conecta los índices de Pinecone de Derecho Virtual con Claude (Claude Desktop)
para hacer búsqueda semántica de SOLO LECTURA sobre jurisprudencia y normativa
española ya indexada.

Embeddings: OpenAI text-embedding-3-large (3072 dim) — debe coincidir con la
dimensión de los índices.

Autor: generado para Carlos Rivero (Derecho Virtual).
"""

import logging
import os
import sys

# Silenciar logs de las librerías para no ensuciar el canal MCP (stdio)
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from mcp.server.fastmcp import FastMCP

# Cargar claves desde el .env que está junto a este archivo
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    sys.stderr.write(
        "ERROR: faltan PINECONE_API_KEY u OPENAI_API_KEY en el .env\n"
    )
    sys.exit(1)

# Índices disponibles (todos 3072 dim, modelo text-embedding-3-large).
# clave amigable -> nombre real del índice en Pinecone
INDICES = {
    "familia": "jurisprudencia-derecho-familia",   # ~592 STS de derecho de familia
    "lec": "lec-espana",                            # Ley de Enjuiciamiento Civil (Ley 1/2000)
    "temario-justicia": "rag-temario-justicia",     # temario oposiciones Justicia
    "temario-justicia-openai": "rag-temario-justicia-openai",
}
INDICE_POR_DEFECTO = "familia"

# Clientes (se crean una vez al arrancar)
_openai = OpenAI(api_key=OPENAI_API_KEY)
_pc = Pinecone(api_key=PINECONE_API_KEY)
_index_cache: dict[str, object] = {}

mcp = FastMCP("pinecone-jurisprudencia")


def _resolver_indice(indice: str) -> str:
    """Acepta una clave amigable ('familia') o el nombre real del índice."""
    if indice in INDICES:
        return INDICES[indice]
    # Si pasan directamente el nombre real, lo aceptamos también
    if indice in INDICES.values():
        return indice
    return INDICES[INDICE_POR_DEFECTO]


def _get_index(nombre_real: str):
    if nombre_real not in _index_cache:
        _index_cache[nombre_real] = _pc.Index(nombre_real)
    return _index_cache[nombre_real]


def _embed(texto: str) -> list[float]:
    resp = _openai.embeddings.create(model=EMBED_MODEL, input=texto)
    return resp.data[0].embedding


@mcp.tool()
def buscar_jurisprudencia(
    consulta: str,
    indice: str = INDICE_POR_DEFECTO,
    top_k: int = 5,
) -> str:
    """Busca semánticamente en una base de datos vectorial (Pinecone) de
    jurisprudencia y normativa española y devuelve los fragmentos más
    relevantes con su cita y metadatos.

    Args:
        consulta: Pregunta o texto a buscar (lenguaje natural). Ej.:
            "extinción de la pensión compensatoria por convivencia marital".
        indice: Base de datos donde buscar. Opciones:
            - "familia": jurisprudencia del Tribunal Supremo en derecho de
              familia (divorcio, custodia, pensión compensatoria, alimentos,
              vivienda familiar, régimen económico). [por defecto]
            - "lec": articulado de la Ley de Enjuiciamiento Civil (Ley 1/2000).
            - "temario-justicia": temario de oposiciones de Justicia.
            - "temario-justicia-openai": variante del temario de Justicia.
        top_k: Número de fragmentos a devolver (por defecto 5, máx. 20).

    Returns:
        Texto con los fragmentos más relevantes, su cita/fuente, metadatos y
        puntuación de similitud (0-1).
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return "Error: la consulta está vacía."

    top_k = max(1, min(int(top_k), 20))
    nombre_real = _resolver_indice(indice)

    try:
        vector = _embed(consulta)
    except Exception as e:  # noqa: BLE001
        return f"Error al generar el embedding con OpenAI: {e}"

    try:
        index = _get_index(nombre_real)
        res = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as e:  # noqa: BLE001
        return f"Error al consultar Pinecone (índice '{nombre_real}'): {e}"

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    if not matches:
        return (
            f"No se encontraron resultados en el índice '{nombre_real}' "
            f"para: {consulta!r}"
        )

    lineas = [
        f"Resultados de '{nombre_real}' para: {consulta!r}",
        f"({len(matches)} fragmentos, ordenados por relevancia)\n",
    ]

    for i, m in enumerate(matches, 1):
        if isinstance(m, dict):
            score = m.get("score", 0.0)
            meta = m.get("metadata", {}) or {}
        else:
            score = getattr(m, "score", 0.0) or 0.0
            meta = getattr(m, "metadata", {}) or {}

        # Construir una cita legible a partir de los metadatos disponibles
        cita_partes = []
        for campo in ("referencia", "articulo", "rubrica", "fecha", "tribunal",
                       "materia", "numero_resolucion", "ecli", "fuente"):
            val = meta.get(campo)
            if val:
                cita_partes.append(f"{campo}: {val}")
        cita = " · ".join(cita_partes) if cita_partes else "(sin metadatos de cita)"

        texto = meta.get("texto") or meta.get("text") or "(sin texto en metadata)"
        if len(texto) > 1500:
            texto = texto[:1500] + " […]"

        link = meta.get("link") or meta.get("url")

        lineas.append(f"--- Resultado {i} | similitud {score:.3f} ---")
        lineas.append(cita)
        if link:
            lineas.append(f"Enlace: {link}")
        lineas.append(texto)
        lineas.append("")

    return "\n".join(lineas)


if __name__ == "__main__":
    mcp.run()
