"""Central configuration. Values can be overridden via environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
UPLOAD_DIR = DATA_DIR / "uploads"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

for _d in (CHROMA_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "anthropic":
    KEY_ENV_NAME = "ANTHROPIC_API_KEY"
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    BASE_URL = None
    MODEL = os.getenv("LLM_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))
    FAST_MODEL = os.getenv("LLM_FAST_MODEL", os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-5-5"))
elif LLM_PROVIDER == "custom":
    KEY_ENV_NAME = "LLM_API_KEY"
    API_KEY = os.getenv("LLM_API_KEY", "")
    BASE_URL = os.getenv("LLM_BASE_URL", "")
    MODEL = os.getenv("LLM_MODEL", "")
    FAST_MODEL = os.getenv("LLM_FAST_MODEL", MODEL)
else:  # groq
    LLM_PROVIDER = "groq"
    KEY_ENV_NAME = "GROQ_API_KEY"
    API_KEY = os.getenv("GROQ_API_KEY", "")
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    FAST_MODEL = os.getenv("LLM_FAST_MODEL", "openai/gpt-oss-20b")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 6))

CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", 3000))
HISTORY_KEEP_TURNS = int(os.getenv("HISTORY_KEEP_TURNS", 3))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", 1))

MAX_UPLOAD_MB = 20
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}