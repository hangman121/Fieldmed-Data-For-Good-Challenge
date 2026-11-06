from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent / "D4GC data" / "D4GC data"
PARSED = CORPUS / "Parsed"
ORIGINAL = CORPUS / "Original"
META_CSV = ROOT.parent / "documentation.csv"

DATA = ROOT / "data"
DB_PATH = DATA / "index.sqlite"
EMB_PATH = DATA / "embeddings.npy"

MODELS = ROOT / "models"
EMBED_MODEL = MODELS / "e5-small"
RERANK_MODEL = MODELS / "minilm-reranker"
WHISPER_MODEL = MODELS / "whisper-base.en"  # Systran/faster-whisper-base.en

CHUNK_WORDS = 180

# Any OpenAI-compatible local endpoint: Ollama, llama-server, ...
LLM_URL = "http://localhost:11434/v1/chat/completions"
LLM_MODEL = "qwen3:0.6b"

# Central (online) server. A dummy hub runs locally for the demo: uvicorn app.hub:app --port 9000
HUB_URL = "http://localhost:9000"
HUB_DATA = DATA / "hub"
SITE_ID = "field-site-1"
SYNC_EVERY_S = 120

# Retrieval
CANDIDATES = 50      # per retriever, before fusion
RERANK_TOP = 20      # fused candidates sent to the cross-encoder
RERANK_THRESHOLD = -2.0  # cross-encoder logit; tuned on eval set
MAX_SOURCES = 6
LLM_SOURCES = 4      # passages given to the LLM (shorter prompt = faster on CPU)
