from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_API_KEY: str = ""

    CHROMA_DB_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "krishi_agent_docs"

    DOCS_SOURCE_DIR: str = "./data/icar_kvk_docs"

    # Multilingual because farmers query in Hindi, but ICAR/KVK source PDFs are mostly
    # English — an English-only embedding model (e.g. all-MiniLM-L6-v2) would retrieve
    # poorly across that language gap. If your machine is low on RAM, swap to
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (~470MB vs ~1GB).
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    RETRIEVER_TOP_K: int = 4
    MAX_LLM_RETRIES: int = 2
    FALLBACK_MAX_WAIT: int = 30


settings = Settings()
