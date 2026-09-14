import time
import logging

import httpx
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.vector_store import get_retriever, build_query_text

logger = logging.getLogger(__name__)

# Krishi Agent prompt — replaces AiVideoAssistant's meeting-transcript prompt.
# Key difference: this one is instructed to answer ONLY from ICAR/KVK context (proposal's
# core promise — "never invented by a model") and to name which source doc it drew from,
# so the Flutter chat UI can render the "ICAR सलाह" / "KVK सलाह" badge from the mockup.
SYSTEM_PROMPT = """Aap ek krishi visheshagya (agriculture expert) hain jo Bharatiya kisano ko
salaah dete hain. Neeche diye gaye ICAR/KVK advisory documents ke context ke AADHAR PAR HI
jawab dein — apni taraf se kuch bhi na jodein ya andaza na lagayen.

Agar context mein iska jawab nahi mil raha, to seedha bol dein:
"Mujhe iski jaankari ICAR/KVK advisory documents mein nahi mili. Kripya apne nazdeeki
Krishi Vigyan Kendra se sampark karein."

Jawab hamesha:
- Hindi mein, seedha aur kisan ko samajh aane wali bhasha mein
- Chhota aur practical (2-4 vaakya), lambi lecture na dein
- Agar koi specific quantity/dose ka zikr hai (jaise khaad ki matra), use exactly context se lein

Context (ICAR/KVK advisory documents se):
{context}"""


def get_llm(provider: str):
    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY .env mein set nahi hai.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
        )
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY .env mein set nahi hai.")
    return ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.2,
    )


def _get_retry_after_seconds(exc: httpx.HTTPStatusError):
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None


def _invoke_with_retry(llm, messages, max_retries):
    """Same 429-backoff pattern as AiVideoAssistant's _try_invoke, kept intact."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code != 429:
                raise
            retry_after = _get_retry_after_seconds(e)
            wait_time = retry_after if retry_after is not None else min(2 ** attempt, settings.FALLBACK_MAX_WAIT)
            logger.warning(f"429 rate limited. Waiting {wait_time:.0f}s (attempt {attempt}/{max_retries})...")
            if attempt < max_retries:
                time.sleep(wait_time)
    raise RuntimeError("rate limited") from last_exc


def format_docs_and_sources(docs):
    """
    Returns (context_text, sources) where sources is a de-duplicated list of the
    "source" filename metadata tag set during ingestion. This is what powers the
    "ICAR सलाह" / "KVK सलाह" citation badge in the Flutter mockup.
    """
    context_text = "\n\n".join(doc.page_content for doc in docs)
    seen = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.append(src)
    return context_text, seen


def answer_query(vector_store, query: str, crop: str = None, district: str = None) -> dict:
    """
    Main entry point called by the router. Returns:
        {"answer": str, "sources": list[str], "provider_used": str}

    Fallback logic (groq -> gemini) kept from AiVideoAssistant's mistral->gemini pattern,
    just repointed at Groq and moved from a "print + return string" shape to a
    structured dict the API can serialize.
    """
    retriever = get_retriever(vector_store, k=settings.RETRIEVER_TOP_K)
    query_text = build_query_text(query, crop=crop, district=district)

    docs = retriever.invoke(query_text)
    context, sources = format_docs_and_sources(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    messages = prompt.format_messages(context=context, question=query)

    provider_used = "groq"
    try:
        llm = get_llm("groq")
        response = _invoke_with_retry(llm, messages, settings.MAX_LLM_RETRIES)
    except (httpx.HTTPStatusError, RuntimeError, ValueError) as e:
        logger.warning(f"Groq failed ({e}), falling back to Gemini...")
        provider_used = "gemini"
        llm = get_llm("gemini")
        response = _invoke_with_retry(llm, messages, settings.MAX_LLM_RETRIES)

    return {
        "answer": response.content,
        "sources": sources,
        "provider_used": provider_used,
    }
