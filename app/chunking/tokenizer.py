"""
Shared token counting for chunking strategies.
Uses tiktoken for consistent, model-agnostic token counts (good enough
proxy even though we're not using an OpenAI model for generation).

Falls back to a simple whitespace-based "tokenizer" if tiktoken's BPE
file can't be downloaded (e.g. no internet / blocked network). On a
normal machine with internet access, tiktoken downloads its encoding
file once and caches it locally -- this fallback only matters in
restricted/offline environments.
"""
try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    _encoder = None
    _USE_TIKTOKEN = False


def count_tokens(text: str) -> int:
    return len(encode(text))


def encode(text: str):
    if _USE_TIKTOKEN:
        return _encoder.encode(text)
    return text.split(" ")  # fallback: whitespace-separated "tokens"


def decode(tokens) -> str:
    if _USE_TIKTOKEN:
        return _encoder.decode(tokens)
    return " ".join(tokens)
