"""
LLM generation -- provider-agnostic.

Switch providers by setting LLM_PROVIDER in .env to one of:
  "anthropic" | "openai" | "gemini" | "groq"

Only the SDK for your chosen provider needs to be installed.
"""
from app.config import LLM_PROVIDER, MAX_TOKENS, get_default_model


def generate(prompt: str, model: str = None) -> str:
    """
    Args:
        model: provider-specific model name. If None, uses the default
               for whichever provider is configured (see config.py).
    """
    model = model or get_default_model()

    if LLM_PROVIDER == "anthropic":
        return _generate_anthropic(prompt, model)
    elif LLM_PROVIDER == "openai":
        return _generate_openai(prompt, model)
    elif LLM_PROVIDER == "gemini":
        return _generate_gemini(prompt, model)
    elif LLM_PROVIDER == "groq":
        return _generate_groq(prompt, model)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Set LLM_PROVIDER in .env to 'anthropic', 'openai', 'gemini', or 'groq'."
        )


def _generate_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    from app.config import ANTHROPIC_API_KEY
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _generate_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    from app.config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _generate_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types
    from app.config import GEMINI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS),
    )
    return response.text


def _generate_groq(prompt: str, model: str) -> str:
    from groq import Groq
    from app.config import GROQ_API_KEY
    client = Groq(api_key=GROQ_API_KEY, max_retries=10)
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
