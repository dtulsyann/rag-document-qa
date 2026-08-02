"""
Phase 4: Prompt construction + grounding instructions.

"Grounding" = forcing the LLM to answer only from retrieved context,
instead of falling back on parametric knowledge or hallucinating when
context is thin. This is what makes it RAG rather than "search then
ask a chatbot".

Chunks are numbered as [Source N] in the context, and the LLM is asked
to tag claims with the source number it used. citations.py later maps
those numbers back to real filename/page metadata.
"""

GROUNDING_INSTRUCTIONS = """You are a document question-answering assistant.
Answer the question using ONLY the context provided below. Do not use any
outside knowledge, and do not speculate or fill gaps with assumptions.

When you make a claim, cite which source it came from using the format [Source N].

If the context does not contain enough information to answer the question,
respond exactly with: "The document does not contain enough information to answer this question."

Format your answer using proper Markdown. If listing items, you MUST use bullet points starting with "-" and put each item on its own new line.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Source {i + 1}: {chunk['filename']}, page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""{GROUNDING_INSTRUCTIONS}

Context:
{context_str}

Question: {question}

Answer:"""
