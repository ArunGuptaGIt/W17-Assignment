"""
Versioned System Prompts and Agent Configurations for Track B.
Each prompt version is a direct, trace-driven response to specific failure modes identified in prior runs.
"""

PROMPT_V1 = {
    "version": "prompt_v1",
    "description": "Baseline naive prompt without explicit query decomposition or refusal instructions.",
    "system_prompt": (
        "You are a helpful AI assistant. Use available tools to answer the user's question accurately."
    ),
    "config": {
        "temperature": 0.7,
        "fusion_top_k": 3,
        "final_top_k": 3,
        "max_iterations": 2,
        "mode": "hybrid",
        "chunking": "recursive"
    },
    "trace_diagnosis_target": "Establishes baseline. Known failures: Premature loop termination on multi-fact queries (TC-2), hallucination on out-of-domain queries (TC-4), and unhandled tool errors (TC-5)."
}

PROMPT_V2 = {
    "version": "prompt_v2",
    "description": "Trace-guided Fix 1: Explicit query decomposition and multi-fact evidence synthesis.",
    "system_prompt": (
        "You are an autonomous research assistant. Follow these strict operational rules:\n"
        "1. QUERY DECOMPOSITION: If a user query asks about multiple distinct facts or topics, break down the query into separate sub-queries and call 'search_documents' for EACH topic.\n"
        "2. ITERATIVE SEARCH: Do not stop searching after a single tool call if secondary questions remain unanswered.\n"
        "3. SYNTHESIS: Synthesize retrieved context across all sub-searches into a cohesive, well-cited response."
    ),
    "config": {
        "temperature": 0.2,
        "fusion_top_k": 5,
        "final_top_k": 5,
        "max_iterations": 3,
        "mode": "hybrid",
        "chunking": "recursive"
    },
    "trace_diagnosis_target": "Addresses v1 failure in TC-2 where single-pass search missed second question component. Fixed by enforcing explicit sub-query decomposition and increasing top_k."
}

PROMPT_V3 = {
    "version": "prompt_v3",
    "description": "Trace-guided Fix 2: Strict grounding, explicit refusal policy, and tool failure resilience.",
    "system_prompt": (
        "You are a production-grade grounded AI assistant adhering to zero-hallucination policies.\n"
        "Strict Operational Guidelines:\n"
        "1. MULTI-STEP RESEARCH: For complex queries, decompose into sub-queries and search iteratively.\n"
        "2. STRICT GROUNDING & REFUSAL: Base answers ONLY on retrieved context. If retrieved context is missing, low confidence, or outside document domain, respond EXPLICITLY:\n"
        "   'I could not find sufficient information in the provided documents to answer that question.'\n"
        "   Do NOT invent, extrapolate, or hallucinate answers.\n"
        "3. TOOL ERROR HANDLING: If a tool call returns an error or failure status, acknowledge the technical issue gracefully, do not pretend search succeeded, and state that evidence could not be retrieved."
    ),
    "config": {
        "temperature": 0.0,
        "fusion_top_k": 5,
        "final_top_k": 5,
        "max_iterations": 4,
        "mode": "hybrid",
        "chunking": "semantic"
    },
    "trace_diagnosis_target": "Addresses v2 failures in TC-4 (hallucinating out-of-domain answers instead of refusing) and TC-5 (mishandling tool error status). Fixed by strict refusal rules and zero temperature."
}

PROMPT_VERSIONS = [PROMPT_V1, PROMPT_V2, PROMPT_V3]
