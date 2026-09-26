"""
Golden Reference Dataset for Track B Agentic Regression Testing.
Contains representative evaluation test cases paired with approved golden reference answers.
"""

GOLDEN_TEST_CASES = [
    {
        "id": "TC-1",
        "query": "How many business days of paid annual leave are employees entitled to per calendar year?",
        "category": "Simple Factual",
        "golden_answer": "Full-time employees are entitled to 20 business days of paid annual leave per calendar year, accrued monthly on a pro-rata basis.",
        "expected_refusal": False,
        "description": "Direct single-fact query with clear answer in policy doc."
    },
    {
        "id": "TC-2",
        "query": "What is the annual equipment reimbursement stipend for home office, and how does hybrid retrieval fuse candidate documents?",
        "category": "Multi-Fact / Complex",
        "golden_answer": "The annual home office equipment reimbursement stipend is $500 per calendar year. Hybrid retrieval fuses candidate documents from dense vector search (ChromaDB) and sparse BM25 keyword search using Reciprocal Rank Fusion (RRF) to score and combine candidate lists.",
        "expected_refusal": False,
        "description": "Multi-fact query requiring cross-document evidence synthesis."
    },
    {
        "id": "TC-3",
        "query": "What core architectural principles does the RAG system follow regarding chunking and retrieval?",
        "category": "Architecture Factual",
        "golden_answer": "The RAG system follows modular architecture principles with persistent dual indexing (ChromaDB vector store + BM25 keyword store), support for both Recursive Character and Semantic chunking, cross-encoder reranking, and model-driven agentic tool execution.",
        "expected_refusal": False,
        "description": "Architecture query with specific structural keywords."
    },
    {
        "id": "TC-4",
        "query": "What is the company policy regarding interdimensional space travel expenses?",
        "category": "Out-of-Domain / Insufficient Evidence",
        "golden_answer": "I could not find sufficient information in the provided documents to answer that question.",
        "expected_refusal": True,
        "description": "Query outside document corpus expecting clear refusal/insufficient evidence report."
    },
    {
        "id": "TC-5",
        "query": "How is the local LLM served and tuned for the 4GB RTX 3050 GPU constraint?",
        "category": "Injected Failure Test",
        "golden_answer": "I could not find sufficient information in the provided documents due to a tool retrieval error.",
        "expected_refusal": True,
        "description": "Failure Injection: search_documents returns empty/errored payload to test agent resilience."
    }
]
