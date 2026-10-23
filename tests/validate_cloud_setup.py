"""
Live Cloud Infrastructure Validation Script for Learn RAG.
Verifies connectivity to Qdrant Cloud and OpenRouter API endpoints.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import get_config, validate_config
from rag.ingestion import get_qdrant_client, ensure_collection_exists
from rag.retrieval import get_embedding_model
from rag.generation import get_llm_client
from langchain_core.messages import HumanMessage


def run_cloud_validation() -> bool:
    print("=" * 60)
    print("  Learn RAG: Live Cloud Infrastructure Validation")
    print("=" * 60)
    
    config = get_config()
    is_valid, issues = validate_config(config)
    
    if not is_valid:
        print("\n[!] Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease update your '.env' file with valid cloud credentials.")
        return False
        
    print("\n[OK] Environment variables and configuration syntax are valid.")
    print(f"  * Qdrant Cloud URL: {config.qdrant_url}")
    print(f"  * Qdrant Collection: {config.qdrant_collection_name}")
    print(f"  * OpenRouter Model: {config.openrouter_model}")

    # 1. Test Qdrant Cloud Connectivity
    print("\n[1/3] Testing Qdrant Cloud connection...")
    try:
        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"[OK] Successfully reached Qdrant Cloud! Found {len(collection_names)} collection(s).")
        
        ensure_collection_exists(client, config.qdrant_collection_name, vector_dim=384)
        print(f"[OK] Collection '{config.qdrant_collection_name}' is verified ready.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant Cloud: {e}")
        return False

    # 2. Test Local Embeddings Model
    print("\n[2/3] Initializing local embedding model...")
    try:
        embeddings = get_embedding_model(config.embedding_model_name)
        vector = embeddings.embed_query("Hello RAG test")
        print(f"[OK] Embeddings loaded successfully! Generated vector with {len(vector)} dimensions.")
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        return False

    # 3. Test OpenRouter Connection
    print("\n[3/3] Testing OpenRouter API connection...")
    try:
        llm = get_llm_client(
            openrouter_api_key=config.openrouter_api_key,
            model_name=config.openrouter_model,
            temperature=0.2,
        )
        response = llm.invoke([HumanMessage(content="Respond with the single word: READY")])
        reply = response.content.strip()
        print(f"[OK] OpenRouter answered: '{reply}'")
    except Exception as e:
        print(f"[ERROR] Failed to connect to OpenRouter API: {e}")
        return False

    print("\n" + "=" * 60)
    print("[SUCCESS] All cloud infrastructure checks PASSED! You are ready to run:")
    print("   streamlit run app.py")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_cloud_validation()
    sys.exit(0 if success else 1)
