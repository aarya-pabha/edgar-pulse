"""
Export nodes from Pinecone vector store to local JSON files for BM25 hybrid retrieval (P2.1).
Pulls all nodes, parses them with LlamaIndex metadata_dict_to_node, and saves to data/nodes_{TICKER}.json.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from rag_guardrails import load_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_nodes")

def export_all_nodes(output_dir: str = "data"):
    # Load .env if present
    load_env()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")

    pc = Pinecone(api_key=api_key)
    index_name = "financial-rag-agent"
    idx = pc.Index(index_name)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Listing all vector IDs in index '%s'...", index_name)
    all_ids = []
    for id_batch in idx.list():
        all_ids.extend(id_batch)
    logger.info("Found %d total vectors.", len(all_ids))

    nodes_by_ticker = {}
    batch_size = 100

    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        logger.info("Fetching batch %d - %d of %d...", i + 1, min(i + batch_size, len(all_ids)), len(all_ids))
        fetched = idx.fetch(ids=batch)
        for vid, vdata in fetched.vectors.items():
            meta = vdata.metadata
            ticker = meta.get("ticker", "UNKNOWN").upper()
            if ticker not in nodes_by_ticker:
                nodes_by_ticker[ticker] = []

            # Store the node dict representation so it can be reloaded easily
            try:
                node = metadata_dict_to_node(meta)
                node_dict = node.to_dict()
                nodes_by_ticker[ticker].append(node_dict)
            except Exception as e:
                logger.warning("Could not convert vector %s to node: %s", vid, e)

    for ticker, nodes in nodes_by_ticker.items():
        file_path = out_path / f"nodes_{ticker}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d nodes for ticker %s to %s", len(nodes), ticker, file_path)

if __name__ == "__main__":
    export_all_nodes()
