"""Output writer module for serializing lead enrichment results to JSON."""

import json
import logging
import os
from typing import List
from src.models import DomainResult

logger = logging.getLogger(__name__)


def write_json_output(
    results: List[DomainResult], output_path: str = "output/output.json"
) -> str:
    """
    Serialize domain enrichment results to output/output.json.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        serialized_data = [res.model_dump(mode="json") for res in results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully wrote {len(results)} domain results to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error writing JSON output to {output_path}: {e}")
        raise RuntimeError(f"Failed to write output JSON: {e}") from e
