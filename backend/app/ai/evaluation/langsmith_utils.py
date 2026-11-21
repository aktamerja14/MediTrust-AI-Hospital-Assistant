"""LangSmith dataset upload utilities for MediTrust evaluation.

Provides functions to upload evaluation datasets to LangSmith.
Does NOT auto-upload - requires explicit invocation.
"""
import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from langsmith import Client
from langsmith.schemas import Example

from app.ai.evaluation.dataset_schema import (
    EvaluationCase,
    load_dataset,
    validate_case,
    get_dataset_stats,
)

logger = logging.getLogger("meditrust.ai.evaluation.langsmith_utils")


# Default dataset name in LangSmith
DEFAULT_DATASET_NAME = "meditrust-rag-evaluation-v1"


class LangSmithUploadError(Exception):
    """Exception for LangSmith upload errors."""
    pass


def create_langsmith_client(api_key: Optional[str] = None) -> Client:
    """Create a LangSmith client.

    Args:
        api_key: Optional API key. If not provided, uses LANGSMITH_API_KEY env var.

    Returns:
        Configured LangSmith Client

    Raises:
        LangSmithUploadError: If API key is not available
    """
    key = api_key or os.getenv("LANGSMITH_API_KEY")
    if not key:
        raise LangSmithUploadError(
            "LangSmith API key not provided. Set LANGSMITH_API_KEY environment variable "
            "or pass api_key parameter."
        )
    return Client(api_key=key)


def dataset_exists(client: Client, dataset_name: str) -> bool:
    """Check if a dataset exists in LangSmith.

    Args:
        client: LangSmith client
        dataset_name: Name of the dataset

    Returns:
        True if dataset exists, False otherwise
    """
    try:
        client.read_dataset(dataset_name=dataset_name)
        return True
    except Exception:
        return False


def create_dataset(
    client: Client,
    dataset_name: str,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """Create a new dataset in LangSmith.

    Args:
        client: LangSmith client
        dataset_name: Name for the new dataset
        description: Description of the dataset
        metadata: Optional metadata dict

    Returns:
        Created dataset object
    """
    return client.create_dataset(
        dataset_name=dataset_name,
        description=description or f"MediTrust RAG Evaluation Dataset - {dataset_name}",
        metadata=metadata or {"project": "meditrust", "type": "rag_evaluation"},
    )


def upload_evaluation_cases(
    client: Client,
    dataset_name: str,
    cases: List[EvaluationCase],
    batch_size: int = 50
) -> Dict[str, Any]:
    """Upload evaluation cases to LangSmith dataset.

    Converts EvaluationCase objects to LangSmith Examples with:
    - inputs: {"question": ..., "category": ..., "language": ..., "tags": ...}
    - outputs: {"expected_answer": ..., "reference_context": ..., "needs_human_verification": ...}

    Args:
        client: LangSmith client
        dataset_name: Target dataset name
        cases: List of EvaluationCase objects
        batch_size: Number of examples per batch

    Returns:
        Dict with upload statistics
    """
    if not cases:
        return {"uploaded": 0, "skipped": 0, "errors": []}

    # Validate all cases first
    errors = []
    valid_cases = []
    for case in cases:
        validation_errors = validate_case(case)
        if validation_errors:
            errors.append({"case_id": case.id, "errors": validation_errors})
        else:
            valid_cases.append(case)

    if errors:
        logger.warning("Skipping %d invalid cases", len(errors))

    # Convert to LangSmith Examples
    examples = []
    for case in valid_cases:
        example = Example(
            id=uuid.uuid4(),
            inputs={
                "question": case.question,
                "category": case.category,
                "language": case.language,
                "tags": case.tags,
            },
            outputs={
                "expected_answer": case.expected_answer,
                "reference_context": case.reference_context,
                "needs_human_verification": case.needs_human_verification,
                "metadata": case.metadata,
            },
            metadata={
                "case_id": case.id,
                "created_at": case.created_at,
                **case.metadata,
            },
        )
        examples.append(example)

    # Upload in batches
    uploaded = 0
    upload_errors = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        try:
            client.create_examples(dataset_name=dataset_name, examples=batch)
            uploaded += len(batch)
            logger.info("Uploaded batch %d-%d (%d examples)", i, i + len(batch), len(batch))
        except Exception as e:
            logger.exception("Failed to upload batch starting at %d: %s", i, e)
            upload_errors.append({"batch_start": i, "error": str(e)})

    return {
        "uploaded": uploaded,
        "skipped": len(errors),
        "validation_errors": errors,
        "upload_errors": upload_errors,
        "total_cases": len(cases),
    }


def upload_dataset_from_file(
    filepath: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    api_key: Optional[str] = None,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    create_if_missing: bool = True,
) -> Dict[str, Any]:
    """Complete workflow: load dataset from JSONL and upload to LangSmith.

    This is the main entry point for uploading evaluation datasets.
    Does NOT run automatically - must be explicitly called.

    Args:
        filepath: Path to JSONL dataset file
        dataset_name: Target LangSmith dataset name
        api_key: Optional LangSmith API key
        description: Dataset description
        metadata: Additional dataset metadata
        create_if_missing: Create dataset if it doesn't exist

    Returns:
        Dict with upload results and statistics

    Raises:
        LangSmithUploadError: If upload fails
    """
    # Load and validate dataset
    logger.info("Loading dataset from %s", filepath)
    cases = load_dataset(filepath)
    stats = get_dataset_stats(cases)
    logger.info("Loaded %d cases: %s", stats["total"], stats["categories"])

    # Create client
    client = create_langsmith_client(api_key)

    # Ensure dataset exists
    if not dataset_exists(client, dataset_name):
        if create_if_missing:
            logger.info("Creating dataset '%s'", dataset_name)
            create_dataset(client, dataset_name, description, metadata)
        else:
            raise LangSmithUploadError(
                f"Dataset '{dataset_name}' does not exist. "
                "Set create_if_missing=True to create it."
            )
    else:
        logger.info("Dataset '%s' already exists", dataset_name)

    # Upload cases
    result = upload_evaluation_cases(client, dataset_name, cases)
    result["dataset_stats"] = stats

    logger.info("Upload complete: %d uploaded, %d skipped", result["uploaded"], result["skipped"])
    return result


def list_datasets(client: Optional[Client] = None, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all datasets in the LangSmith workspace.

    Args:
        client: Optional existing client
        api_key: Optional API key if client not provided

    Returns:
        List of dataset info dicts
    """
    if client is None:
        client = create_langsmith_client(api_key)

    datasets = []
    for ds in client.list_datasets():
        datasets.append({
            "name": ds.dataset_name,
            "id": str(ds.id),
            "description": ds.description,
            "created_at": ds.created_at.isoformat() if ds.created_at else None,
            "example_count": ds.example_count,
        })
    return datasets


def delete_dataset(dataset_name: str, api_key: Optional[str] = None) -> bool:
    """Delete a dataset from LangSmith (use with caution).

    Args:
        dataset_name: Name of dataset to delete
        api_key: Optional API key

    Returns:
        True if deleted, False if not found
    """
    client = create_langsmith_client(api_key)
    try:
        client.delete_dataset(dataset_name=dataset_name)
        logger.info("Deleted dataset '%s'", dataset_name)
        return True
    except Exception as e:
        logger.warning("Failed to delete dataset '%s': %s", dataset_name, e)
        return False


def get_dataset_examples(
    dataset_name: str,
    api_key: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch examples from a LangSmith dataset.

    Args:
        dataset_name: Dataset name
        api_key: Optional API key
        limit: Maximum number of examples to fetch

    Returns:
        List of example dicts
    """
    client = create_langsmith_client(api_key)
    examples = []

    for ex in client.list_examples(dataset_name=dataset_name, limit=limit):
        examples.append({
            "id": str(ex.id),
            "inputs": ex.inputs,
            "outputs": ex.outputs,
            "metadata": ex.metadata,
            "created_at": ex.created_at.isoformat() if ex.created_at else None,
        })

    return examples