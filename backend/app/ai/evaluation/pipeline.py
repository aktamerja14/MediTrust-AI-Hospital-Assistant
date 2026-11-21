"""Offline RAG Evaluation Pipeline for MediTrust.

Executes approved evaluation questions against the existing RAGChain
and saves generated responses with their corresponding reference data.
"""
import asyncio
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from app.ai.evaluation.dataset_schema import (
    EvaluationCase,
    load_dataset,
    VALID_CATEGORIES,
)
from app.ai.evaluation.validation import (
    validate_dataset,
    ValidationReport,
)
from app.ai.rag.chain import RAGChain, RAGResponse, RAGChainError

logger = logging.getLogger("meditrust.ai.evaluation.pipeline")


@dataclass
class EvaluationResult:
    """Result of a single evaluation case execution.

    Contains the generated answer, retrieved context, and metadata
    for comparison with expected answers.
    """
    evaluation_id: str
    question: str
    expected_answer: str
    actual_answer: str
    reference_context: str
    retrieved_context: str
    category: str
    language: str
    tags: List[str]
    execution_status: str  # "success", "failed", "skipped", "unverified"
    error_message: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    has_context: bool = False
    model: str = ""
    doctor_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    disclaimer: str = ""
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class PipelineSummary:
    """Summary of pipeline execution."""
    total_cases: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    unverified: int = 0
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class OfflineEvaluationPipeline:
    """Offline evaluation pipeline for MediTrust RAG system.

    Loads evaluation cases, runs them through RAGChain, and collects results.
    Does not upload to LangSmith or any external service.
    """

    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        dataset_path: Optional[str] = None,
        include_unverified: bool = False,
        category_filter: Optional[List[str]] = None,
        case_id_filter: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ):
        """Initialize the pipeline.

        Args:
            rag_chain: RAGChain instance (creates new if not provided)
            dataset_path: Path to JSONL dataset file
            include_unverified: Whether to include cases needing human verification
            category_filter: List of categories to include (None = all)
            case_id_filter: List of case IDs to include (None = all)
            top_k: Number of chunks to retrieve (uses default if None)
        """
        self.rag_chain = rag_chain or RAGChain()
        self.dataset_path = dataset_path
        self.include_unverified = include_unverified
        self.category_filter = category_filter
        self.case_id_filter = case_id_filter
        self.top_k = top_k
        self._validation_report: Optional[ValidationReport] = None
        self._cases: List[EvaluationCase] = []

    def load_and_validate_dataset(self, dataset_path: Optional[str] = None) -> ValidationReport:
        """Load and validate the evaluation dataset.

        Args:
            dataset_path: Optional override for dataset path

        Returns:
            ValidationReport with validation results

        Raises:
            ValueError: If dataset file not found or invalid
        """
        path = dataset_path or self.dataset_path
        if not path:
            raise ValueError("Dataset path not provided")

        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Dataset file not found: {path}")

        logger.info("Loading and validating dataset from %s", path)
        self._validation_report = validate_dataset(path)

        # Load cases after validation
        self._cases = load_dataset(path)
        logger.info("Loaded %d cases from dataset", len(self._cases))

        return self._validation_report

    def _filter_cases(self, cases: List[EvaluationCase]) -> List[EvaluationCase]:
        """Apply filters to the cases.

        Args:
            cases: List of cases to filter

        Returns:
            Filtered list of cases
        """
        filtered = cases

        # Filter by category
        if self.category_filter:
            valid_categories = set(self.category_filter)
            invalid = valid_categories - VALID_CATEGORIES
            if invalid:
                logger.warning("Invalid categories in filter (ignored): %s", invalid)
            filtered = [c for c in filtered if c.category in valid_categories]
            logger.info("Filtered by category: %d cases remaining", len(filtered))

        # Filter by case ID
        if self.case_id_filter:
            id_set = set(self.case_id_filter)
            filtered = [c for c in filtered if c.id in id_set]
            logger.info("Filtered by case ID: %d cases remaining", len(filtered))

        # Filter by human verification flag
        if not self.include_unverified:
            original_count = len(filtered)
            filtered = [c for c in filtered if not c.needs_human_verification]
            skipped = original_count - len(filtered)
            if skipped > 0:
                logger.info("Excluded %d cases requiring human verification (use --include-unverified to include)", skipped)
        else:
            # Warn when explicitly including unverified cases
            unverified_cases = [c for c in filtered if c.needs_human_verification]
            if unverified_cases:
                warnings.warn(
                    f"Including {len(unverified_cases)} cases that require human verification. "
                    "Results for these cases should be reviewed by a medical professional.",
                    UserWarning,
                    stacklevel=2
                )
                logger.warning(
                    "INCLUDING %d CASES REQUIRING HUMAN VERIFICATION: %s",
                    len(unverified_cases),
                    [c.id for c in unverified_cases]
                )

        return filtered

    def _extract_retrieved_context(self, rag_response: RAGResponse) -> str:
        """Extract retrieved context from RAG response sources.

        Uses full_content when available (for evaluation accuracy),
        falls back to content_preview for compatibility.

        Args:
            rag_response: Response from RAGChain.generate()

        Returns:
            Concatenated retrieved context text
        """
        if not rag_response.sources:
            return ""

        context_parts = []
        for source in rag_response.sources:
            # Prefer full_content for evaluation; fallback to content_preview
            content = source.get("full_content") or source.get("content_preview", "")
            if content:
                context_parts.append(content)

        return "\n\n".join(context_parts)

    async def _run_single_case(self, case: EvaluationCase) -> EvaluationResult:
        """Run a single evaluation case through the RAG chain.

        Args:
            case: Evaluation case to run

        Returns:
            EvaluationResult with generated answer and metadata
        """
        start_time = datetime.utcnow()

        try:
            logger.debug("Running evaluation case: %s", case.id)

            # Generate answer using RAGChain
            rag_response = await self.rag_chain.generate(
                query=case.question,
                top_k=self.top_k,
            )

            # Extract retrieved context from sources
            retrieved_context = self._extract_retrieved_context(rag_response)

            # Determine execution status
            if not rag_response.answer or not rag_response.answer.strip():
                execution_status = "failed"
                error_message = "Empty response from RAG chain"
            else:
                execution_status = "success"
                error_message = None

            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return EvaluationResult(
                evaluation_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer=rag_response.answer,
                reference_context=case.reference_context,
                retrieved_context=retrieved_context,
                category=case.category,
                language=case.language,
                tags=case.tags,
                execution_status=execution_status,
                error_message=error_message,
                sources=rag_response.sources,
                has_context=rag_response.has_context,
                model=rag_response.model,
                doctor_recommendations=rag_response.doctor_recommendations or [],
                disclaimer=rag_response.disclaimer or "",
                execution_time_ms=execution_time_ms,
            )

        except RAGChainError as e:
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error("RAG chain error for case %s: %s", case.id, e)
            return EvaluationResult(
                evaluation_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer="",
                reference_context=case.reference_context,
                retrieved_context="",
                category=case.category,
                language=case.language,
                tags=case.tags,
                execution_status="failed",
                error_message=f"RAGChainError: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.exception("Unexpected error for case %s: %s", case.id, e)
            return EvaluationResult(
                evaluation_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer="",
                reference_context=case.reference_context,
                retrieved_context="",
                category=case.category,
                language=case.language,
                tags=case.tags,
                execution_status="failed",
                error_message=f"UnexpectedError: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

    async def run(
        self,
        dataset_path: Optional[str] = None,
        max_concurrent: int = 1,
    ) -> tuple[List[EvaluationResult], PipelineSummary]:
        """Run the evaluation pipeline.

        Args:
            dataset_path: Optional override for dataset path
            max_concurrent: Maximum concurrent evaluations (default 1 for stability)

        Returns:
            Tuple of (results list, pipeline summary)
        """
        # Load and validate dataset if not already done
        if not self._cases:
            self.load_and_validate_dataset(dataset_path)

        # Apply filters
        cases_to_run = self._filter_cases(self._cases)

        if not cases_to_run:
            logger.warning("No cases to evaluate after filtering")
            return [], PipelineSummary(total_cases=0)

        logger.info("Starting evaluation of %d cases", len(cases_to_run))

        # Run evaluations
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(case: EvaluationCase) -> EvaluationResult:
            async with semaphore:
                return await self._run_single_case(case)

        # Execute all cases
        tasks = [run_with_semaphore(case) for case in cases_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Build summary
        summary = PipelineSummary(
            total_cases=len(results),
            successful=sum(1 for r in results if r.execution_status == "success"),
            failed=sum(1 for r in results if r.execution_status == "failed"),
            skipped=sum(1 for r in results if r.execution_status == "skipped"),
            unverified=sum(1 for r in results if r.execution_status == "unverified"),
            execution_time_ms=sum(r.execution_time_ms for r in results),
        )

        logger.info(
            "Evaluation complete: %d successful, %d failed, %d skipped, %d unverified",
            summary.successful, summary.failed, summary.skipped, summary.unverified
        )

        return results, summary

    def save_results(
        self,
        results: List[EvaluationResult],
        summary: PipelineSummary,
        output_path: str,
    ) -> None:
        """Save evaluation results to JSON file.

        Args:
            results: List of evaluation results
            summary: Pipeline summary
            output_path: Path to output JSON file
        """
        output_data = {
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", output_path)


async def run_offline_evaluation(
    dataset_path: str,
    output_path: str,
    include_unverified: bool = False,
    category_filter: Optional[List[str]] = None,
    case_id_filter: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    max_concurrent: int = 1,
    rag_chain: Optional[RAGChain] = None,
) -> tuple[List[EvaluationResult], PipelineSummary]:
    """Convenience function to run offline evaluation.

    Args:
        dataset_path: Path to JSONL dataset file
        output_path: Path to output JSON file
        include_unverified: Include cases needing human verification
        category_filter: List of categories to include
        case_id_filter: List of case IDs to include
        top_k: Number of chunks to retrieve
        max_concurrent: Maximum concurrent evaluations
        rag_chain: Optional RAGChain instance

    Returns:
        Tuple of (results list, pipeline summary)
    """
    pipeline = OfflineEvaluationPipeline(
        rag_chain=rag_chain,
        dataset_path=dataset_path,
        include_unverified=include_unverified,
        category_filter=category_filter,
        case_id_filter=case_id_filter,
        top_k=top_k,
    )

    results, summary = await pipeline.run(max_concurrent=max_concurrent)
    pipeline.save_results(results, summary, output_path)

    return results, summary