"""LangSmith Experiment Runner for MediTrust RAG Evaluation.

Runs evaluation experiments using the MediTrust RAG pipeline with existing evaluators.
Integrates programmatic evaluators and LLM judges with LangSmith experiments.
"""
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Awaitable

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example

from app.ai.evaluation.dataset_schema import (
    EvaluationCase,
    load_dataset,
    validate_case,
    get_dataset_stats,
)
from app.ai.evaluation.pipeline import (
    EvaluationResult as PipelineEvaluationResult,
    PipelineSummary,
    OfflineEvaluationPipeline,
    run_offline_evaluation,
)
from app.ai.evaluation.programmatic_evaluators import (
    EvaluatorRunner,
    run_programmatic_evaluation,
    PROGRAMMATIC_EVALUATORS,
    DEFAULT_EVALUATORS as PROGRAMMATIC_DEFAULT_EVALUATORS,
    EvaluationStatus,
    EvaluatorResult,
)
from app.ai.evaluation.llm_judges import (
    LLMJudgeRunner,
    run_llm_evaluation,
    LLM_JUDGES,
    LLM_DEFAULT_EVALUATORS,
    LLMJudgeResult,
    JudgeStatus,
)
from app.ai.evaluation.langsmith_utils import (
    create_langsmith_client,
    upload_dataset_from_file,
    DEFAULT_DATASET_NAME,
    LangSmithUploadError,
    list_datasets,
)

logger = logging.getLogger("meditrust.ai.evaluation.langsmith_experiment")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a LangSmith experiment run."""
    experiment_name: str
    dataset_name: str = "meditrust-rag-evaluation-v1"
    evaluator_names: Optional[List[str]] = None
    include_programmatic: bool = True
    include_llm_judges: bool = True
    max_concurrent: int = 3
    max_examples: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Result of a single example evaluation in an experiment."""
    example_id: str
    question: str
    expected_answer: str
    actual_answer: str
    reference_context: str
    retrieved_context: str
    category: str
    execution_status: str
    evaluator_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSummary:
    """Summary of experiment execution."""
    experiment_name: str
    dataset_name: str
    total_examples: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    evaluator_summary: Dict[str, Any] = field(default_factory=dict)
    langsmith_experiment_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# LangSmith Experiment Runner
# =============================================================================

class LangSmithExperimentRunner:
    """Runs LangSmith evaluation experiments using MediTrust RAG pipeline."""

    def __init__(
        self,
        config: ExperimentConfig,
        rag_chain: Optional["RAGChain"] = None,
        client: Optional[Client] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration
            rag_chain: Optional RAGChain instance (creates new if not provided)
            client: Optional LangSmith client
            api_key: Optional LangSmith API key
        """
        self.config = config
        self.rag_chain = rag_chain
        self.client = client or create_langsmith_client(api_key)
        self._pipeline: Optional[OfflineEvaluationPipeline] = None
        self._programmatic_runner: Optional[EvaluatorRunner] = None
        self._llm_judge_runner: Optional[LLMJudgeRunner] = None

    def _get_pipeline(self) -> OfflineEvaluationPipeline:
        """Get or create the offline evaluation pipeline."""
        if self._pipeline is None:
            self._pipeline = OfflineEvaluationPipeline(
                rag_chain=self.rag_chain,
            )
        return self._pipeline

    def _get_programmatic_runner(self) -> EvaluatorRunner:
        """Get or create the programmatic evaluator runner."""
        if self._programmatic_runner is None:
            evaluator_names = None
            if self.config.evaluator_names:
                evaluator_names = [
                    name for name in self.config.evaluator_names
                    if name in PROGRAMMATIC_EVALUATORS
                ]
            self._programmatic_runner = EvaluatorRunner(
                evaluator_names=evaluator_names,
            )
        return self._programmatic_runner

    def _get_llm_judge_runner(self) -> LLMJudgeRunner:
        """Get or create the LLM judge runner."""
        if self._llm_judge_runner is None:
            evaluator_names = None
            if self.config.evaluator_names:
                evaluator_names = [
                    name for name in self.config.evaluator_names
                    if name in LLM_JUDGES
                ]
            self._llm_judge_runner = LLMJudgeRunner(
                evaluator_names=evaluator_names,
                max_concurrent=self.config.max_concurrent,
            )
        return self._llm_judge_runner

    def _prepare_example_for_evaluation(self, example: Example) -> Dict[str, Any]:
        """Convert LangSmith Example to pipeline evaluation input."""
        return {
            "id": str(example.id),
            "question": example.inputs.get("question", ""),
            "expected_answer": example.outputs.get("expected_answer", ""),
            "reference_context": example.outputs.get("reference_context", ""),
            "category": example.inputs.get("category", "general_info"),
            "language": example.inputs.get("language", "en"),
            "tags": example.inputs.get("tags", []),
            "needs_human_verification": example.outputs.get("needs_human_verification", False),
        }

    async def _run_rag_generation(self, example: Example) -> Dict[str, Any]:
        """Run RAG generation for a single example."""
        pipeline = self._get_pipeline()

        # Create a pipeline evaluation case from the example
        example_data = self._prepare_example_for_evaluation(example)

        from app.ai.evaluation.dataset_schema import EvaluationCase
        case = EvaluationCase(
            id=example_data["id"],
            question=example_data["question"],
            expected_answer=example_data["expected_answer"],
            reference_context=example_data["reference_context"],
            category=example_data["category"],
            language=example_data["language"],
            tags=example_data["tags"],
            needs_human_verification=example_data["needs_human_verification"],
        )

        # Run the case through the pipeline
        result = await pipeline._run_single_case(case)

        return {
            "actual_answer": result.actual_answer,
            "retrieved_context": result.retrieved_context,
            "execution_status": result.execution_status,
            "error_message": result.error_message,
            "sources": result.sources,
            "has_context": result.has_context,
            "model": result.model,
            "doctor_recommendations": result.doctor_recommendations,
            "disclaimer": result.disclaimer,
            "execution_time_ms": result.execution_time_ms,
        }

    def _run_programmatic_evaluators(self, pipeline_result: Dict[str, Any], example: Example) -> Dict[str, Dict[str, Any]]:
        """Run programmatic evaluators on the pipeline result."""
        runner = self._get_programmatic_runner()

        # Create a pipeline evaluation result for the evaluators
        from app.ai.evaluation.pipeline import EvaluationResult as PipelineEvaluationResult

        pipeline_eval_result = PipelineEvaluationResult(
            evaluation_id=str(example.id),
            question=example.inputs.get("question", ""),
            expected_answer=example.outputs.get("expected_answer", ""),
            actual_answer=pipeline_result["actual_answer"],
            reference_context=example.outputs.get("reference_context", ""),
            retrieved_context=pipeline_result["retrieved_context"],
            category=example.inputs.get("category", "general_info"),
            language=example.inputs.get("language", "en"),
            tags=example.inputs.get("tags", []),
            execution_status=pipeline_result["execution_status"],
            error_message=pipeline_result.get("error_message"),
            sources=pipeline_result.get("sources", []),
            has_context=pipeline_result.get("has_context", False),
            model=pipeline_result.get("model", ""),
            doctor_recommendations=pipeline_result.get("doctor_recommendations", []),
            disclaimer=pipeline_result.get("disclaimer", ""),
            execution_time_ms=pipeline_result.get("execution_time_ms", 0.0),
        )

        evaluator_results = runner.run_evaluators(pipeline_eval_result)
        return {name: result.to_dict() for name, result in evaluator_results.items()}

    async def _run_llm_judges(self, pipeline_result: Dict[str, Any], example: Example) -> Dict[str, Dict[str, Any]]:
        """Run LLM judges on the pipeline result."""
        judge_runner = self._get_llm_judge_runner()

        from app.ai.evaluation.pipeline import EvaluationResult as PipelineEvaluationResult

        pipeline_eval_result = PipelineEvaluationResult(
            evaluation_id=str(example.id),
            question=example.inputs.get("question", ""),
            expected_answer=example.outputs.get("expected_answer", ""),
            actual_answer=pipeline_result["actual_answer"],
            reference_context=example.outputs.get("reference_context", ""),
            retrieved_context=pipeline_result["retrieved_context"],
            category=example.inputs.get("category", "general_info"),
            language=example.inputs.get("language", "en"),
            tags=example.inputs.get("tags", []),
            execution_status=pipeline_result["execution_status"],
            error_message=pipeline_result.get("error_message"),
            sources=pipeline_result.get("sources", []),
            has_context=pipeline_result.get("has_context", False),
            model=pipeline_result.get("model", ""),
            doctor_recommendations=pipeline_result.get("doctor_recommendations", []),
            disclaimer=pipeline_result.get("disclaimer", ""),
            execution_time_ms=pipeline_result.get("execution_time_ms", 0.0),
        )

        judge_results = await judge_runner.run_judges(pipeline_eval_result)
        return {name: result.to_dict() for name, result in judge_results.items()}

    async def _evaluate_single_example(self, example: Example) -> ExperimentResult:
        """Evaluate a single example through the full pipeline."""
        start_time = datetime.utcnow()

        try:
            # Run RAG generation
            pipeline_result = await self._run_rag_generation(example)

            # Run evaluators
            evaluator_results = {}

            # Run programmatic evaluators
            if self.config.include_programmatic:
                programmatic_results = self._run_programmatic_evaluators(pipeline_result, example)
                evaluator_results.update(programmatic_results)

            # Run LLM judges
            if self.config.include_llm_judges:
                llm_results = await self._run_llm_judges(pipeline_result, example)
                evaluator_results.update(llm_results)

            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ExperimentResult(
                example_id=str(example.id),
                question=example.inputs.get("question", ""),
                expected_answer=example.outputs.get("expected_answer", ""),
                actual_answer=pipeline_result["actual_answer"],
                reference_context=example.outputs.get("reference_context", ""),
                retrieved_context=pipeline_result["retrieved_context"],
                category=example.inputs.get("category", "general_info"),
                execution_status=pipeline_result["execution_status"],
                evaluator_results=evaluator_results,
                execution_time_ms=execution_time_ms,
                error_message=pipeline_result.get("error_message"),
            )

        except Exception as e:
            logger.exception("Error evaluating example %s: %s", example.id, e)
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ExperimentResult(
                example_id=str(example.id),
                question=example.inputs.get("question", ""),
                expected_answer=example.outputs.get("expected_answer", ""),
                actual_answer="",
                reference_context=example.outputs.get("reference_context", ""),
                retrieved_context="",
                category=example.inputs.get("category", "general_info"),
                execution_status="error",
                evaluator_results={},
                execution_time_ms=execution_time_ms,
                error_message=str(e),
            )

    async def run_experiment(self) -> tuple[List[ExperimentResult], ExperimentSummary]:
        """Run the full experiment on the LangSmith dataset."""
        logger.info("Starting experiment '%s' on dataset '%s'",
                   self.config.experiment_name, self.config.dataset_name)

        # Fetch examples from LangSmith dataset
        examples = list(self.client.list_examples(
            dataset_name=self.config.dataset_name,
            limit=self.config.max_examples,
        ))

        logger.info("Fetched %d examples from dataset '%s'",
                   len(examples), self.config.dataset_name)

        # Run evaluation on each example
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def run_with_semaphore(example: Example) -> ExperimentResult:
            async with semaphore:
                return await self._evaluate_single_example(example)

        tasks = [run_with_semaphore(ex) for ex in examples]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Build summary
        total = len(results)
        successful = sum(1 for r in results if r.execution_status == "success")
        failed = sum(1 for r in results if r.execution_status == "failed")
        skipped = sum(1 for r in results if r.execution_status == "skipped")
        total_time = sum(r.execution_time_ms for r in results)

        # Compute evaluator summary
        evaluator_summary = self._compute_evaluator_summary(results)

        summary = ExperimentSummary(
            experiment_name=self.config.experiment_name,
            dataset_name=self.config.dataset_name,
            total_examples=total,
            successful=successful,
            failed=failed,
            skipped=skipped,
            execution_time_ms=total_time,
            evaluator_summary=evaluator_summary,
        )

        logger.info(
            "Experiment complete: %d total, %d successful, %d failed, %d skipped",
            total, successful, failed, skipped
        )

        return results, summary

    def _compute_evaluator_summary(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute summary statistics for all evaluators."""
        summary = {}

        # Collect all evaluator names used
        all_evaluators = set()
        for result in results:
            all_evaluators.update(result.evaluator_results.keys())

        for evaluator_name in sorted(all_evaluators):
            eval_results = [
                r.evaluator_results.get(evaluator_name, {}) for r in results
            ]
            passed = sum(1 for r in eval_results if r.get("passed") is True)
            failed = sum(1 for r in eval_results if r.get("passed") is False)
            needs_review = sum(1 for r in eval_results if r.get("status") == "needs_human_review")
            insufficient = sum(1 for r in eval_results if r.get("status") == "insufficient_context")
            errors = sum(1 for r in eval_results if r.get("status") == "error")

            scores = [r.get("score") for r in eval_results if r.get("score") is not None]
            avg_score = sum(scores) / len(scores) if scores else None

            confidences = [r.get("confidence") for r in eval_results if r.get("confidence") is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            summary[evaluator_name] = {
                "passed": passed,
                "failed": failed,
                "needs_human_review": needs_review,
                "insufficient_context": insufficient,
                "errors": errors,
                "average_score": round(avg_score, 3) if avg_score is not None else None,
                "average_confidence": round(avg_confidence, 3) if avg_confidence is not None else None,
            }

        return summary

    def save_results(
        self,
        results: List[ExperimentResult],
        summary: ExperimentSummary,
        output_path: str,
    ) -> None:
        """Save experiment results to JSON file."""
        output_data = {
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Experiment results saved to %s", output_path)


# =============================================================================
# Dataset Management Functions
# =============================================================================

def validate_dataset_before_upload(
    filepath: str,
    exclude_unverified: bool = True,
) -> Dict[str, Any]:
    """Validate dataset before uploading to LangSmith.

    Args:
        filepath: Path to JSONL dataset file
        exclude_unverified: Whether to exclude cases needing human verification

    Returns:
        Dict with validation results and statistics
    """
    from app.ai.evaluation.validation import validate_dataset
    from app.ai.evaluation.dataset_schema import load_dataset, get_dataset_stats

    # Load and validate
    cases = load_dataset(filepath)
    stats = get_dataset_stats(cases)

    # Run full validation
    validation_report = validate_dataset(filepath)

    # Count unverified cases
    unverified_count = sum(1 for case in cases if case.needs_human_verification)

    return {
        "valid": validation_report.is_valid,
        "total_cases": len(cases),
        "valid_cases": validation_report.valid_cases,
        "invalid_cases": validation_report.invalid_cases,
        "unverified_cases": unverified_count,
        "categories": stats["categories"],
        "languages": stats["languages"],
        "validation_report": validation_report.to_dict(),
    }


def dry_run_dataset_upload(
    filepath: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    exclude_unverified: bool = True,
) -> Dict[str, Any]:
    """Simulate dataset upload without making external API calls.

    Args:
        filepath: Path to JSONL dataset file
        dataset_name: Target LangSmith dataset name
        exclude_unverified: Whether to exclude cases needing human verification

    Returns:
        Dict with what would be uploaded
    """
    from app.ai.evaluation.dataset_schema import load_dataset, get_dataset_stats
    from app.ai.evaluation.validation import validate_dataset

    cases = load_dataset(filepath)

    # Filter unverified if requested
    if exclude_unverified:
        cases = [c for c in cases if not c.needs_human_verification]

    # Validate
    validation_report = validate_dataset(filepath)

    # Compute stats on filtered cases
    stats = get_dataset_stats(cases)

    # Build example previews
    examples = []
    for case in cases:
        examples.append({
            "id": case.id,
            "question": case.question[:100] + "..." if len(case.question) > 100 else case.question,
            "category": case.category,
            "language": case.language,
            "needs_human_verification": case.needs_human_verification,
        })

    return {
        "dataset_name": dataset_name,
        "total_cases_in_file": len(load_dataset(filepath)),
        "cases_after_filtering": len(cases),
        "excluded_unverified": sum(1 for c in load_dataset(filepath) if c.needs_human_verification) if exclude_unverified else 0,
        "validation_passed": validation_report.is_valid,
        "examples_preview": examples[:10],  # Show first 10
        "categories": stats["categories"],
        "languages": stats["languages"],
    }


def upload_evaluation_dataset(
    filepath: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    api_key: Optional[str] = None,
    description: str = "",
    create_if_missing: bool = True,
    exclude_unverified: bool = True,
) -> Dict[str, Any]:
    """Upload evaluation dataset to LangSmith with filtering.

    Args:
        filepath: Path to JSONL dataset file
        dataset_name: Target LangSmith dataset name
        api_key: Optional LangSmith API key
        description: Dataset description
        create_if_missing: Create dataset if it doesn't exist
        exclude_unverified: Exclude cases requiring human verification

    Returns:
        Upload results
    """
    # This reuses the existing upload_dataset_from_file but adds filtering
    from app.ai.evaluation.dataset_schema import load_dataset

    # Load and optionally filter
    cases = load_dataset(filepath)
    if exclude_unverified:
        original_count = len(cases)
        cases = [c for c in cases if not c.needs_human_verification]
        logger.info("Excluded %d cases requiring human verification",
                   original_count - len(cases))

    # Save filtered cases to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for case in cases:
            f.write(case.to_jsonl() + "\n")
        temp_path = f.name

    try:
        result = upload_dataset_from_file(
            filepath=temp_path,
            dataset_name=dataset_name,
            api_key=api_key,
            description=description,
            create_if_missing=create_if_missing,
        )
        return result
    finally:
        os.unlink(temp_path)


# =============================================================================
# Experiment Result Export
# =============================================================================

def export_experiment_results(
    experiment_name: str,
    output_path: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Export experiment results from LangSmith.

    Args:
        experiment_name: Name of the experiment
        output_path: Path to save exported results
        api_key: Optional LangSmith API key

    Returns:
        Export results
    """
    client = create_langsmith_client(api_key)

    # Find experiment
    experiments = list(client.list_experiments(experiment_name=experiment_name))
    if not experiments:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    experiment = experiments[0]

    # Get experiment results
    results = list(client.list_experiment_results(experiment_id=experiment.id))

    export_data = {
        "experiment_name": experiment_name,
        "experiment_id": str(experiment.id),
        "dataset_name": experiment.dataset_name,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "results": [],
    }

    for result in results:
        export_data["results"].append({
            "id": str(result.id),
            "inputs": result.inputs,
            "outputs": result.outputs,
            "feedback": result.feedback,
            "evaluation_results": result.evaluation_results,
            "latency_ms": result.latency_ms,
            "error": result.error,
        })

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d results to %s", len(results), output_path)

    return {
        "experiment_name": experiment_name,
        "total_results": len(results),
        "output_path": str(output_file),
    }


# =============================================================================
# Utility Functions
# =============================================================================

import tempfile  # Moved here to avoid circular import issues