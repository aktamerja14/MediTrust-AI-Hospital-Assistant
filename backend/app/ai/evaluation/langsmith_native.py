"""Native LangSmith Experiment Integration for MediTrust RAG Evaluation (Phase 6.1/6.2).

This module provides native LangSmith experiment integration using the official
`langsmith.evaluation.evaluate()` API. It wraps existing Phase 4 programmatic
evaluators and Phase 5 LLM judges as LangSmith-compatible evaluators.

Phase 6.2 improvements:
- Eliminates duplicate RAG/evaluator execution by using native results as single source of truth
- Adds feedback ID tracking and mapping
- Adds PHI/sensitive data sanitization for all LangSmith uploads
"""
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Awaitable, Union

from langsmith import Client
from langsmith.evaluation import evaluate, aevaluate, aevaluate_existing
from langsmith.schemas import Example, Run
from langsmith.evaluation import EvaluationResult as LangSmithEvaluationResult
from langsmith.evaluation.evaluator import RunEvaluator

from app.ai.factory import ProviderFactory
from app.ai.providers.base import LLMMessage, BaseLLMProvider, LLMProviderError
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
    PROGRAMMATIC_EVALUATORS,
    DEFAULT_EVALUATORS as PROGRAMMATIC_DEFAULT_EVALUATORS,
    EvaluationStatus,
    EvaluatorResult,
    get_evaluator as get_programmatic_evaluator,
)
from app.ai.evaluation.llm_judges import (
    LLM_JUDGES,
    LLM_DEFAULT_EVALUATORS,
    LLMJudgeResult,
    JudgeStatus,
    CorrectnessJudge,
    FaithfulnessJudge,
    RelevanceJudge,
    MedicalSafetyJudge,
)
from app.ai.evaluation.langsmith_utils import (
    create_langsmith_client,
    upload_dataset_from_file,
    DEFAULT_DATASET_NAME,
    LangSmithUploadError,
    list_datasets,
)
from app.ai.evaluation.validation import validate_dataset
from app.ai.rag.chain import RAGChain, RAGResponse

logger = logging.getLogger("meditrust.ai.evaluation.langsmith_native")

from app.ai.evaluation.langsmith_utils import (
    detect_sensitive_data,
    sanitize_text,
    sanitize_dict,
    validate_no_sensitive_data,
    validate_dict_no_sensitive_data,
    SanitizationError,
    _sanitize_case_for_upload,
    sanitize_experiment_payload,
    ExperimentSanitizationError,
)


# =============================================================================
# LangSmith-Compatible Evaluator Wrappers
# =============================================================================

class ProgrammaticEvaluatorWrapper(RunEvaluator):
    """Wraps a Phase 4 programmatic evaluator as a LangSmith RunEvaluator."""

    def __init__(self, evaluator_name: str, evaluator_func: Callable):
        self._evaluator_name = evaluator_name
        self._evaluator_func = evaluator_func

    @property
    def name(self) -> str:
        return self._evaluator_name

    @property
    def feedback_keys(self) -> List[str]:
        return [self._evaluator_name]

    def evaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[uuid.UUID] = None,
    ) -> LangSmithEvaluationResult:
        """Evaluate a single run using the wrapped programmatic evaluator."""
        inputs = run.inputs if run.inputs else {}
        outputs = run.outputs if run.outputs else {}
        example_inputs = example.inputs if example else {}
        example_outputs = example.outputs if example else {}

        pipeline_result = PipelineEvaluationResult(
            evaluation_id=str(run.id),
            question=inputs.get("question", "") or example_inputs.get("question", ""),
            expected_answer=example_outputs.get("expected_answer", ""),
            actual_answer=outputs.get("answer", "") or outputs.get("actual_answer", ""),
            reference_context=example_outputs.get("reference_context", ""),
            retrieved_context=outputs.get("retrieved_context", ""),
            category=inputs.get("category", "") or example_inputs.get("category", "general_info"),
            language=inputs.get("language", "") or example_inputs.get("language", "en"),
            tags=inputs.get("tags", []) or example_inputs.get("tags", []),
            execution_status="success" if outputs.get("answer") else "failed",
            error_message=outputs.get("error_message"),
            sources=outputs.get("sources", []),
            has_context=outputs.get("has_context", False),
            model=outputs.get("model", ""),
            doctor_recommendations=outputs.get("doctor_recommendations", []),
            disclaimer=outputs.get("disclaimer", ""),
            execution_time_ms=0.0,
        )

        try:
            result = self._evaluator_func(pipeline_result)

            score = result.score
            if score is None:
                if result.status == EvaluationStatus.PASSED:
                    score = 1.0
                elif result.status == EvaluationStatus.FAILED:
                    score = 0.0
                else:
                    score = None

            value = result.status.value

            # PHI protection: evaluator comments/metadata are uploaded as
            # LangSmith feedback, so sanitize before returning them.
            safe_issues = [sanitize_text(str(issue)) for issue in (result.issues or [])]
            safe_metadata = sanitize_dict(result.metadata) if result.metadata else {}

            return LangSmithEvaluationResult(
                key=self._evaluator_name,
                score=score,
                value=value,
                comment=sanitize_text(result.reason or ""),
                metadata={
                    "issues": safe_issues,
                    "evaluator_status": result.status.value,
                    "passed": result.passed,
                    **safe_metadata,
                }
            )
        except Exception as e:
            logger.exception("Programmatic evaluator %s failed: %s", self._evaluator_name, e)
            safe_error = sanitize_text(str(e))
            return LangSmithEvaluationResult(
                key=self._evaluator_name,
                score=None,
                value="error",
                comment=f"Evaluator error: {safe_error}",
                metadata={"error": safe_error}
            )


class LLMJudgeWrapper(RunEvaluator):
    """Wraps a Phase 5 LLM judge as a LangSmith RunEvaluator."""

    def __init__(
        self,
        judge_name: str,
        judge_instance,
    ):
        self._judge_name = judge_name
        self._judge = judge_instance

    @property
    def name(self) -> str:
        return self._judge_name

    @property
    def feedback_keys(self) -> List[str]:
        return [self._judge_name]

    def evaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[uuid.UUID] = None,
    ) -> LangSmithEvaluationResult:
        raise NotImplementedError("Use aevaluate_run for LLM judges")

    async def aevaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[uuid.UUID] = None,
    ) -> LangSmithEvaluationResult:
        """Evaluate a single run using the LLM judge."""
        inputs = run.inputs if run.inputs else {}
        outputs = run.outputs if run.outputs else {}
        example_inputs = example.inputs if example else {}
        example_outputs = example.outputs if example else {}

        pipeline_result = PipelineEvaluationResult(
            evaluation_id=str(run.id),
            question=inputs.get("question", "") or example_inputs.get("question", ""),
            expected_answer=example_outputs.get("expected_answer", ""),
            actual_answer=outputs.get("answer", "") or outputs.get("actual_answer", ""),
            reference_context=example_outputs.get("reference_context", ""),
            retrieved_context=outputs.get("retrieved_context", ""),
            category=inputs.get("category", "") or example_inputs.get("category", "general_info"),
            language=inputs.get("language", "") or example_inputs.get("language", "en"),
            tags=inputs.get("tags", []) or example_inputs.get("tags", []),
            execution_status="success" if outputs.get("answer") else "failed",
            error_message=outputs.get("error_message"),
            sources=outputs.get("sources", []),
            has_context=outputs.get("has_context", False),
            model=outputs.get("model", ""),
            doctor_recommendations=outputs.get("doctor_recommendations", []),
            disclaimer=outputs.get("disclaimer", ""),
            execution_time_ms=0.0,
        )

        try:
            result = await self._judge.evaluate(pipeline_result)

            status_value = result.status.value
            score = result.score
            if score is None:
                if result.status == JudgeStatus.PASSED:
                    score = 1.0
                elif result.status == JudgeStatus.FAILED:
                    score = 0.0
                else:
                    score = None

            # PHI protection: judge reasoning/evidence are uploaded as
            # LangSmith feedback, so sanitize before returning them.
            safe_evidence = [sanitize_text(str(item)) for item in (result.evidence or [])]
            safe_issues = [sanitize_text(str(issue)) for issue in (result.issues or [])]

            return LangSmithEvaluationResult(
                key=self._judge_name,
                score=score,
                value=status_value,
                comment=sanitize_text(result.reasoning or ""),
                metadata={
                    "evidence": safe_evidence,
                    "issues": safe_issues,
                    "confidence": result.confidence,
                    "judge_model": result.judge_model,
                    "judge_provider": result.judge_provider,
                }
            )
        except Exception as e:
            logger.exception("LLM judge %s failed: %s", self._judge_name, e)
            safe_error = sanitize_text(str(e))
            return LangSmithEvaluationResult(
                key=self._judge_name,
                score=None,
                value="error",
                comment=f"Judge error: {safe_error}",
                metadata={"error": safe_error}
            )


# =============================================================================
# RAG Chain Target Wrapper for LangSmith
# =============================================================================

class RAGChainTarget:
    """Wraps the MediTrust RAGChain as a LangSmith-compatible target."""

    def __init__(self, rag_chain: Optional[RAGChain] = None):
        self._rag_chain = rag_chain

    @property
    def rag_chain(self) -> RAGChain:
        if self._rag_chain is None:
            self._rag_chain = RAGChain()
        return self._rag_chain

    async def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """LangSmith target interface - takes inputs dict, returns outputs dict.

        Phase 6.3: Fail-closed PHI sanitization on the entire output dict before
        it is transmitted to LangSmith as a run output.
        """
        question = inputs.get("question", "")
        conversation_history = inputs.get("conversation_history")
        top_k = inputs.get("top_k")
        # Use example-level case_id for redacted log messages when available.
        case_id = str(inputs.get("case_id", inputs.get("example_id", "unknown")))

        try:
            response = await self.rag_chain.generate(
                query=question,
                conversation_history=conversation_history,
                top_k=top_k,
            )

            # Build the raw output dict.
            raw_output = {
                "answer": response.answer or "",
                "sources": response.sources or [],
                "has_context": response.has_context,
                "model": response.model,
                "doctor_recommendations": response.doctor_recommendations or [],
                "disclaimer": response.disclaimer or "",
                "langsmith_run_id": response.langsmith_run_id,
                "langsmith_trace_id": response.langsmith_trace_id,
            }

            # Fail-closed: sanitize and re-validate before handing to LangSmith.
            safe_output = sanitize_experiment_payload(
                raw_output,
                case_id=case_id,
                fields=["answer", "disclaimer", "error_message"],
            )
            # Sources are dicts — sanitize recursively.
            safe_output["sources"] = sanitize_dict(
                {"sources": safe_output.get("sources", [])}
            ).get("sources", [])
            safe_output["doctor_recommendations"] = sanitize_dict(
                {"items": safe_output.get("doctor_recommendations", [])}
            ).get("items", [])
            return safe_output

        except ExperimentSanitizationError:
            # Re-raise immediately — do not send anything to LangSmith.
            raise
        except Exception as e:
            logger.exception("RAG chain generation failed: %s", e)
            return {
                "answer": "",
                "sources": [],
                "has_context": False,
                "model": "",
                "doctor_recommendations": [],
                "disclaimer": "",
                "error_message": sanitize_text(str(e)),
            }


# =============================================================================
# LangSmith Native Experiment Runner
# =============================================================================

@dataclass
class NativeExperimentConfig:
    """Configuration for a native LangSmith experiment run."""
    experiment_name: str
    dataset_name: str = "meditrust-rag-evaluation-v1"
    evaluator_names: Optional[List[str]] = None
    include_programmatic: bool = True
    include_llm_judges: bool = True
    max_concurrent: int = 3
    max_examples: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    experiment_prefix: Optional[str] = None
    description: Optional[str] = None
    num_repetitions: int = 1
    project: Optional[str] = None  # Phase 6.3: LangSmith project name


@dataclass
class NativeExampleResult:
    """Per-example result extracted from a native LangSmith experiment row."""
    example_id: str
    run_id: str
    question: str
    expected_answer: str
    actual_answer: str
    reference_context: str
    retrieved_context: str = ""
    category: str = "general_info"
    language: str = "en"
    tags: List[str] = field(default_factory=list)
    execution_status: str = "success"
    evaluator_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error_message: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    has_context: bool = False
    model: str = ""
    doctor_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NativeExperimentResult:
    """Result of a native LangSmith experiment."""
    experiment_name: str
    experiment_id: str
    dataset_name: str
    total_examples: int = 0
    successful: int = 0
    failed: int = 0
    errors: int = 0
    execution_time_ms: float = 0.0
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    evaluator_summary: Dict[str, Any] = field(default_factory=dict)
    langsmith_experiment_url: Optional[str] = None
    run_ids: List[str] = field(default_factory=list)
    dataset_id: Optional[str] = None
    feedback_ids: List[str] = field(default_factory=list)
    feedback_mapping: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NativeLangSmithExperimentRunner:
    """Runs native LangSmith experiments using the official evaluate() API."""

    def __init__(
        self,
        config: NativeExperimentConfig,
        rag_chain: Optional[RAGChain] = None,
        client: Optional[Client] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config
        self._rag_chain = rag_chain
        self.client = client or create_langsmith_client(api_key, project=config.project)
        self._target: Optional[RAGChainTarget] = None
        self._programmatic_evaluators: Dict[str, ProgrammaticEvaluatorWrapper] = {}
        self._llm_judges: Dict[str, LLMJudgeWrapper] = {}
        self._feedback_mapping: List[Dict[str, Any]] = []

    @property
    def rag_chain(self) -> RAGChain:
        if self._rag_chain is None:
            self._rag_chain = RAGChain()
        return self._rag_chain

    @property
    def target(self) -> RAGChainTarget:
        if self._target is None:
            self._target = RAGChainTarget(self.rag_chain)
        return self._target

    def _get_programmatic_evaluators(self) -> Dict[str, ProgrammaticEvaluatorWrapper]:
        """Get or create programmatic evaluator wrappers."""
        if not self._programmatic_evaluators:
            evaluator_names = self.config.evaluator_names
            if evaluator_names is None:
                evaluator_names = []
                if self.config.include_programmatic:
                    evaluator_names.extend([n for n in PROGRAMMATIC_EVALUATORS.keys()])
                if self.config.include_llm_judges:
                    evaluator_names.extend([n for n in LLM_JUDGES.keys()])

            for name in evaluator_names:
                if name in PROGRAMMATIC_EVALUATORS:
                    self._programmatic_evaluators[name] = ProgrammaticEvaluatorWrapper(
                        name, PROGRAMMATIC_EVALUATORS[name]
                    )
                elif name in LLM_JUDGES:
                    judge_class = LLM_JUDGES[name]
                    self._programmatic_evaluators[name] = LLMJudgeWrapper(
                        name, judge_class()
                    )
        return self._programmatic_evaluators

    def _get_evaluator_list(self) -> List[RunEvaluator]:
        """Get list of evaluators for the experiment."""
        evaluators = []

        if self.config.include_programmatic:
            for name in PROGRAMMATIC_DEFAULT_EVALUATORS:
                if self.config.evaluator_names is None or name in self.config.evaluator_names:
                    if name in PROGRAMMATIC_EVALUATORS:
                        evaluators.append(ProgrammaticEvaluatorWrapper(
                            name, PROGRAMMATIC_EVALUATORS[name]
                        ))

        if self.config.include_llm_judges:
            for name in LLM_DEFAULT_EVALUATORS:
                if self.config.evaluator_names is None or name in self.config.evaluator_names:
                    if name in LLM_JUDGES:
                        judge_class = LLM_JUDGES[name]
                        evaluators.append(LLMJudgeWrapper(name, judge_class()))

        return evaluators

    async def run_experiment(self) -> tuple[List[Dict[str, Any]], NativeExperimentResult]:
        """Run the native LangSmith experiment using the official evaluate() API.
        
        Phase 6.2: Uses native LangSmith experiment results as the single source of truth.
        Does NOT re-run RAG generation or evaluators locally - extracts results from
        the native experiment to build local summary reports.
        """
        logger.info(
            "Starting native LangSmith experiment '%s' on dataset '%s'",
            self.config.experiment_name, self.config.dataset_name
        )

        start_time = datetime.utcnow()

        raw_target = self.target
        if hasattr(raw_target, "ainvoke"):
            async def target(inputs: dict) -> dict:
                return await raw_target.ainvoke(inputs)
        elif hasattr(raw_target, "invoke"):
            def target(inputs: dict) -> dict:
                return raw_target.invoke(inputs)
        elif callable(raw_target):
            async def target(inputs: dict) -> dict:
                res = raw_target(inputs)
                return await res if hasattr(res, "__await__") else res
        else:
            target = raw_target
        evaluators = self._get_evaluator_list()

        logger.info(
            "Running experiment with %d evaluators: %s",
            len(evaluators), [e.name for e in evaluators]
        )

        try:
            experiment_results = await aevaluate(
                target,
                data=self.config.dataset_name,
                evaluators=evaluators,
                experiment_prefix=self.config.experiment_prefix,
                description=self.config.description,
                max_concurrency=self.config.max_concurrent,
                num_repetitions=self.config.num_repetitions,
                client=self.client,
                metadata=self.config.metadata,
            )

            await experiment_results.wait()

            experiment_id = str(experiment_results.experiment_id)
            experiment_url = experiment_results.url
            dataset_id = await experiment_results.get_dataset_id()

            run_ids = []
            feedback_ids = []
            feedback_mapping = []

            # Collect experiment rows to extract results
            experiment_rows = []
            async for row in experiment_results:
                if row.get("run"):
                    run_ids.append(str(row["run"].id))
                experiment_rows.append(row)

            logger.info(
                "Native LangSmith experiment completed: experiment_id=%s, runs=%d",
                experiment_id, len(run_ids)
            )

            # Extract evaluation results from native experiment rows (single source of truth)
            example_results = await self._extract_results_from_experiment(
                experiment_rows, run_ids
            )

            # Retrieve feedback IDs from LangSmith (mapped to run/evaluator)
            feedback_ids = await self._retrieve_feedback_ids(experiment_id, run_ids)
            feedback_mapping = list(self._feedback_mapping)

            total = len(example_results)
            successful = sum(1 for r in example_results if r.execution_status == "success")
            failed = sum(1 for r in example_results if r.execution_status == "failed")
            errors = sum(1 for r in example_results if r.execution_status == "error")

            duration_seconds = (datetime.utcnow() - start_time).total_seconds()
            execution_time_ms = duration_seconds * 1000

            evaluator_results_by_example = {
                r.example_id: r.evaluator_results for r in example_results
            }
            evaluator_summary = self._compute_evaluator_summary_from_native(evaluator_results_by_example)

            summary = NativeExperimentResult(
                experiment_name=self.config.experiment_name,
                experiment_id=experiment_id,
                dataset_name=self.config.dataset_name,
                total_examples=total,
                successful=successful,
                failed=failed,
                errors=errors,
                execution_time_ms=execution_time_ms,
                duration_seconds=duration_seconds,
                evaluator_summary=evaluator_summary,
                langsmith_experiment_url=experiment_url,
                run_ids=run_ids,
                dataset_id=dataset_id,
                feedback_ids=feedback_ids,
                feedback_mapping=feedback_mapping,
            )

            detailed_results = [r.to_dict() for r in example_results]

            return detailed_results, summary

        except Exception as e:
            logger.exception("Native experiment failed: %s", e)
            raise

    @staticmethod
    def _row_get(obj: Any, key: str, default: Any = None) -> Any:
        """Read a field from either a dict-like row or an SDK model object."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _as_dict(cls, value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    async def _extract_results_from_experiment(
        self,
        experiment_rows: List[Dict[str, Any]],
        run_ids: List[str],
    ) -> List[NativeExampleResult]:
        """Extract example results and evaluator results from native experiment rows.

        This uses the native LangSmith experiment results as the single source of
        truth, avoiding duplicate RAG generation and evaluator execution. Each
        ExperimentResultRow already contains the example, the run, and the
        evaluator feedback produced by LangSmith, so no second local run is needed.

        Args:
            experiment_rows: Rows from the native experiment results iterator
            run_ids: List of run IDs from the experiment (kept for compatibility)

        Returns:
            List of NativeExampleResult objects, one per experiment row
        """
        example_results: List[NativeExampleResult] = []

        for row in experiment_rows:
            run = self._row_get(row, "run")
            example = self._row_get(row, "example")
            if run is None or example is None:
                continue

            run_id = str(self._row_get(run, "id", ""))
            example_id = str(self._row_get(example, "id", run_id))
            run_outputs = self._as_dict(self._row_get(run, "outputs", {}))
            example_inputs = self._as_dict(self._row_get(example, "inputs", {}))
            example_outputs = self._as_dict(self._row_get(example, "outputs", {}))

            # LangSmith returns EvaluationResults = {"results": [EvaluationResult, ...]}
            raw_eval = self._row_get(row, "evaluation_results", {})
            raw_results = []
            if isinstance(raw_eval, dict):
                raw_results = raw_eval.get("results", [])
            elif isinstance(raw_eval, list):
                raw_results = raw_eval

            evaluator_results = {}
            for eval_result in raw_results:
                eval_name = self._row_get(eval_result, "key")
                if not eval_name:
                    continue
                metadata = self._as_dict(self._row_get(eval_result, "metadata", {}))
                value = self._row_get(eval_result, "value")
                passed = metadata.get("passed")
                if passed is None:
                    if value == "passed":
                        passed = True
                    elif value == "failed":
                        passed = False

                evaluator_results[str(eval_name)] = {
                    "score": self._row_get(eval_result, "score"),
                    "passed": passed,
                    "status": value if value is not None else metadata.get("evaluator_status", "error"),
                    "reason": sanitize_text(str(self._row_get(eval_result, "comment", "") or "")),
                    "issues": [
                        sanitize_text(str(issue))
                        for issue in (metadata.get("issues") or [])
                    ],
                    "confidence": metadata.get("confidence"),
                    "metadata": sanitize_dict(metadata) if metadata else {},
                }

            example_results.append(NativeExampleResult(
                example_id=example_id,
                run_id=run_id,
                question=sanitize_text(str(example_inputs.get("question", "") or "")),
                evaluator_results=evaluator_results,
                expected_answer=sanitize_text(str(example_outputs.get("expected_answer", "") or "")),
                actual_answer=sanitize_text(str(
                    run_outputs.get("answer", "") or run_outputs.get("actual_answer", "") or ""
                )),
                reference_context=sanitize_text(str(example_outputs.get("reference_context", "") or "")),
                retrieved_context=sanitize_text(str(run_outputs.get("retrieved_context", "") or "")),
                category=example_inputs.get("category", "general_info"),
                language=example_inputs.get("language", "en"),
                tags=example_inputs.get("tags", []),
                execution_status="success" if run_outputs.get("answer") else "failed",
                error_message=sanitize_text(str(run_outputs.get("error_message") or "")) or None,
                sources=sanitize_dict({"sources": run_outputs.get("sources", [])}).get("sources", []),
                has_context=run_outputs.get("has_context", False),
                model=run_outputs.get("model", ""),
                doctor_recommendations=sanitize_dict(
                    {"items": run_outputs.get("doctor_recommendations", [])}
                ).get("items", []),
                disclaimer=sanitize_text(str(run_outputs.get("disclaimer", "") or "")),
                execution_time_ms=0.0,  # Not available from native experiment
            ))

        return example_results

    async def _retrieve_feedback_ids(
        self,
        experiment_id: str,
        run_ids: List[str],
    ) -> List[str]:
        """Retrieve feedback IDs from LangSmith for the experiment runs.
        
        Uses the public LangSmith SDK API to fetch feedback records and maps them
        to experiment ID, run ID, example ID, evaluator name, and score.
        
        Args:
            experiment_id: The experiment ID
            run_ids: List of run IDs from the experiment
            
        Returns:
            List of feedback IDs
        """
        self._feedback_mapping = []
        feedback_ids: List[str] = []

        if not run_ids:
            return feedback_ids

        try:
            # LangSmith SDK v0.15.0 public API supports filtering by run IDs.
            feedback_records = list(self.client.list_feedback(run_ids=run_ids))

            for feedback in feedback_records:
                feedback_id = str(getattr(feedback, "id", "") or "")
                run_id = str(getattr(feedback, "run_id", "") or "")
                if feedback_id:
                    feedback_ids.append(feedback_id)

                # Map feedback -> run/evaluator. Comment is sanitized because
                # it may echo answer/context text.
                self._feedback_mapping.append({
                    "feedback_id": feedback_id,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "example_id": str(getattr(feedback, "example_id", "") or ""),
                    "evaluator_name": str(getattr(feedback, "key", "") or ""),
                    "score": getattr(feedback, "score", None),
                    "value": getattr(feedback, "value", ""),
                    "comment": sanitize_text(str(getattr(feedback, "comment", "") or "")),
                })

            logger.info(
                "Retrieved %d feedback records for experiment %s",
                len(feedback_ids), experiment_id
            )
            for mapping in self._feedback_mapping:
                logger.debug(
                    "Feedback mapping: run=%s, evaluator=%s, score=%s",
                    mapping["run_id"][:8] + "...", mapping["evaluator_name"], mapping["score"]
                )

        except Exception as e:
            # Handle missing feedback / SDK errors gracefully - never fail the experiment
            logger.warning(
                "Could not retrieve feedback IDs: %s. Continuing without feedback tracking.",
                sanitize_text(str(e)),
            )
            self._feedback_mapping = []
            feedback_ids = []

        return feedback_ids

    def _compute_evaluator_summary_from_native(
        self,
        evaluator_results_by_example: Dict[str, Dict[str, Dict]],
    ) -> Dict[str, Any]:
        """Compute summary statistics from native experiment evaluator results.
        
        Args:
            evaluator_results_by_example: Dict mapping example_id -> evaluator_name -> evaluator_result
            
        Returns:
            Summary statistics dict
        """
        summary = {}
        
        # Collect all evaluator names
        all_evaluators = {}
        for example_id, eval_results in evaluator_results_by_example.items():
            for eval_name, eval_result in eval_results.items():
                if eval_name not in all_evaluators:
                    all_evaluators[eval_name] = []
                all_evaluators[eval_name].append(eval_result)
        
        for name, results_list in all_evaluators.items():
            if not results_list:
                continue
            
            passed = sum(1 for r in results_list if r.get("passed") is True)
            failed = sum(1 for r in results_list if r.get("passed") is False)
            needs_review = sum(1 for r in results_list if r.get("status") == "needs_human_review")
            insufficient = sum(1 for r in results_list if r.get("status") == "insufficient_context")
            errors = sum(1 for r in results_list if r.get("status") == "error")
            
            scores = [r.get("score") for r in results_list if r.get("score") is not None]
            avg_score = sum(scores) / len(scores) if scores else None
            
            confidences = [r.get("confidence") for r in results_list if r.get("confidence") is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            summary[name] = {
                "total": len(results_list),
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
        results: List[Dict[str, Any]],
        summary: "NativeExperimentResult",
        output_path: str,
    ) -> None:
        """Save experiment results to a JSON file.

        Writes only metrics and ``case_id`` values — no raw clinical text.

        Args:
            results: Per-example result dicts (from ``run_experiment``).
            summary: Experiment-level summary object.
            output_path: Filesystem path to write JSON output.
        """
        output_data = {
            "summary": summary.to_dict(),
            "results": results,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Experiment results saved to %s", output_path)


# =============================================================================
# Dataset Deduplication
# =============================================================================

async def upload_dataset_deduplicated(
    filepath: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    api_key: Optional[str] = None,
    description: str = "",
    create_if_missing: bool = True,
    exclude_unverified: bool = True,
    update_existing: bool = False,
    sanitize: bool = True,
) -> Dict[str, Any]:
    """Upload evaluation dataset with deduplication by case_id.

    Uses case_id from the source JSONL as the stable identifier.
    Checks for existing examples with the same case_id before uploading.

    Phase 6.2: Applies PHI/sensitive data sanitization before upload (fail-closed).

    Args:
        filepath: Path to JSONL dataset file
        dataset_name: Target LangSmith dataset name
        api_key: Optional LangSmith API key
        description: Dataset description
        create_if_missing: Create dataset if it doesn't exist
        exclude_unverified: Exclude cases requiring human verification
        update_existing: If True, update existing examples; if False, skip them
        sanitize: Whether to apply PHI sanitization (default: True, fail-closed)

    Returns:
        Upload results with deduplication stats

    Raises:
        SanitizationUploadError: If sensitive data is detected and cannot be safely handled
    """
    from app.ai.evaluation.dataset_schema import load_dataset
    from app.ai.evaluation.validation import validate_dataset
    from app.ai.evaluation.langsmith_utils import upload_evaluation_cases, SanitizationUploadError

    client = create_langsmith_client(api_key)

    cases = load_dataset(filepath)
    validation_report = validate_dataset(filepath)

    if not validation_report.is_valid:
        raise ValueError(f"Dataset validation failed: {validation_report}")

    if exclude_unverified:
        cases = [c for c in cases if not c.needs_human_verification]

    existing_case_ids = set()
    existing_examples_by_case_id = {}
    try:
        existing_examples = list(client.list_examples(
            dataset_name=dataset_name,
            limit=None,
        ))
        for ex in existing_examples:
            case_id = ex.metadata.get("case_id") or ex.inputs.get("case_id")
            if case_id:
                existing_case_ids.add(case_id)
                existing_examples_by_case_id[case_id] = ex.id
        logger.info("Found %d existing examples in dataset '%s'", len(existing_case_ids), dataset_name)
    except Exception as e:
        logger.warning("Could not fetch existing examples: %s", e)

    new_cases = []
    skipped_duplicates = 0
    update_cases = []
    for case in cases:
        if case.id in existing_case_ids:
            if update_existing:
                existing_example_id = existing_examples_by_case_id.get(case.id)
                if existing_example_id:
                    update_cases.append((case, existing_example_id))
                else:
                    skipped_duplicates += 1
            else:
                skipped_duplicates += 1
        else:
            new_cases.append(case)

    logger.info(
        "Dataset upload: %d total, %d new, %d updates, %d skipped (duplicates)",
        len(cases), len(new_cases), len(update_cases), skipped_duplicates
    )

    uploaded = 0
    updated = 0
    validation_errors = []
    upload_errors = []
    sanitization_errors = []

    # Apply sanitization to new cases and update cases
    all_cases_to_upload = new_cases + [c for c, _ in update_cases]
    sanitized_cases = []
    for case in all_cases_to_upload:
        try:
            if sanitize:
                sanitized_case = _sanitize_case_for_upload(case)
                sanitized_cases.append(sanitized_case)
            else:
                sanitized_cases.append(case)
        except SanitizationError as e:
            logger.error("Sanitization failed for case %s: %s", case.id, e)
            sanitization_errors.append({"case_id": case.id, "error": str(e)})

    if sanitization_errors:
        raise SanitizationUploadError(
            f"Upload blocked: {len(sanitization_errors)} case(s) contain sensitive data that cannot be safely uploaded. "
            f"Errors: {sanitization_errors}"
        )

    if sanitized_cases:
        # Upload sanitized cases using the existing upload function
        result = upload_evaluation_cases(
            client=client,
            dataset_name=dataset_name,
            cases=sanitized_cases,
            sanitize=False,  # Already sanitized
        )
        uploaded = result.get("uploaded", 0)
        validation_errors = result.get("validation_errors", [])
        upload_errors = result.get("upload_errors", [])

    if update_cases:
        for case, example_id in update_cases:
            try:
                # Sanitize update data
                if sanitize:
                    update_inputs = {"question": sanitize_text(case.question)}
                    update_outputs = {
                        "expected_answer": sanitize_text(case.expected_answer),
                        "reference_context": sanitize_text(case.reference_context),
                        "needs_human_verification": case.needs_human_verification,
                        "metadata": sanitize_dict(case.metadata) if case.metadata else {},
                    }
                    update_metadata = {
                        "case_id": case.id,
                        "created_at": case.created_at,
                        **(sanitize_dict(case.metadata) if case.metadata else {}),
                    }
                else:
                    update_inputs = {"question": case.question}
                    update_outputs = {
                        "expected_answer": case.expected_answer,
                        "reference_context": case.reference_context,
                        "needs_human_verification": case.needs_human_verification,
                        "metadata": case.metadata,
                    }
                    update_metadata = {
                        "case_id": case.id,
                        "created_at": case.created_at,
                        **case.metadata,
                    }
                
                client.update_example(
                    example_id=example_id,
                    inputs=update_inputs,
                    outputs=update_outputs,
                    metadata=update_metadata,
                )
                updated += 1
            except Exception as e:
                logger.warning("Failed to update example %s: %s", example_id, e)
                upload_errors.append({"example_id": str(example_id), "error": str(e)})

    result = {
        "uploaded": uploaded,
        "updated": updated,
        "skipped_duplicates": skipped_duplicates,
        "validation_errors": validation_errors,
        "sanitization_errors": sanitization_errors,
        "upload_errors": upload_errors,
        "total_cases": len(cases),
        "new_cases_uploaded": len(new_cases),
        "existing_cases_updated": len(update_cases),
    }
    return result


# =============================================================================
# Dry-run and Validation (re-exported for CLI)
# =============================================================================

def validate_dataset_before_upload(
    filepath: str,
    exclude_unverified: bool = True,
) -> Dict[str, Any]:
    """Validate dataset before uploading to LangSmith."""
    from app.ai.evaluation.validation import validate_dataset
    from app.ai.evaluation.dataset_schema import load_dataset, get_dataset_stats

    cases = load_dataset(filepath)
    stats = get_dataset_stats(cases)
    validation_report = validate_dataset(filepath)
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
    """Simulate dataset upload without making external API calls."""
    from app.ai.evaluation.dataset_schema import load_dataset, get_dataset_stats
    from app.ai.evaluation.validation import validate_dataset

    cases = load_dataset(filepath)

    if exclude_unverified:
        cases = [c for c in cases if not c.needs_human_verification]

    validation_report = validate_dataset(filepath)
    stats = get_dataset_stats(cases)

    examples = []
    for case in cases[:10]:
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
        "examples_preview": examples,
        "categories": stats["categories"],
        "languages": stats["languages"],
    }


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Evaluator wrappers
    "ProgrammaticEvaluatorWrapper",
    "LLMJudgeWrapper",
    # Target
    "RAGChainTarget",
    # Experiment
    "NativeExperimentConfig",
    "NativeExperimentResult",
    "NativeLangSmithExperimentRunner",
    # Dataset
    "upload_dataset_deduplicated",
    "validate_dataset_before_upload",
    "dry_run_dataset_upload",
    # Constants
    "DEFAULT_DATASET_NAME",
]