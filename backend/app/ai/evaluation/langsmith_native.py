"""Native LangSmith Experiment Integration for MediTrust RAG Evaluation (Phase 6.1).

This module provides native LangSmith experiment integration using the official
`langsmith.evaluation.evaluate()` API. It wraps existing Phase 4 programmatic
evaluators and Phase 5 LLM judges as LangSmith-compatible evaluators.
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
from app.ai.evaluation.dataset_schema import (
    EvaluationCase,
    load_dataset,
    validate_case,
    get_dataset_stats,
)
from app.ai.evaluation.validation import validate_dataset
from app.ai.evaluation.pipeline import (
    EvaluationResult as PipelineEvaluationResult,
    PipelineSummary,
    OfflineEvaluationPipeline,
    run_offline_evaluation,
)
from app.ai.rag.chain import RAGChain, RAGResponse

logger = logging.getLogger("meditrust.ai.evaluation.langsmith_native")


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

            return LangSmithEvaluationResult(
                key=self._evaluator_name,
                score=score,
                value=value,
                comment=result.reason,
                metadata={
                    "issues": result.issues,
                    "evaluator_status": result.status.value,
                    "passed": result.passed,
                    **result.metadata,
                }
            )
        except Exception as e:
            logger.exception("Programmatic evaluator %s failed: %s", self._evaluator_name, e)
            return LangSmithEvaluationResult(
                key=self._evaluator_name,
                score=None,
                value="error",
                comment=f"Evaluator error: {str(e)}",
                metadata={"error": str(e)}
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

            return LangSmithEvaluationResult(
                key=self._judge_name,
                score=score,
                value=status_value,
                comment=result.reasoning,
                metadata={
                    "evidence": result.evidence,
                    "issues": result.issues,
                    "confidence": result.confidence,
                    "judge_model": result.judge_model,
                    "judge_provider": result.judge_provider,
                }
            )
        except Exception as e:
            logger.exception("LLM judge %s failed: %s", self._judge_name, e)
            return LangSmithEvaluationResult(
                key=self._judge_name,
                score=None,
                value="error",
                comment=f"Judge error: {str(e)}",
                metadata={"error": str(e)}
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
        """LangSmith target interface - takes inputs dict, returns outputs dict."""
        question = inputs.get("question", "")
        conversation_history = inputs.get("conversation_history")
        top_k = inputs.get("top_k")

        try:
            response = await self.rag_chain.generate(
                query=question,
                conversation_history=conversation_history,
                top_k=top_k,
            )

            return {
                "answer": response.answer,
                "sources": response.sources,
                "has_context": response.has_context,
                "model": response.model,
                "doctor_recommendations": response.doctor_recommendations or [],
                "disclaimer": response.disclaimer or "",
                "langsmith_run_id": response.langsmith_run_id,
                "langsmith_trace_id": response.langsmith_trace_id,
            }
        except Exception as e:
            logger.exception("RAG chain generation failed: %s", e)
            return {
                "answer": "",
                "sources": [],
                "has_context": False,
                "model": "",
                "doctor_recommendations": [],
                "disclaimer": "",
                "error_message": str(e),
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
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    evaluator_summary: Dict[str, Any] = field(default_factory=dict)
    langsmith_experiment_url: Optional[str] = None
    run_ids: List[str] = field(default_factory=list)
    dataset_id: Optional[str] = None
    feedback_ids: List[str] = field(default_factory=list)

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
        self.client = client or create_langsmith_client(api_key)
        self._target: Optional[RAGChainTarget] = None
        self._programmatic_evaluators: Dict[str, ProgrammaticEvaluatorWrapper] = {}
        self._llm_judges: Dict[str, LLMJudgeWrapper] = {}

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
        """Run the native LangSmith experiment using the official evaluate() API."""
        logger.info(
            "Starting native LangSmith experiment '%s' on dataset '%s'",
            self.config.experiment_name, self.config.dataset_name
        )

        start_time = datetime.utcnow()

        target = self.target
        evaluators = self._get_evaluator_list()

        logger.info(
            "Running experiment with %d evaluators: %s",
            len(evaluators), [e.name for e in evaluators]
        )

        try:
            start_time = datetime.utcnow()

            experiment_results = await aevaluate(
                target=target,
                data=self.config.dataset_name,
                evaluators=evaluators,
                evaluator_config={
                    "max_concurrency": self.config.max_concurrent,
                },
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

            async for row in experiment_results:
                if row.get("run"):
                    run_ids.append(str(row["run"].id))

            logger.info(
                "Native LangSmith experiment completed: experiment_id=%s, runs=%d",
                experiment_id, len(run_ids)
            )

            example_results = await self._evaluate_examples_locally()

            programmatic_results = await self._run_programmatic_on_examples(example_results)
            llm_judge_results = await self._run_llm_judges_on_examples(example_results)

            combined_results = {}
            all_evaluator_names = set()
            for example_id, eval_results in programmatic_results.items():
                combined_results[example_id] = {}
                for eval_name, eval_result in eval_results.items():
                    combined_results[example_id][eval_name] = eval_result
                    all_evaluator_names.add(eval_name)

            for example_id, eval_results in llm_judge_results.items():
                if example_id not in combined_results:
                    combined_results[example_id] = {}
                for eval_name, eval_result in eval_results.items():
                    combined_results[example_id][eval_name] = eval_result
                    all_evaluator_names.add(eval_name)

            total = len(example_results)
            successful = sum(1 for r in example_results if r.get("execution_status") == "success")
            failed = sum(1 for r in example_results if r.get("execution_status") == "failed")
            errors = sum(1 for r in example_results if r.get("execution_status") == "error")

            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            evaluator_summary = self._compute_evaluator_summary(
                example_results, programmatic_results, llm_judge_results
            )

            summary = NativeExperimentResult(
                experiment_name=self.config.experiment_name,
                experiment_id=experiment_id,
                dataset_name=self.config.dataset_name,
                total_examples=total,
                successful=successful,
                failed=failed,
                errors=errors,
                execution_time_ms=execution_time_ms,
                evaluator_summary=evaluator_summary,
                langsmith_experiment_url=experiment_url,
                run_ids=run_ids,
                dataset_id=dataset_id,
                feedback_ids=feedback_ids,
            )

            detailed_results = []
            for i, ex_result in enumerate(example_results):
                example_id = str(ex_result.get("example_id", f"unknown-{i}"))
                detailed_results.append({
                    "example_id": example_id,
                    "question": ex_result.get("question", ""),
                    "expected_answer": ex_result.get("expected_answer", ""),
                    "actual_answer": ex_result.get("actual_answer", ""),
                    "reference_context": ex_result.get("reference_context", ""),
                    "retrieved_context": ex_result.get("retrieved_context", ""),
                    "category": ex_result.get("category", "general_info"),
                    "execution_status": ex_result.get("execution_status", "error"),
                    "evaluator_results": combined_results.get(example_id, {}),
                    "execution_time_ms": ex_result.get("execution_time_ms", 0.0),
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_message": ex_result.get("error_message"),
                })

            return detailed_results, summary

        except Exception as e:
            logger.exception("Native experiment failed: %s", e)
            raise

    async def _evaluate_examples_locally(self) -> List[Dict[str, Any]]:
        """Evaluate examples locally to get RAG outputs for programmatic evaluation."""
        examples = list(self.client.list_examples(
            dataset_name=self.config.dataset_name,
            limit=self.config.max_examples,
        ))

        logger.info("Fetched %d examples from dataset '%s'",
                   len(examples), self.config.dataset_name)

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def run_with_semaphore(example: Example) -> Dict[str, Any]:
            async with semaphore:
                return await self._evaluate_single_example(example)

        tasks = [run_with_semaphore(ex) for ex in examples]
        example_results = await asyncio.gather(*tasks, return_exceptions=False)

        return example_results

    async def _evaluate_single_example(self, example: Example) -> Dict[str, Any]:
        """Evaluate a single example through the RAG pipeline and return results."""
        start_time = datetime.utcnow()

        try:
            pipeline_result = await self._run_rag_generation(example)

            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                "example_id": str(example.id),
                "question": example.inputs.get("question", ""),
                "expected_answer": example.outputs.get("expected_answer", ""),
                "actual_answer": pipeline_result.get("actual_answer", ""),
                "reference_context": example.outputs.get("reference_context", ""),
                "retrieved_context": pipeline_result.get("retrieved_context", ""),
                "category": example.inputs.get("category", "general_info"),
                "language": example.inputs.get("language", "en"),
                "tags": example.inputs.get("tags", []),
                "execution_status": pipeline_result.get("execution_status", "success"),
                "error_message": pipeline_result.get("error_message"),
                "sources": pipeline_result.get("sources", []),
                "has_context": pipeline_result.get("has_context", False),
                "model": pipeline_result.get("model", ""),
                "doctor_recommendations": pipeline_result.get("doctor_recommendations", []),
                "disclaimer": pipeline_result.get("disclaimer", ""),
                "execution_time_ms": execution_time_ms,
            }

        except Exception as e:
            logger.exception("Error evaluating example %s: %s", example.id, e)
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "example_id": str(example.id),
                "question": example.inputs.get("question", ""),
                "expected_answer": example.outputs.get("expected_answer", ""),
                "actual_answer": "",
                "reference_context": example.outputs.get("reference_context", ""),
                "retrieved_context": "",
                "category": example.inputs.get("category", "general_info"),
                "language": example.inputs.get("language", "en"),
                "tags": example.inputs.get("tags", []),
                "execution_status": "error",
                "error_message": str(e),
                "sources": [],
                "has_context": False,
                "model": "",
                "doctor_recommendations": [],
                "disclaimer": "",
                "execution_time_ms": execution_time_ms,
            }

    async def _run_rag_generation(self, example: Example) -> Dict[str, Any]:
        """Run RAG generation for a single example."""
        try:
            rag_response = await self.rag_chain.generate(
                query=example.inputs.get("question", ""),
                top_k=self.config.top_k if hasattr(self.config, 'top_k') else None,
            )

            retrieved_context = self._extract_retrieved_context(rag_response)

            if not rag_response.answer or not rag_response.answer.strip():
                execution_status = "failed"
                error_message = "Empty response from RAG chain"
            else:
                execution_status = "success"
                error_message = None

            return {
                "actual_answer": rag_response.answer,
                "retrieved_context": retrieved_context,
                "execution_status": execution_status,
                "error_message": error_message,
                "sources": rag_response.sources,
                "has_context": rag_response.has_context,
                "model": rag_response.model,
                "doctor_recommendations": rag_response.doctor_recommendations or [],
                "disclaimer": rag_response.disclaimer or "",
                "execution_time_ms": 0.0,
            }

        except Exception as e:
            logger.exception("RAG generation failed for example: %s", e)
            return {
                "actual_answer": "",
                "retrieved_context": "",
                "execution_status": "error",
                "error_message": str(e),
                "sources": [],
                "has_context": False,
                "model": "",
                "doctor_recommendations": [],
                "disclaimer": "",
                "execution_time_ms": 0.0,
            }

    def _extract_retrieved_context(self, rag_response: RAGResponse) -> str:
        """Extract retrieved context from RAG response sources."""
        if not rag_response.sources:
            return ""

        context_parts = []
        for source in rag_response.sources:
            preview = source.get("content_preview", "")
            if preview:
                context_parts.append(preview)

        return "\n\n".join(context_parts)

    async def _run_programmatic_on_examples(self, example_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict]]:
        """Run programmatic evaluators on all example results.

        Returns:
            Dict mapping example_id -> evaluator_name -> evaluator_result_dict
        """
        results: Dict[str, Dict[str, Dict]] = {}

        evaluator_names = self.config.evaluator_names or []
        if not evaluator_names and self.config.include_programmatic:
            evaluator_names = list(PROGRAMMATIC_DEFAULT_EVALUATORS)

        for name in evaluator_names:
            if name not in PROGRAMMATIC_EVALUATORS:
                continue

            evaluator_func = get_programmatic_evaluator(name)

            for ex_result in example_results:
                example_id = ex_result.get("example_id", "unknown")

                pipeline_result = PipelineEvaluationResult(
                    evaluation_id=example_id,
                    question=ex_result.get("question", ""),
                    expected_answer=ex_result.get("expected_answer", ""),
                    actual_answer=ex_result.get("actual_answer", ""),
                    reference_context=ex_result.get("reference_context", ""),
                    retrieved_context=ex_result.get("retrieved_context", ""),
                    category=ex_result.get("category", "general_info"),
                    language=ex_result.get("language", "en"),
                    tags=ex_result.get("tags", []),
                    execution_status=ex_result.get("execution_status", "success"),
                    error_message=ex_result.get("error_message"),
                    sources=ex_result.get("sources", []),
                    has_context=ex_result.get("has_context", False),
                    model=ex_result.get("model", ""),
                    doctor_recommendations=ex_result.get("doctor_recommendations", []),
                    disclaimer=ex_result.get("disclaimer", ""),
                    execution_time_ms=ex_result.get("execution_time_ms", 0.0),
                )

                try:
                    result = evaluator_func(pipeline_result)

                    score = result.score
                    if score is None:
                        if result.status == EvaluationStatus.PASSED:
                            score = 1.0
                        elif result.status == EvaluationStatus.FAILED:
                            score = 0.0
                        else:
                            score = None

                    eval_result = {
                        "score": score,
                        "passed": result.passed,
                        "status": result.status.value,
                        "reason": result.reason,
                        "issues": result.issues,
                        "metadata": result.metadata,
                    }

                    if example_id not in results:
                        results[example_id] = {}
                    results[example_id][name] = eval_result

                except Exception as e:
                    logger.exception("Evaluator %s failed for case: %s", name, e)
                    eval_result = {
                        "score": None,
                        "passed": None,
                        "status": "error",
                        "reason": f"Evaluator error: {str(e)}",
                        "issues": [str(e)],
                        "metadata": {"error": str(e)},
                    }
                    if example_id not in results:
                        results[example_id] = {}
                    results[example_id][name] = eval_result

        return results

    async def _run_llm_judges_on_examples(self, example_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict]]:
        """Run LLM judges on all example results.

        Returns:
            Dict mapping example_id -> evaluator_name -> evaluator_result_dict
        """
        results: Dict[str, Dict[str, Dict]] = {}

        evaluator_names = self.config.evaluator_names or []
        if not evaluator_names and self.config.include_llm_judges:
            evaluator_names = list(LLM_DEFAULT_EVALUATORS)

        for name in evaluator_names:
            if name not in LLM_JUDGES:
                continue

            judge_class = LLM_JUDGES[name]
            judge = judge_class()

            for ex_result in example_results:
                example_id = ex_result.get("example_id", "unknown")

                pipeline_result = PipelineEvaluationResult(
                    evaluation_id=example_id,
                    question=ex_result.get("question", ""),
                    expected_answer=ex_result.get("expected_answer", ""),
                    actual_answer=ex_result.get("actual_answer", ""),
                    reference_context=ex_result.get("reference_context", ""),
                    retrieved_context=ex_result.get("retrieved_context", ""),
                    category=ex_result.get("category", "general_info"),
                    language=ex_result.get("language", "en"),
                    tags=ex_result.get("tags", []),
                    execution_status=ex_result.get("execution_status", "success"),
                    error_message=ex_result.get("error_message"),
                    sources=ex_result.get("sources", []),
                    has_context=ex_result.get("has_context", False),
                    model=ex_result.get("model", ""),
                    doctor_recommendations=ex_result.get("doctor_recommendations", []),
                    disclaimer=ex_result.get("disclaimer", ""),
                    execution_time_ms=ex_result.get("execution_time_ms", 0.0),
                )

                try:
                    result = await judge.evaluate(pipeline_result)

                    score = result.score
                    if score is None:
                        if result.status == JudgeStatus.PASSED:
                            score = 1.0
                        elif result.status == JudgeStatus.FAILED:
                            score = 0.0
                        else:
                            score = None

                    eval_result = {
                        "score": score,
                        "passed": result.passed,
                        "status": result.status.value,
                        "reasoning": result.reasoning,
                        "evidence": result.evidence,
                        "issues": result.issues,
                        "confidence": result.confidence,
                        "judge_model": result.judge_model,
                        "judge_provider": result.judge_provider,
                        "metadata": result.metadata,
                    }

                    if example_id not in results:
                        results[example_id] = {}
                    results[example_id][name] = eval_result

                except Exception as e:
                    logger.exception("LLM judge %s failed: %s", name, e)
                    eval_result = {
                        "score": None,
                        "passed": None,
                        "status": "error",
                        "reasoning": f"Judge error: {str(e)}",
                        "evidence": [],
                        "issues": [str(e)],
                        "confidence": 0.0,
                        "judge_model": "",
                        "judge_provider": "",
                        "metadata": {"error": str(e)},
                    }
                    if example_id not in results:
                        results[example_id] = {}
                    results[example_id][name] = eval_result

        return results

    def _compute_evaluator_summary(
        self,
        example_results: List[Dict[str, Any]],
        programmatic_results: Dict[str, Dict[str, Dict]],
        llm_judge_results: Dict[str, Dict[str, Dict]],
    ) -> Dict[str, Any]:
        """Compute summary statistics for all evaluators."""
        summary = {}

        all_evaluators = {}
        for name, results in programmatic_results.items():
            all_evaluators[name] = results
        for name, results in llm_judge_results.items():
            all_evaluators[name] = results

        for name, example_results_dict in all_evaluators.items():
            results_list = list(example_results_dict.values())
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
                "passed": passed,
                "failed": failed,
                "needs_human_review": needs_review,
                "insufficient_context": insufficient,
                "errors": errors,
                "average_score": round(avg_score, 3) if avg_score is not None else None,
                "average_confidence": round(avg_confidence, 3) if avg_confidence is not None else None,
            }

        return summary


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
) -> Dict[str, Any]:
    """Upload evaluation dataset with deduplication by case_id.

    Uses case_id from the source JSONL as the stable identifier.
    Checks for existing examples with the same case_id before uploading.

    Args:
        filepath: Path to JSONL dataset file
        dataset_name: Target LangSmith dataset name
        api_key: Optional LangSmith API key
        description: Dataset description
        create_if_missing: Create dataset if it doesn't exist
        exclude_unverified: Exclude cases requiring human verification
        update_existing: If True, update existing examples; if False, skip them

    Returns:
        Upload results with deduplication stats
    """
    from app.ai.evaluation.dataset_schema import load_dataset
    from app.ai.evaluation.validation import validate_dataset

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

    all_cases_to_upload = new_cases + [c for c, _ in update_cases]

    if all_cases_to_upload:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for case in new_cases:
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
            uploaded = result.get("uploaded", 0)
            validation_errors = result.get("validation_errors", [])
            upload_errors = result.get("upload_errors", [])
        finally:
            os.unlink(temp_path)

        if update_cases:
            for case, example_id in update_cases:
                try:
                    client.update_example(
                        example_id=example_id,
                        inputs={"question": case.question},
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
                    updated += 1
                except Exception as e:
                    logger.warning("Failed to update example %s: %s", example_id, e)
                    upload_errors.append({"example_id": str(example_id), "error": str(e)})

    result = {
        "uploaded": uploaded,
        "updated": updated,
        "skipped_duplicates": skipped_duplicates,
        "validation_errors": validation_errors,
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