"""MediTrust AI Evaluation Package.

Phase 1: Offline RAG Evaluation Foundation
- Dataset schema and validation
- Initial evaluation dataset (20-30 cases)
- Modular evaluators: correctness, faithfulness, answer_relevance, medical_safety
- LangSmith dataset upload utility (manual invocation only)

Phase 3: Offline Evaluation Pipeline
- Executes evaluation cases against RAGChain
- Captures generated answers and retrieved context
- Saves structured results to JSON

Phase 4: Programmatic Evaluators
- Deterministic Python rule-based evaluators
- Common interface with standardized results
- Evaluator runner for batch processing

Phase 5: LLM-as-Judge Evaluators
- Semantic evaluation using LLM judges
- Complementary to programmatic evaluators
- Common interface with standardized results
- Judge runner for batch processing

Phase 6: LangSmith Integration
- Dataset management and validation
- Experiment execution with RAG pipeline
- Evaluator integration (programmatic + LLM judges)
- Result export and reporting

Phase 6.1: Native LangSmith Experiment Integration
- Native LangSmith experiment execution via evaluate() API
- Dataset deduplication with case_id tracking
- Evaluator feedback attached to LangSmith runs
- Real LangSmith experiment runs with feedback
"""
from app.ai.evaluation.dataset_schema import (
    EvaluationCase,
    VALID_CATEGORIES,
    VALID_LANGUAGES,
    validate_case,
    load_dataset,
    save_dataset,
    get_dataset_stats,
)

from app.ai.evaluation.evaluators import (
    EvaluationResult,
    correctness_evaluator,
    faithfulness_evaluator,
    answer_relevance_evaluator,
    medical_safety_evaluator,
    citation_accuracy_evaluator,
    EVALUATORS,
    DEFAULT_EVALUATORS,
    get_evaluator,
    get_default_evaluators,
    run_all_evaluators,
)

from app.ai.evaluation.langsmith_utils import (
    LangSmithUploadError,
    create_langsmith_client,
    dataset_exists,
    create_dataset,
    upload_evaluation_cases,
    upload_dataset_from_file,
    list_datasets,
    delete_dataset,
    get_dataset_examples,
    DEFAULT_DATASET_NAME,
)

from app.ai.evaluation.pipeline import (
    EvaluationResult as PipelineEvaluationResult,
    PipelineSummary,
    OfflineEvaluationPipeline,
    run_offline_evaluation,
)

from app.ai.evaluation.programmatic_evaluators import (
    EvaluatorResult,
    EvaluationStatus,
    evaluate_correctness,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_medical_safety,
    PROGRAMMATIC_EVALUATORS,
    DEFAULT_EVALUATORS as PROGRAMMATIC_DEFAULT_EVALUATORS,
    get_evaluator as get_programmatic_evaluator,
    ScoredEvaluationResult,
    EvaluatorRunner,
    run_programmatic_evaluation,
)

from app.ai.evaluation.llm_judges import (
    LLMJudgeResult,
    JudgeStatus,
    BaseLLMJudge,
    CorrectnessJudge,
    FaithfulnessJudge,
    RelevanceJudge,
    MedicalSafetyJudge,
    LLM_JUDGES,
    LLM_DEFAULT_EVALUATORS,
    get_llm_judge,
    LLMScoredEvaluationResult,
    LLMJudgeRunner,
    run_llm_evaluation,
)

from app.ai.evaluation.langsmith_experiment import (
    ExperimentConfig,
    LangSmithExperimentRunner,
    ExperimentResult,
    ExperimentSummary,
    validate_dataset_before_upload,
    dry_run_dataset_upload,
    upload_evaluation_dataset,
    export_experiment_results,
)

from app.ai.evaluation.langsmith_native import (
    NativeExperimentConfig,
    NativeExperimentResult,
    NativeLangSmithExperimentRunner,
    ProgrammaticEvaluatorWrapper,
    LLMJudgeWrapper,
    RAGChainTarget,
    upload_dataset_deduplicated,
    validate_dataset_before_upload as native_validate_dataset_before_upload,
    dry_run_dataset_upload as native_dry_run_dataset_upload,
    DEFAULT_DATASET_NAME,
)

__all__ = [
    # Dataset schema
    "EvaluationCase",
    "VALID_CATEGORIES",
    "VALID_LANGUAGES",
    "validate_case",
    "load_dataset",
    "save_dataset",
    "get_dataset_stats",
    # Evaluators (Phase 1 - LangSmith compatible)
    "EvaluationResult",
    "correctness_evaluator",
    "faithfulness_evaluator",
    "answer_relevance_evaluator",
    "medical_safety_evaluator",
    "citation_accuracy_evaluator",
    "EVALUATORS",
    "DEFAULT_EVALUATORS",
    "get_evaluator",
    "get_default_evaluators",
    "run_all_evaluators",
    # LangSmith utils
    "LangSmithUploadError",
    "create_langsmith_client",
    "dataset_exists",
    "create_dataset",
    "upload_evaluation_cases",
    "upload_dataset_from_file",
    "list_datasets",
    "delete_dataset",
    "get_dataset_examples",
    "DEFAULT_DATASET_NAME",
    # Pipeline (Phase 3)
    "PipelineEvaluationResult",
    "PipelineSummary",
    "OfflineEvaluationPipeline",
    "run_offline_evaluation",
    # Programmatic Evaluators (Phase 4)
    "EvaluatorResult",
    "EvaluationStatus",
    "evaluate_correctness",
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "evaluate_medical_safety",
    "PROGRAMMATIC_EVALUATORS",
    "PROGRAMMATIC_DEFAULT_EVALUATORS",
    "get_programmatic_evaluator",
    "ScoredEvaluationResult",
    "EvaluatorRunner",
    "run_programmatic_evaluation",
    # LLM Judges (Phase 5)
    "LLMJudgeResult",
    "JudgeStatus",
    "BaseLLMJudge",
    "CorrectnessJudge",
    "FaithfulnessJudge",
    "RelevanceJudge",
    "MedicalSafetyJudge",
    "LLM_JUDGES",
    "LLM_DEFAULT_EVALUATORS",
    "get_llm_judge",
    "LLMScoredEvaluationResult",
    "LLMJudgeRunner",
    "run_llm_evaluation",
    # LangSmith Experiment (Phase 6)
    "ExperimentConfig",
    "LangSmithExperimentRunner",
    "ExperimentResult",
    "ExperimentSummary",
    "validate_dataset_before_upload",
    "dry_run_dataset_upload",
    "upload_evaluation_dataset",
    "export_experiment_results",
    # Native LangSmith Experiment (Phase 6.1)
    "NativeExperimentConfig",
    "NativeExperimentResult",
    "NativeLangSmithExperimentRunner",
    "ProgrammaticEvaluatorWrapper",
    "LLMJudgeWrapper",
    "RAGChainTarget",
    "upload_dataset_deduplicated",
    "native_validate_dataset_before_upload",
    "native_dry_run_dataset_upload",
]