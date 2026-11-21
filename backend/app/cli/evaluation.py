"""CLI commands for MediTrust evaluation dataset management.

Usage:
    python -m app.cli.evaluation upload --file dataset.jsonl
    python -m app.cli.evaluation validate --file dataset.jsonl
    python -m app.cli.evaluation validate-full --file dataset.jsonl
    python -m app.cli.evaluation stats --file dataset.jsonl
    python -m app.cli.evaluation run --file dataset.jsonl --output results.json
    python -m app.cli.evaluation score --input results.json --output scored_results.json
    python -m app.cli.evaluation llm-eval --input results.json --output llm_scored_results.json
    python -m app.cli.evaluation validate-dataset --file dataset.jsonl
    python -m app.cli.evaluation dry-run --file dataset.jsonl
    python -m app.cli.evaluation upload-dataset --file dataset.jsonl
    python -m app.cli.evaluation experiment --name exp1 --dataset dataset_name
    python -m app.cli.evaluation export-experiment --name exp1 --output results.json
"""
import argparse
import asyncio
import sys
import json
import os
from pathlib import Path

from app.ai.evaluation.dataset_schema import load_dataset, get_dataset_stats, VALID_CATEGORIES
from app.ai.evaluation.validation import validate_dataset, save_validation_report
from app.ai.evaluation.langsmith_utils import (
    upload_dataset_from_file,
    LangSmithUploadError,
    list_datasets,
    create_langsmith_client,
)
from app.ai.evaluation.langsmith_experiment import (
    ExperimentConfig,
    LangSmithExperimentRunner,
    run_llm_evaluation,
    validate_dataset_before_upload,
    dry_run_dataset_upload,
    upload_evaluation_dataset,
    export_experiment_results,
    LLM_JUDGES,
    LLM_DEFAULT_EVALUATORS,
    PROGRAMMATIC_EVALUATORS,
    PROGRAMMATIC_DEFAULT_EVALUATORS,
)
from app.ai.evaluation.langsmith_native import (
    upload_dataset_deduplicated,
    validate_dataset_before_upload as validate_dataset_dedup,
    dry_run_dataset_upload as dry_run_dataset_upload_native,
    upload_dataset_deduplicated as upload_dataset_deduplicated_native,
    NativeLangSmithExperimentRunner,
    NativeExperimentConfig,
    NativeExperimentResult,
    LLM_JUDGES,
    LLM_DEFAULT_EVALUATORS,
    PROGRAMMATIC_EVALUATORS,
    PROGRAMMATIC_DEFAULT_EVALUATORS,
)
from app.ai.evaluation.pipeline import (
    OfflineEvaluationPipeline,
    run_offline_evaluation,
    PipelineSummary,
    EvaluationResult as PipelineEvaluationResult,
)
from app.ai.evaluation.programmatic_evaluators import (
    EvaluatorRunner,
    run_programmatic_evaluation,
    PROGRAMMATIC_EVALUATORS,
    DEFAULT_EVALUATORS as PROGRAMMATIC_DEFAULT_EVALUATORS,
)
from app.ai.evaluation.llm_judges import (
    LLMJudgeRunner,
    run_llm_evaluation,
    LLM_JUDGES,
    LLM_DEFAULT_EVALUATORS,
)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate dataset file (basic schema validation)."""
    try:
        cases = load_dataset(args.file)
        print(f"[OK] Validated {len(cases)} cases from {args.file}")
        stats = get_dataset_stats(cases)
        print(f"  Categories: {stats['categories']}")
        print(f"  Languages: {stats['languages']}")
        print(f"  Needs human verification: {stats['needs_human_verification']}")
        return 0
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        return 1


def cmd_validate_full(args: argparse.Namespace) -> int:
    """Run comprehensive dataset validation with detailed report."""
    try:
        report = validate_dataset(args.file)

        if args.output:
            save_validation_report(report, args.output)
            print(f"[OK] Validation report saved to {args.output}")

        report.print_summary()

        # Return non-zero if validation failed
        if not report.is_valid or report.invalid_cases > 0 or len(report.duplicate_ids) > 0:
            return 1
        return 0

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Show dataset statistics."""
    try:
        cases = load_dataset(args.file)
        stats = get_dataset_stats(cases)
        print(f"Dataset: {args.file}")
        print(f"Total cases: {stats['total']}")
        print(f"Categories: {stats['categories']}")
        print(f"Languages: {stats['languages']}")
        print(f"Needs human verification: {stats['needs_human_verification']}")
        return 0
    except Exception as e:
        print(f"[ERROR] Stats failed: {e}", file=sys.stderr)
        return 1


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload dataset to LangSmith."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        result = upload_dataset_from_file(
            filepath=args.file,
            dataset_name=args.dataset,
            api_key=args.api_key,
            description=args.description,
            create_if_missing=not args.no_create,
        )

        print(f"[OK] Upload complete to dataset '{args.dataset}'")
        print(f"  Uploaded: {result['uploaded']}")
        print(f"  Skipped (invalid): {result['skipped']}")
        if result['validation_errors']:
            print(f"  Validation errors: {len(result['validation_errors'])}")
        if result['upload_errors']:
            print(f"  Upload errors: {len(result['upload_errors'])}")

        stats = result.get('dataset_stats', {})
        print(f"  Categories: {stats.get('categories', {})}")
        print(f"  Languages: {stats.get('languages', {})}")

        return 0

    except LangSmithUploadError as e:
        print(f"[ERROR] Upload failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List LangSmith datasets."""
    try:
        client = create_langsmith_client(args.api_key)
        datasets = list_datasets(client)

        if not datasets:
            print("No datasets found.")
            return 0

        print(f"{'Name':<40} {'ID':<36} {'Examples':<10} {'Created'}")
        print("-" * 100)
        for ds in datasets:
            created = ds['created_at'][:10] if ds['created_at'] else 'N/A'
            print(f"{ds['name']:<40} {ds['id']:<36} {ds['example_count']:<10} {created}")

        return 0
    except Exception as e:
        print(f"[ERROR] List failed: {e}", file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run offline evaluation pipeline against RAGChain."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        # Validate category filter
        if args.category:
            invalid_cats = set(args.category) - VALID_CATEGORIES
            if invalid_cats:
                print(f"[ERROR] Invalid categories: {invalid_cats}. Valid: {sorted(VALID_CATEGORIES)}", file=sys.stderr)
                return 1

        print(f"[INFO] Starting offline evaluation...")
        print(f"  Dataset: {args.file}")
        print(f"  Output: {args.output}")
        print(f"  Include unverified: {args.include_unverified}")
        print(f"  Category filter: {args.category or 'none'}")
        print(f"  Case ID filter: {args.case_id or 'none'}")
        print(f"  Top-K: {args.top_k or 'default'}")
        print(f"  Max concurrent: {args.max_concurrent}")

        # Run the evaluation pipeline
        results, summary = asyncio.run(run_offline_evaluation(
            dataset_path=args.file,
            output_path=args.output,
            include_unverified=args.include_unverified,
            category_filter=args.category,
            case_id_filter=args.case_id,
            top_k=args.top_k,
            max_concurrent=args.max_concurrent,
        ))

        # Print summary
        print("\n" + "=" * 60)
        print("OFFLINE EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total cases:      {summary.total_cases}")
        print(f"Successful:       {summary.successful}")
        print(f"Failed:           {summary.failed}")
        print(f"Skipped:          {summary.skipped}")
        print(f"Unverified:       {summary.unverified}")
        print(f"Total time:       {summary.execution_time_ms:.0f} ms")
        print(f"Results saved to: {args.output}")
        print("=" * 60)

        # Return non-zero if any failures
        if summary.failed > 0:
            print(f"[WARNING] {summary.failed} case(s) failed", file=sys.stderr)
            return 1

        return 0

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        return 1


def cmd_score(args: argparse.Namespace) -> int:
    """Run programmatic evaluators on Phase 3 evaluation results."""
    try:
        if not Path(args.input).exists():
            print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
            return 1

        # Validate evaluator names
        if args.evaluator:
            invalid_evals = set(args.evaluator) - set(PROGRAMMATIC_EVALUATORS.keys())
            if invalid_evals:
                print(f"[ERROR] Invalid evaluators: {invalid_evals}. Valid: {sorted(PROGRAMMATIC_EVALUATORS.keys())}", file=sys.stderr)
                return 1

        evaluator_names = args.evaluator or PROGRAMMATIC_DEFAULT_EVALUATORS

        print(f"[INFO] Starting programmatic evaluation...")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Evaluators: {evaluator_names}")
        print(f"  Correctness threshold: {args.correctness_threshold}")
        print(f"  Faithfulness threshold: {args.faithfulness_threshold}")
        print(f"  Relevance threshold: {args.relevance_threshold}")

        # Run programmatic evaluation
        scored_results, summary = run_programmatic_evaluation(
            input_path=args.input,
            output_path=args.output,
            evaluator_names=evaluator_names,
            correctness_threshold=args.correctness_threshold,
            faithfulness_threshold=args.faithfulness_threshold,
            relevance_threshold=args.relevance_threshold,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("PROGRAMMATIC EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total cases scored: {summary.get('total', 0)}")
        for eval_name, eval_summary in summary.get("evaluators", {}).items():
            print(f"\n  {eval_name}:")
            print(f"    Passed:              {eval_summary.get('passed', 0)}")
            print(f"    Failed:              {eval_summary.get('failed', 0)}")
            print(f"    Needs Human Review:  {eval_summary.get('needs_human_review', 0)}")
            print(f"    Insufficient Context: {eval_summary.get('insufficient_context', 0)}")
            print(f"    Errors:              {eval_summary.get('errors', 0)}")
            if eval_summary.get('average_score') is not None:
                print(f"    Average Score:       {eval_summary['average_score']:.3f}")
        print(f"\nResults saved to: {args.output}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"[ERROR] Scoring failed: {e}", file=sys.stderr)
        return 1


def cmd_llm_eval(args: argparse.Namespace) -> int:
    """Run LLM-as-Judge evaluators on Phase 3 evaluation results."""
    try:
        if not Path(args.input).exists():
            print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
            return 1

        # Validate evaluator names
        if args.evaluator:
            invalid_evals = set(args.evaluator) - set(LLM_JUDGES.keys())
            if invalid_evals:
                print(f"[ERROR] Invalid LLM evaluators: {invalid_evals}. Valid: {sorted(LLM_JUDGES.keys())}", file=sys.stderr)
                return 1

        evaluator_names = args.evaluator or LLM_DEFAULT_EVALUATORS

        print(f"[INFO] Starting LLM-as-Judge evaluation...")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Evaluators: {evaluator_names}")
        print(f"  Max concurrent: {args.max_concurrent}")

        # Run LLM evaluation
        scored_results, summary = asyncio.run(run_llm_evaluation(
            input_path=args.input,
            output_path=args.output,
            evaluator_names=evaluator_names,
            max_concurrent=args.max_concurrent,
        ))

        # Print summary
        print("\n" + "=" * 60)
        print("LLM-AS-JUDGE EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total cases scored: {summary.get('total', 0)}")
        for eval_name, eval_summary in summary.get("evaluators", {}).items():
            print(f"\n  {eval_name}:")
            print(f"    Passed:              {eval_summary.get('passed', 0)}")
            print(f"    Failed:              {eval_summary.get('failed', 0)}")
            print(f"    Needs Human Review:  {eval_summary.get('needs_human_review', 0)}")
            print(f"    Insufficient Context: {eval_summary.get('insufficient_context', 0)}")
            print(f"    Errors:              {eval_summary.get('errors', 0)}")
            if eval_summary.get('average_score') is not None:
                print(f"    Average Score:       {eval_summary['average_score']:.3f}")
            if eval_summary.get('average_confidence') is not None:
                print(f"    Average Confidence:  {eval_summary['average_confidence']:.3f}")
        print(f"\nResults saved to: {args.output}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"[ERROR] LLM evaluation failed: {e}", file=sys.stderr)
        return 1


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    """Validate dataset before LangSmith upload."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        print(f"[INFO] Validating dataset: {args.file}")
        result = validate_dataset_before_upload(
            filepath=args.file,
            exclude_unverified=not args.include_unverified,
        )

        print(f"\n[VALIDATION RESULT]")
        print(f"  Valid: {result['valid']}")
        print(f"  Total cases: {result['total_cases']}")
        print(f"  Valid cases: {result['valid_cases']}")
        print(f"  Invalid cases: {result['invalid_cases']}")
        print(f"  Unverified cases: {result['unverified_cases']}")
        print(f"  Categories: {result['categories']}")
        print(f"  Languages: {result['languages']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Validation report saved to {args.output}")

        return 0 if result['valid'] else 1

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        return 1


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Dry-run dataset upload without making external API calls."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        print(f"[INFO] Dry-run dataset upload: {args.file}")
        result = dry_run_dataset_upload(
            filepath=args.file,
            dataset_name=args.dataset,
            exclude_unverified=not args.include_unverified,
        )

        print(f"\n[DRY RUN RESULT]")
        print(f"  Dataset name: {result['dataset_name']}")
        print(f"  Total cases in file: {result['total_cases_in_file']}")
        print(f"  Cases after filtering: {result['cases_after_filtering']}")
        print(f"  Excluded unverified: {result['excluded_unverified']}")
        print(f"  Validation passed: {result['validation_passed']}")
        print(f"\n  Categories: {result['categories']}")
        print(f"  Languages: {result['languages']}")
        print(f"\n  Examples preview (first 10):")
        for ex in result['examples_preview']:
            print(f"    - {ex['id']}: {ex['question']} ({ex['category']}, {ex['language']})")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Dry-run report saved to {args.output}")

        return 0

    except Exception as e:
        print(f"[ERROR] Dry-run failed: {e}", file=sys.stderr)
        return 1


def cmd_upload_dataset(args: argparse.Namespace) -> int:
    """Upload evaluation dataset to LangSmith."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        print(f"[INFO] Uploading dataset to LangSmith...")
        print(f"  File: {args.file}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Include unverified: {args.include_unverified}")

        result = upload_evaluation_dataset(
            filepath=args.file,
            dataset_name=args.dataset,
            api_key=args.api_key,
            description=args.description,
            create_if_missing=not args.no_create,
            exclude_unverified=not args.include_unverified,
        )

        print(f"\n[OK] Upload complete to dataset '{args.dataset}'")
        print(f"  Uploaded: {result['uploaded']}")
        print(f"  Skipped (invalid): {result['skipped']}")
        if result['validation_errors']:
            print(f"  Validation errors: {len(result['validation_errors'])}")
        if result['upload_errors']:
            print(f"  Upload errors: {len(result['upload_errors'])}")

        stats = result.get('dataset_stats', {})
        print(f"  Categories: {stats.get('categories', {})}")
        print(f"  Languages: {stats.get('languages', {})}")

        return 0

    except LangSmithUploadError as e:
        print(f"[ERROR] Upload failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run a LangSmith evaluation experiment."""
    try:
        print(f"[INFO] Starting LangSmith experiment...")
        print(f"  Experiment name: {args.name}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Output: {args.output}")
        print(f"  Max examples: {args.max_examples or 'all'}")
        print(f"  Max concurrent: {args.max_concurrent}")
        print(f"  Include programmatic: {args.include_programmatic}")
        print(f"  Include LLM judges: {args.include_llm_judges}")

        config = ExperimentConfig(
            experiment_name=args.name,
            dataset_name=args.dataset,
            evaluator_names=args.evaluator,
            include_programmatic=args.include_programmatic,
            include_llm_judges=args.include_llm_judges,
            max_concurrent=args.max_concurrent,
            max_examples=args.max_examples,
            metadata={"cli_invoked": True},
        )

        runner = LangSmithExperimentRunner(config=config)
        results, summary = asyncio.run(runner.run_experiment())

        # Save results
        runner.save_results(results, summary, args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("LANGSMITH EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Experiment: {summary.experiment_name}")
        print(f"Dataset: {summary.dataset_name}")
        print(f"Total examples: {summary.total_examples}")
        print(f"Successful: {summary.successful}")
        print(f"Failed: {summary.failed}")
        print(f"Skipped: {summary.skipped}")
        print(f"Total time: {summary.execution_time_ms:.0f} ms")

        print("\nEvaluator Summary:")
        for eval_name, eval_summary in summary.evaluator_summary.items():
            print(f"\n  {eval_name}:")
            print(f"    Passed: {eval_summary.get('passed', 0)}")
            print(f"    Failed: {eval_summary.get('failed', 0)}")
            print(f"    Needs Human Review: {eval_summary.get('needs_human_review', 0)}")
            print(f"    Insufficient Context: {eval_summary.get('insufficient_context', 0)}")
            print(f"    Errors: {eval_summary.get('errors', 0)}")
            if eval_summary.get('average_score') is not None:
                print(f"  Average Score: {eval_summary['average_score']:.3f}")
            if eval_summary.get('average_confidence') is not None:
                print(f"  Average Confidence: {eval_summary['average_confidence']:.3f}")

        print(f"\nResults saved to: {args.output}")
        print("=" * 60)

        return 0 if summary.failed == 0 else 1

    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}", file=sys.stderr)
        return 1


def cmd_export_experiment(args: argparse.Namespace) -> int:
    """Export experiment results from LangSmith."""
    try:
        print(f"[INFO] Exporting experiment results...")
        print(f"  Experiment: {args.name}")
        print(f"  Output: {args.output}")

        result = export_experiment_results(
            experiment_name=args.name,
            output_path=args.output,
            api_key=args.api_key,
        )

        print(f"\n[OK] Export complete")
        print(f"  Experiment: {result['experiment_name']}")
        print(f"  Total results: {result['total_results']}")
        print(f"  Output: {result['output_path']}")

        return 0

    except Exception as e:
        print(f"[ERROR] Export failed: {e}", file=sys.stderr)
        return 1


def cmd_native_experiment(args: argparse.Namespace) -> int:
    """Run a native LangSmith evaluation experiment using the official evaluate() API."""
    try:
        print(f"[INFO] Starting native LangSmith experiment...")
        print(f"  Experiment name: {args.name}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Output: {args.output}")
        print(f"  Max examples: {args.max_examples or 'all'}")
        print(f"  Max concurrent: {args.max_concurrent}")
        print(f"  Include programmatic: {args.include_programmatic}")
        print(f"  Include LLM judges: {args.include_llm_judges}")
        if args.experiment_prefix:
            print(f"  Experiment prefix: {args.experiment_prefix}")
        if args.description:
            print(f"  Description: {args.description}")
        print(f"  Num repetitions: {args.num_repetitions}")

        config = NativeExperimentConfig(
            experiment_name=args.name,
            dataset_name=args.dataset,
            evaluator_names=args.evaluator,
            include_programmatic=args.include_programmatic,
            include_llm_judges=args.include_llm_judges,
            max_concurrent=args.max_concurrent,
            max_examples=args.max_examples,
            experiment_prefix=args.experiment_prefix,
            description=args.description,
            num_repetitions=args.num_repetitions,
        )

        runner = NativeLangSmithExperimentRunner(config=config)
        results, summary = asyncio.run(runner.run_experiment())

        # Save results
        runner.save_results(results, summary, args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("NATIVE LANGSMITH EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Experiment: {summary.experiment_name}")
        print(f"Dataset: {summary.dataset_name}")
        print(f"Total examples: {summary.total_examples}")
        print(f"Successful: {summary.successful}")
        print(f"Failed: {summary.failed}")
        print(f"Errors: {summary.errors}")
        print(f"Total time: {summary.execution_time_ms:.0f} ms")

        print("\nEvaluator Summary:")
        for eval_name, eval_summary in summary.evaluator_summary.items():
            print(f"\n  {eval_name}:")
            print(f"    Passed: {eval_summary.get('passed', 0)}")
            print(f"    Failed: {eval_summary.get('failed', 0)}")
            print(f"    Needs Human Review: {eval_summary.get('needs_human_review', 0)}")
            print(f"    Insufficient Context: {eval_summary.get('insufficient_context', 0)}")
            print(f"    Errors: {eval_summary.get('errors', 0)}")
            if eval_summary.get('average_score') is not None:
                print(f"  Average Score: {eval_summary['average_score']:.3f}")
            if eval_summary.get('average_confidence') is not None:
                print(f"  Average Confidence: {eval_summary['average_confidence']:.3f}")

        print(f"\nResults saved to: {args.output}")
        print("=" * 60)

        return 0 if summary.failed == 0 else 1

    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}", file=sys.stderr)
        return 1


def cmd_upload_dataset_dedup(args: argparse.Namespace) -> int:
    """Upload evaluation dataset to LangSmith with deduplication."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        print(f"[INFO] Uploading dataset to LangSmith with deduplication...")
        print(f"  File: {args.file}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Include unverified: {args.include_unverified}")
        print(f"  Update existing: {args.update_existing}")

        result = asyncio.run(upload_dataset_deduplicated_native(
            filepath=args.file,
            dataset_name=args.dataset,
            api_key=args.api_key,
            description=args.description,
            create_if_missing=not args.no_create,
            exclude_unverified=not args.include_unverified,
            update_existing=args.update_existing,
        ))

        print(f"\n[OK] Upload complete to dataset '{args.dataset}'")
        print(f"  Uploaded: {result.get('uploaded', 0)}")
        print(f"  Skipped (invalid): {result.get('skipped', 0)}")
        print(f"  Skipped duplicates: {result.get('skipped_duplicates', 0)}")
        if result.get('new_cases_uploaded') is not None:
            print(f"  New cases uploaded: {result.get('new_cases_uploaded', 0)}")
        if result['validation_errors']:
            print(f"  Validation errors: {len(result['validation_errors'])}")
        if result['upload_errors']:
            print(f"  Upload errors: {len(result['upload_errors'])}")

        stats = result.get('dataset_stats', {})
        print(f"  Categories: {stats.get('categories', {})}")
        print(f"  Languages: {stats.get('languages', {})}")

        return 0

    except LangSmithUploadError as e:
        print(f"[ERROR] Upload failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_validate_dataset_dedup(args: argparse.Namespace) -> int:
    """Validate dataset with deduplication check."""
    try:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1

        print(f"[INFO] Validating dataset: {args.file}")
        result = validate_dataset_dedup(
            filepath=args.file,
            exclude_unverified=not args.include_unverified,
        )

        print(f"\n[VALIDATION RESULT]")
        print(f"  Valid: {result['valid']}")
        print(f"  Total cases: {result['total_cases']}")
        print(f"  Valid cases: {result['valid_cases']}")
        print(f"  Invalid cases: {result['invalid_cases']}")
        print(f"  Unverified cases: {result['unverified_cases']}")
        print(f"  Categories: {result['categories']}")
        print(f"  Languages: {result['languages']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Validation report saved to {args.output}")

        return 0 if result['valid'] else 1

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        return 1


def main():

    # Validate command (basic)
    validate_parser = subparsers.add_parser("validate", help="Validate dataset JSONL file (basic schema)")
    validate_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_parser.set_defaults(func=cmd_validate)

    # Validate full command (comprehensive)
    validate_full_parser = subparsers.add_parser("validate-full", help="Comprehensive dataset validation with detailed report")
    validate_full_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_full_parser.add_argument("--output", "-o", help="Path to save JSON validation report")
    validate_full_parser.set_defaults(func=cmd_validate_full)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    stats_parser.set_defaults(func=cmd_stats)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload dataset to LangSmith")
    upload_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    upload_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                               help="LangSmith dataset name")
    upload_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    upload_parser.add_argument("--description", help="Dataset description")
    upload_parser.add_argument("--no-create", action="store_true",
                               help="Fail if dataset doesn't exist (don't create)")
    upload_parser.set_defaults(func=cmd_upload)

    # List command
    list_parser = subparsers.add_parser("list", help="List LangSmith datasets")
    list_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    list_parser.set_defaults(func=cmd_list)

    # Run command (Phase 3 - Offline Evaluation)
    run_parser = subparsers.add_parser("run", help="Run offline evaluation against RAGChain")
    run_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    run_parser.add_argument("--output", "-o", required=True, help="Path to output JSON results file")
    run_parser.add_argument("--include-unverified", action="store_true",
                           help="Include cases requiring human verification (WARNING: results need medical review)")
    run_parser.add_argument("--category", "-c", action="append",
                           help=f"Filter by category (can specify multiple). Valid: {', '.join(sorted(VALID_CATEGORIES))}")
    run_parser.add_argument("--case-id", action="append",
                           help="Filter by case ID (can specify multiple)")
    run_parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve (default: config)")
    run_parser.add_argument("--max-concurrent", type=int, default=1,
                           help="Maximum concurrent evaluations (default: 1)")
    run_parser.set_defaults(func=cmd_run)

    # Score command (Phase 4 - Programmatic Evaluators)
    score_parser = subparsers.add_parser("score", help="Run programmatic evaluators on Phase 3 results")
    score_parser.add_argument("--input", "-i", required=True, help="Path to Phase 3 results JSON file")
    score_parser.add_argument("--output", "-o", required=True, help="Path to output scored results JSON file")
    score_parser.add_argument("--evaluator", "-e", action="append",
                             help=f"Evaluator to run (can specify multiple). Valid: {', '.join(sorted(PROGRAMMATIC_EVALUATORS.keys()))}")
    score_parser.add_argument("--correctness-threshold", type=float, default=0.4,
                             help="Token overlap threshold for correctness (default: 0.4)")
    score_parser.add_argument("--faithfulness-threshold", type=float, default=0.3,
                             help="Support threshold for faithfulness (default: 0.3)")
    score_parser.add_argument("--relevance-threshold", type=float, default=0.1,
                             help="Jaccard threshold for relevance (default: 0.1)")
    score_parser.set_defaults(func=cmd_score)

    # LLM Eval command (Phase 5 - LLM-as-Judge Evaluators)
    llm_eval_parser = subparsers.add_parser("llm-eval", help="Run LLM-as-Judge evaluators on Phase 3 results")
    llm_eval_parser.add_argument("--input", "-i", required=True, help="Path to Phase 3 results JSON file")
    llm_eval_parser.add_argument("--output", "-o", required=True, help="Path to output LLM scored results JSON file")
    llm_eval_parser.add_argument("--evaluator", "-e", action="append",
                                help=f"LLM evaluator to run (can specify multiple). Valid: {', '.join(sorted(LLM_JUDGES.keys()))}")
    llm_eval_parser.add_argument("--max-concurrent", type=int, default=3,
                                help="Maximum concurrent LLM evaluations (default: 3)")
    llm_eval_parser.set_defaults(func=cmd_llm_eval)

    # Validate dataset command (Phase 6 - LangSmith Integration)
    validate_dataset_parser = subparsers.add_parser("validate-dataset", help="Validate dataset before LangSmith upload")
    validate_dataset_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_dataset_parser.add_argument("--output", "-o", help="Path to save JSON validation report")
    validate_dataset_parser.add_argument("--include-unverified", action="store_true",
                                        help="Include cases requiring human verification")
    validate_dataset_parser.set_defaults(func=cmd_validate_dataset)

    # Dry-run command
    dry_run_parser = subparsers.add_parser("dry-run", help="Simulate dataset upload without making external API calls")
    dry_run_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    dry_run_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                               help="LangSmith dataset name")
    dry_run_parser.add_argument("--output", "-o", help="Path to save dry-run report")
    dry_run_parser.add_argument("--include-unverified", action="store_true",
                               help="Include cases requiring human verification")
    dry_run_parser.set_defaults(func=cmd_dry_run)

    # Upload dataset command
    upload_dataset_parser = subparsers.add_parser("upload-dataset", help="Upload evaluation dataset to LangSmith with filtering")
    upload_dataset_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    upload_dataset_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                      help="LangSmith dataset name")
    upload_dataset_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    upload_dataset_parser.add_argument("--description", help="Dataset description")
    upload_dataset_parser.add_argument("--no-create", action="store_true",
                                      help="Fail if dataset doesn't exist (don't create)")
    upload_dataset_parser.add_argument("--include-unverified", action="store_true",
                                      help="Include cases requiring human verification (WARNING: results need medical review)")
    upload_dataset_parser.set_defaults(func=cmd_upload_dataset)

    # Experiment command (Phase 6 - LangSmith Experiments)
    experiment_parser = subparsers.add_parser("experiment", help="Run LangSmith evaluation experiment")
    experiment_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    experiment_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                  help="LangSmith dataset name")
    experiment_parser.add_argument("--output", "-o", required=True, help="Path to output JSON results file")
    experiment_parser.add_argument("--evaluator", "-e", action="append",
                                  help=f"Evaluator to run (can specify multiple). Valid: {', '.join(sorted(set(list(PROGRAMMATIC_EVALUATORS.keys()) + list(LLM_JUDGES.keys()))))}")
    experiment_parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    experiment_parser.add_argument("--max-concurrent", type=int, default=3,
                                  help="Maximum concurrent evaluations (default: 3)")
    experiment_parser.add_argument("--include-programmatic", action="store_true", default=True,
                                  help="Include programmatic evaluators (default: True)")
    experiment_parser.add_argument("--no-programmatic", action="store_false", dest="include_programmatic",
                                  help="Exclude programmatic evaluators")
    experiment_parser.add_argument("--include-llm-judges", action="store_true", default=True,
                                  help="Include LLM judges (default: True)")
    experiment_parser.add_argument("--no-llm-judges", action="store_false", dest="include_llm_judges",
                                  help="Exclude LLM judges")
    experiment_parser.set_defaults(func=cmd_experiment)

    # Export experiment command
    export_parser = subparsers.add_parser("export-experiment", help="Export LangSmith experiment results")
    export_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    export_parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    export_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    export_parser.set_defaults(func=cmd_export_experiment)

    # Native Experiment command (Phase 6.1 - Native LangSmith Experiment Integration)
    native_experiment_parser = subparsers.add_parser("native-experiment", help="Run native LangSmith evaluation experiment")
    native_experiment_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    native_experiment_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                         help="LangSmith dataset name")
    native_experiment_parser.add_argument("--output", "-o", required=True, help="Path to output JSON results file")
    native_experiment_parser.add_argument("--evaluator", "-e", action="append",
                                         help=f"Evaluator to run (can specify multiple). Valid: {', '.join(sorted(set(list(PROGRAMMATIC_EVALUATORS.keys()) + list(LLM_JUDGES.keys()))))}")
    native_experiment_parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    native_experiment_parser.add_argument("--max-concurrent", type=int, default=3,
                                         help="Maximum concurrent evaluations (default: 3)")
    native_experiment_parser.add_argument("--include-programmatic", action="store_true", default=True,
                                         help="Include programmatic evaluators (default: True)")
    native_experiment_parser.add_argument("--no-programmatic", action="store_false", dest="include_programmatic",
                                         help="Exclude programmatic evaluators")
    native_experiment_parser.add_argument("--include-llm-judges", action="store_true", default=True,
                                         help="Include LLM judges (default: True)")
    native_experiment_parser.add_argument("--no-llm-judges", action="store_false", dest="include_llm_judges",
                                         help="Exclude LLM judges")
    native_experiment_parser.add_argument("--experiment-prefix", help="Experiment name prefix")
    native_experiment_parser.add_argument("--description", help="Experiment description")
    native_experiment_parser.add_argument("--num-repetitions", type=int, default=1,
                                         help="Number of repetitions per example (default: 1)")
    native_experiment_parser.set_defaults(func=cmd_native_experiment)

    # Upload dataset with deduplication command
    upload_dedup_parser = subparsers.add_parser("upload-dataset-dedup", help="Upload evaluation dataset to LangSmith with deduplication")
    upload_dedup_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    upload_dedup_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                    help="LangSmith dataset name")
    upload_dedup_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    upload_dedup_parser.add_argument("--description", help="Dataset description")
    upload_dedup_parser.add_argument("--no-create", action="store_true",
                                    help="Fail if dataset doesn't exist (don't create)")
    upload_dedup_parser.add_argument("--include-unverified", action="store_true",
                                    help="Include cases requiring human verification (WARNING: results need medical review)")
    upload_dedup_parser.add_argument("--update-existing", action="store_true",
                                    help="Update existing examples instead of skipping")
    upload_dedup_parser.set_defaults(func=cmd_upload_dataset_dedup)

    # Validate dataset with deduplication command
    validate_dedup_parser = subparsers.add_parser("validate-dataset-dedup", help="Validate dataset with deduplication check")
    validate_dedup_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_dedup_parser.add_argument("--output", "-o", help="Path to save JSON validation report")
    validate_dedup_parser.add_argument("--include-unverified", action="store_true",
                                      help="Include cases requiring human verification")
    validate_dedup_parser.set_defaults(func=cmd_validate_dataset_dedup)

    args = parser.parse_args()
    return args.func(args)


def main():
    parser = argparse.ArgumentParser(
        description="MediTrust Evaluation Dataset CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate command (basic)
    validate_parser = subparsers.add_parser("validate", help="Validate dataset JSONL file (basic schema)")
    validate_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_parser.set_defaults(func=cmd_validate)

    # Validate full command (comprehensive)
    validate_full_parser = subparsers.add_parser("validate-full", help="Comprehensive dataset validation with detailed report")
    validate_full_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_full_parser.add_argument("--output", "-o", help="Path to save JSON validation report")
    validate_full_parser.set_defaults(func=cmd_validate_full)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    stats_parser.set_defaults(func=cmd_stats)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload dataset to LangSmith")
    upload_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    upload_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                               help="LangSmith dataset name")
    upload_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    upload_parser.add_argument("--description", help="Dataset description")
    upload_parser.add_argument("--no-create", action="store_true",
                               help="Fail if dataset doesn't exist (don't create)")
    upload_parser.set_defaults(func=cmd_upload)

    # List command
    list_parser = subparsers.add_parser("list", help="List LangSmith datasets")
    list_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    list_parser.set_defaults(func=cmd_list)

    # Run command (Phase 3 - Offline Evaluation)
    run_parser = subparsers.add_parser("run", help="Run offline evaluation against RAGChain")
    run_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    run_parser.add_argument("--output", "-o", required=True, help="Path to output JSON results file")
    run_parser.add_argument("--include-unverified", action="store_true",
                           help="Include cases requiring human verification (WARNING: results need medical review)")
    run_parser.add_argument("--category", "-c", action="append",
                           help=f"Filter by category (can specify multiple). Valid: {', '.join(sorted(VALID_CATEGORIES))}")
    run_parser.add_argument("--case-id", action="append",
                           help="Filter by case ID (can specify multiple)")
    run_parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve (default: config)")
    run_parser.add_argument("--max-concurrent", type=int, default=1,
                           help="Maximum concurrent evaluations (default: 1)")
    run_parser.set_defaults(func=cmd_run)

    # Score command (Phase 4 - Programmatic Evaluators)
    score_parser = subparsers.add_parser("score", help="Run programmatic evaluators on Phase 3 results")
    score_parser.add_argument("--input", "-i", required=True, help="Path to Phase 3 results JSON file")
    score_parser.add_argument("--output", "-o", required=True, help="Path to output scored results JSON file")
    score_parser.add_argument("--evaluator", "-e", action="append",
                             help=f"Evaluator to run (can specify multiple). Valid: {', '.join(sorted(PROGRAMMATIC_EVALUATORS.keys()))}")
    score_parser.add_argument("--correctness-threshold", type=float, default=0.4,
                             help="Token overlap threshold for correctness (default: 0.4)")
    score_parser.add_argument("--faithfulness-threshold", type=float, default=0.3,
                             help="Support threshold for faithfulness (default: 0.3)")
    score_parser.add_argument("--relevance-threshold", type=float, default=0.1,
                             help="Jaccard threshold for relevance (default: 0.1)")
    score_parser.set_defaults(func=cmd_score)

    # LLM Eval command (Phase 5 - LLM-as-Judge Evaluators)
    llm_eval_parser = subparsers.add_parser("llm-eval", help="Run LLM-as-Judge evaluators on Phase 3 results")
    llm_eval_parser.add_argument("--input", "-i", required=True, help="Path to Phase 3 results JSON file")
    llm_eval_parser.add_argument("--output", "-o", required=True, help="Path to output LLM scored results JSON file")
    llm_eval_parser.add_argument("--evaluator", "-e", action="append",
                                help=f"LLM evaluator to run (can specify multiple). Valid: {', '.join(sorted(LLM_JUDGES.keys()))}")
    llm_eval_parser.add_argument("--max-concurrent", type=int, default=3,
                                help="Maximum concurrent LLM evaluations (default: 3)")
    llm_eval_parser.set_defaults(func=cmd_llm_eval)

    # Validate dataset command (Phase 6 - LangSmith Integration)
    validate_dataset_parser = subparsers.add_parser("validate-dataset", help="Validate dataset before LangSmith upload")
    validate_dataset_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_dataset_parser.add_argument("--output", "-o", help="Path to save JSON validation report")
    validate_dataset_parser.add_argument("--include-unverified", action="store_true",
                                        help="Include cases requiring human verification")
    validate_dataset_parser.set_defaults(func=cmd_validate_dataset)

    # Dry-run command
    dry_run_parser = subparsers.add_parser("dry-run", help="Simulate dataset upload without making external API calls")
    dry_run_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    dry_run_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                               help="LangSmith dataset name")
    dry_run_parser.add_argument("--output", "-o", help="Path to save dry-run report")
    dry_run_parser.add_argument("--include-unverified", action="store_true",
                               help="Include cases requiring human verification")
    dry_run_parser.set_defaults(func=cmd_dry_run)

    # Upload dataset command
    upload_dataset_parser = subparsers.add_parser("upload-dataset", help="Upload evaluation dataset to LangSmith with filtering")
    upload_dataset_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    upload_dataset_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                      help="LangSmith dataset name")
    upload_dataset_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    upload_dataset_parser.add_argument("--description", help="Dataset description")
    upload_dataset_parser.add_argument("--no-create", action="store_true",
                                      help="Fail if dataset doesn't exist (don't create)")
    upload_dataset_parser.add_argument("--include-unverified", action="store_true",
                                      help="Include cases requiring human verification (WARNING: results need medical review)")
    upload_dataset_parser.set_defaults(func=cmd_upload_dataset)

    # Experiment command (Phase 6 - LangSmith Experiments)
    experiment_parser = subparsers.add_parser("experiment", help="Run LangSmith evaluation experiment")
    experiment_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    experiment_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                  help="LangSmith dataset name")
    experiment_parser.add_argument("--output", "-o", required=True, help="Path to output JSON results file")
    experiment_parser.add_argument("--evaluator", "-e", action="append",
                                  help=f"Evaluator to run (can specify multiple). Valid: {', '.join(sorted(set(list(PROGRAMMATIC_EVALUATORS.keys()) + list(LLM_JUDGES.keys()))))}")
    experiment_parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    experiment_parser.add_argument("--max-concurrent", type=int, default=3,
                                  help="Maximum concurrent evaluations (default: 3)")
    experiment_parser.add_argument("--include-programmatic", action="store_true", default=True,
                                  help="Include programmatic evaluators (default: True)")
    experiment_parser.add_argument("--no-programmatic", action="store_false", dest="include_programmatic",
                                  help="Exclude programmatic evaluators")
    experiment_parser.add_argument("--include-llm-judges", action="store_true", default=True,
                                  help="Include LLM judges (default: True)")
    experiment_parser.add_argument("--no-llm-judges", action="store_false", dest="include_llm_judges",
                                  help="Exclude LLM judges")
    experiment_parser.set_defaults(func=cmd_experiment)

    # Export experiment command
    export_parser = subparsers.add_parser("export-experiment", help="Export LangSmith experiment results")
    export_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    export_parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    export_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    export_parser.set_defaults(func=cmd_export_experiment)

    # Native Experiment command (Phase 6.1 - Native LangSmith Experiment Integration)
    native_experiment_parser = subparsers.add_parser("native-experiment", help="Run native LangSmith evaluation experiment")
    native_experiment_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    native_experiment_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                         help="LangSmith dataset name")
    native_experiment_parser.add_argument("--output", "-o", required=True, help="Path to output JSON results file")
    native_experiment_parser.add_argument("--evaluator", "-e", action="append",
                                         help=f"Evaluator to run (can specify multiple). Valid: {', '.join(sorted(set(list(PROGRAMMATIC_EVALUATORS.keys()) + list(LLM_JUDGES.keys()))))}")
    native_experiment_parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    native_experiment_parser.add_argument("--max-concurrent", type=int, default=3,
                                         help="Maximum concurrent evaluations (default: 3)")
    native_experiment_parser.add_argument("--include-programmatic", action="store_true", default=True,
                                         help="Include programmatic evaluators (default: True)")
    native_experiment_parser.add_argument("--no-programmatic", action="store_false", dest="include_programmatic",
                                         help="Exclude programmatic evaluators")
    native_experiment_parser.add_argument("--include-llm-judges", action="store_true", default=True,
                                         help="Include LLM judges (default: True)")
    native_experiment_parser.add_argument("--no-llm-judges", action="store_false", dest="include_llm_judges",
                                         help="Exclude LLM judges")
    native_experiment_parser.add_argument("--experiment-prefix", help="Experiment name prefix")
    native_experiment_parser.add_argument("--description", help="Experiment description")
    native_experiment_parser.add_argument("--num-repetitions", type=int, default=1,
                                         help="Number of repetitions per example (default: 1)")
    native_experiment_parser.set_defaults(func=cmd_native_experiment)

    # Upload dataset with deduplication command
    upload_dedup_parser = subparsers.add_parser("upload-dataset-dedup", help="Upload evaluation dataset to LangSmith with deduplication")
    upload_dedup_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    upload_dedup_parser.add_argument("--dataset", "-d", default="meditrust-rag-evaluation-v1",
                                    help="LangSmith dataset name")
    upload_dedup_parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY)")
    upload_dedup_parser.add_argument("--description", help="Dataset description")
    upload_dedup_parser.add_argument("--no-create", action="store_true",
                                    help="Fail if dataset doesn't exist (don't create)")
    upload_dedup_parser.add_argument("--include-unverified", action="store_true",
                                    help="Include cases requiring human verification (WARNING: results need medical review)")
    upload_dedup_parser.add_argument("--update-existing", action="store_true",
                                    help="Update existing examples instead of skipping")
    upload_dedup_parser.set_defaults(func=cmd_upload_dataset_dedup)

    # Validate dataset with deduplication command
    validate_dedup_parser = subparsers.add_parser("validate-dataset-dedup", help="Validate dataset with deduplication check")
    validate_dedup_parser.add_argument("--file", "-f", required=True, help="Path to dataset JSONL file")
    validate_dedup_parser.add_argument("--output", "-o", help="Path to save JSON validation report")
    validate_dedup_parser.add_argument("--include-unverified", action="store_true",
                                      help="Include cases requiring human verification")
    validate_dedup_parser.set_defaults(func=cmd_validate_dataset_dedup)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())