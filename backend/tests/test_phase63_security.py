"""Phase 6.3 & Phase 7 security and baseline evaluation tests.

Covers:
- Missing LANGSMITH_API_KEY fails fast with a clear message (no env-var leakage).
- LANGSMITH_PROJECT routes traces to the configured project.
- Simulated PII/PHI in a RAG response triggers fail-closed blocking before
  reaching LangSmith.
- assert_safe_environment blocks --no-sanitize outside development.
- cmd_run_baseline verifies dataset, stamps metadata, and saves a clean report.
"""
import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(
    question: str = "What are visiting hours?",
    expected: str = "9 AM to 5 PM",
    context: str = "Visiting hours: 9 AM to 5 PM",
    needs_human_verification: bool = False,
):
    from langsmith.schemas import Example

    return Example(
        id=uuid.uuid4(),
        inputs={"question": question, "category": "general_info", "language": "en", "tags": []},
        outputs={
            "expected_answer": expected,
            "reference_context": context,
            "needs_human_verification": needs_human_verification,
        },
    )


# ---------------------------------------------------------------------------
# 1. Missing LANGSMITH_API_KEY — fast failure, no secret in error message
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    """create_langsmith_client raises clearly when no API key is available."""

    def test_missing_key_raises_upload_error(self):
        from app.ai.evaluation.langsmith_utils import (
            LangSmithUploadError,
            create_langsmith_client,
        )

        with patch.dict(os.environ, {}, clear=True):
            # Also clear LANGSMITH_API_KEY if present in settings default
            env_without_key = {
                k: v for k, v in os.environ.items() if k != "LANGSMITH_API_KEY"
            }
            with patch.dict(os.environ, env_without_key, clear=True):
                with pytest.raises(LangSmithUploadError) as exc_info:
                    create_langsmith_client()

        msg = str(exc_info.value)
        assert "API key" in msg, "Error should mention API key"
        # Must NOT leak any actual key value.
        assert "sk-" not in msg
        assert "lsv2_" not in msg

    def test_error_message_does_not_contain_env_values(self):
        """Even if the key is present but wrong-shaped, no value is leaked."""
        from app.ai.evaluation.langsmith_utils import (
            LangSmithUploadError,
            create_langsmith_client,
        )

        # Patch Client to raise so the key is never used in a real call.
        with patch("app.ai.evaluation.langsmith_utils.Client") as mock_cls:
            mock_cls.side_effect = Exception("connection refused")
            with patch.dict(os.environ, {"LANGSMITH_API_KEY": "SECRET_VALUE_123"}):
                with pytest.raises(Exception) as exc_info:
                    create_langsmith_client()
                # The original exception is re-raised — it must not contain the key.
                assert "SECRET_VALUE_123" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. LANGSMITH_PROJECT routing
# ---------------------------------------------------------------------------

class TestProjectRouting:
    """LANGSMITH_PROJECT is set on the OS environment before the Client is built."""

    def test_project_from_explicit_arg(self):
        from app.ai.evaluation.langsmith_utils import create_langsmith_client

        with patch("app.ai.evaluation.langsmith_utils.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            # Isolate LANGSMITH_PROJECT so the function's write is the only source.
            isolated_env = {
                k: v for k, v in os.environ.items()
                if k not in ("LANGSMITH_PROJECT",)
            }
            isolated_env["LANGSMITH_API_KEY"] = "test-key"
            with patch.dict(os.environ, isolated_env, clear=True):
                create_langsmith_client(project="meditrust-rag-evaluation")
                # Read immediately while the patched env is still active.
                assert os.environ.get("LANGSMITH_PROJECT") == "meditrust-rag-evaluation"

        mock_cls.assert_called_once_with(api_key="test-key")

    def test_project_from_env_var(self):
        from app.ai.evaluation.langsmith_utils import create_langsmith_client

        with patch("app.ai.evaluation.langsmith_utils.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            isolated_env = {
                k: v for k, v in os.environ.items()
                if k not in ("LANGSMITH_PROJECT",)
            }
            isolated_env["LANGSMITH_API_KEY"] = "test-key"
            isolated_env["LANGSMITH_PROJECT"] = "my-custom-project"
            with patch.dict(os.environ, isolated_env, clear=True):
                create_langsmith_client()
                assert os.environ.get("LANGSMITH_PROJECT") == "my-custom-project"

    def test_project_falls_back_to_settings_default(self):
        from app.ai.evaluation.langsmith_utils import create_langsmith_client

        with patch("app.ai.evaluation.langsmith_utils.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            isolated_env = {
                k: v for k, v in os.environ.items()
                if k not in ("LANGSMITH_PROJECT",)
            }
            isolated_env["LANGSMITH_API_KEY"] = "test-key"
            with patch.dict(os.environ, isolated_env, clear=True):
                create_langsmith_client()
                # Should have fallen back to settings.LANGSMITH_PROJECT or hardcoded default.
                project_set = os.environ.get("LANGSMITH_PROJECT", "")
                assert project_set != ""

    def test_native_experiment_config_project_threaded_to_client(self):
        """NativeLangSmithExperimentRunner passes project to create_langsmith_client."""
        from app.ai.evaluation.langsmith_native import (
            NativeExperimentConfig,
            NativeLangSmithExperimentRunner,
        )

        config = NativeExperimentConfig(
            experiment_name="test",
            project="my-isolated-project",
        )

        with patch(
            "app.ai.evaluation.langsmith_native.create_langsmith_client"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}, clear=False):
                NativeLangSmithExperimentRunner(config=config)

        mock_create.assert_called_once_with(None, project="my-isolated-project")

    def test_experiment_config_project_threaded_to_client(self):
        """LangSmithExperimentRunner passes project to create_langsmith_client."""
        from app.ai.evaluation.langsmith_experiment import (
            ExperimentConfig,
            LangSmithExperimentRunner,
        )

        config = ExperimentConfig(
            experiment_name="test",
            project="my-isolated-project",
        )

        with patch(
            "app.ai.evaluation.langsmith_experiment.create_langsmith_client"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}, clear=False):
                LangSmithExperimentRunner(config=config)

        mock_create.assert_called_once_with(None, project="my-isolated-project")


# ---------------------------------------------------------------------------
# 3. PHI/PII in RAG output — fail-closed blocking before LangSmith
# ---------------------------------------------------------------------------

class TestExperimentSanitization:
    """sanitize_experiment_payload blocks PHI transmission and logs only redacted info."""

    def test_clean_payload_passes_through(self):
        from app.ai.evaluation.langsmith_utils import sanitize_experiment_payload

        payload = {
            "answer": "Visiting hours are 9 AM to 5 PM on weekdays.",
            "has_context": True,
            "model": "test-model",
        }
        result = sanitize_experiment_payload(payload, case_id="case_001")
        assert result["answer"] == payload["answer"]  # unchanged
        assert result["has_context"] is True

    def test_ssn_in_answer_raises_sanitization_error(self):
        from app.ai.evaluation.langsmith_utils import (
            ExperimentSanitizationError,
            sanitize_experiment_payload,
        )

        payload = {"answer": "Patient SSN is 123-45-6789, see records."}
        with pytest.raises(ExperimentSanitizationError) as exc_info:
            sanitize_experiment_payload(payload, case_id="case_ssn")

        msg = str(exc_info.value)
        assert "case_ssn" in msg
        # Raw text must NOT appear in the exception message.
        assert "123-45-6789" not in msg

    def test_email_in_answer_raises_sanitization_error(self):
        from app.ai.evaluation.langsmith_utils import (
            ExperimentSanitizationError,
            sanitize_experiment_payload,
        )

        payload = {"answer": "Contact the patient at john.doe@hospital.com for follow-up."}
        with pytest.raises(ExperimentSanitizationError) as exc_info:
            sanitize_experiment_payload(payload, case_id="case_email")

        assert "case_email" in str(exc_info.value)
        assert "john.doe@hospital.com" not in str(exc_info.value)

    def test_mrn_in_context_raises_sanitization_error(self):
        from app.ai.evaluation.langsmith_utils import (
            ExperimentSanitizationError,
            sanitize_experiment_payload,
        )

        payload = {"reference_context": "MRN: 00123456 — patient admitted 03/15/2024"}
        with pytest.raises(ExperimentSanitizationError) as exc_info:
            sanitize_experiment_payload(payload, case_id="case_mrn")

        assert "case_mrn" in str(exc_info.value)
        assert "00123456" not in str(exc_info.value)

    def test_nested_phi_in_sources_blocked(self):
        from app.ai.evaluation.langsmith_utils import (
            ExperimentSanitizationError,
            sanitize_experiment_payload,
        )

        payload = {
            "answer": "See your doctor.",
            "sources": [{"content": "Patient John Doe: DOB 01/01/1980"}],
        }
        # sanitize_experiment_payload validates the entire dict recursively.
        # PHI nested in sources must be detected and blocked.
        with pytest.raises(ExperimentSanitizationError) as exc_info:
            sanitize_experiment_payload(payload, case_id="case_nested")

        msg = str(exc_info.value)
        assert "case_nested" in msg
        # Raw text must NOT appear in the exception message.
        assert "John Doe" not in msg
        assert "01/01/1980" not in msg

    @pytest.mark.asyncio
    async def test_ragchain_target_blocks_ssn_in_answer(self):
        """RAGChainTarget raises ExperimentSanitizationError when answer contains SSN."""
        from app.ai.evaluation.langsmith_native import RAGChainTarget
        from app.ai.rag.chain import RAGChain, RAGResponse
        from app.ai.evaluation.langsmith_utils import ExperimentSanitizationError

        mock_chain = RAGChain()
        mock_chain.generate = AsyncMock(
            return_value=RAGResponse(
                answer="Patient SSN: 123-45-6789 confirmed",
                sources=[],
                has_context=False,
                model="test-model",
            )
        )

        target = RAGChainTarget(mock_chain)
        with pytest.raises(ExperimentSanitizationError):
            await target({"question": "What is the patient's SSN?", "case_id": "ssn_test"})

    @pytest.mark.asyncio
    async def test_ragchain_target_allows_clean_answer(self):
        """RAGChainTarget returns sanitized output for a clean answer."""
        from app.ai.evaluation.langsmith_native import RAGChainTarget
        from app.ai.rag.chain import RAGChain, RAGResponse

        mock_chain = RAGChain()
        mock_chain.generate = AsyncMock(
            return_value=RAGResponse(
                answer="Visiting hours are 9 AM to 5 PM.",
                sources=[{"content_preview": "Hospital hours: Mon-Fri 9-17"}],
                has_context=True,
                model="test-model",
            )
        )

        target = RAGChainTarget(mock_chain)
        result = await target({"question": "What are visiting hours?", "case_id": "vh_test"})
        assert result["answer"] == "Visiting hours are 9 AM to 5 PM."
        assert result["has_context"] is True


# ---------------------------------------------------------------------------
# 4. --no-sanitize blocked outside development
# ---------------------------------------------------------------------------

class TestUnsafeEnvironmentGuard:
    """assert_safe_environment blocks --no-sanitize in non-development environments."""

    def test_allow_unsafe_false_always_passes(self):
        from app.ai.evaluation.langsmith_utils import assert_safe_environment

        # When allow_unsafe=False the function is a no-op.
        assert_safe_environment(allow_unsafe=False)  # no raise

    def test_allow_unsafe_true_in_development_passes(self):
        from app.ai.evaluation.langsmith_utils import assert_safe_environment

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert_safe_environment(allow_unsafe=True)  # no raise

    def test_allow_unsafe_true_in_production_raises(self):
        from app.ai.evaluation.langsmith_utils import assert_safe_environment

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with pytest.raises(RuntimeError) as exc_info:
                assert_safe_environment(allow_unsafe=True)

        assert "production" in str(exc_info.value)

    def test_allow_unsafe_true_in_staging_raises(self):
        from app.ai.evaluation.langsmith_utils import assert_safe_environment

        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            with pytest.raises(RuntimeError):
                assert_safe_environment(allow_unsafe=True)

    def test_allow_unsafe_true_with_local_env_passes(self):
        from app.ai.evaluation.langsmith_utils import assert_safe_environment

        for env_name in ("local", "dev", "development"):
            with patch.dict(os.environ, {"ENVIRONMENT": env_name}):
                assert_safe_environment(allow_unsafe=True)  # no raise


# ---------------------------------------------------------------------------
# 5. Baseline runner (cmd_run_baseline) — unit-level mock
# ---------------------------------------------------------------------------

class TestRunBaselineCommand:
    """cmd_run_baseline verifies dataset, stamps metadata, and saves a clean report."""

    def _make_namespace(self, **kwargs):
        import argparse

        defaults = {
            "dataset": "meditrust-rag-evaluation-v1",
            "output": "",
            "project": "meditrust-rag-evaluation",
            "max_concurrent": 3,
            "max_examples": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_missing_api_key_returns_error_code(self, tmp_path, capsys):
        from app.cli.evaluation import cmd_run_baseline
        from app.ai.evaluation.langsmith_utils import LangSmithUploadError

        with patch(
            "app.cli.evaluation.create_langsmith_client",
            side_effect=LangSmithUploadError("LangSmith API key not provided"),
        ):
            args = self._make_namespace(output=str(tmp_path / "baseline.json"))
            result = cmd_run_baseline(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "API key" in captured.err
        # Key value must NOT appear
        assert "lsv2_" not in captured.err
        assert "sk-" not in captured.err

    def test_dataset_not_found_returns_error_code(self, tmp_path, capsys):
        from app.cli.evaluation import cmd_run_baseline

        mock_client = MagicMock()
        with patch("app.cli.evaluation.create_langsmith_client", return_value=mock_client):
            with patch("app.cli.evaluation.dataset_exists", return_value=False):
                args = self._make_namespace(output=str(tmp_path / "baseline.json"))
                result = cmd_run_baseline(args)

        assert result == 1
        assert "does not exist" in capsys.readouterr().err

    def test_unverified_examples_blocked(self, tmp_path, capsys):
        from app.cli.evaluation import cmd_run_baseline

        mock_client = MagicMock()
        unverified_ex = _make_example(needs_human_verification=True)
        mock_client.list_examples.return_value = [unverified_ex]

        with patch("app.cli.evaluation.create_langsmith_client", return_value=mock_client):
            with patch("app.cli.evaluation.dataset_exists", return_value=True):
                args = self._make_namespace(output=str(tmp_path / "baseline.json"))
                result = cmd_run_baseline(args)

        assert result == 1
        assert "needs_human_verification" in capsys.readouterr().err

    def test_baseline_report_contains_no_raw_text(self, tmp_path):
        """The saved report contains only metrics — no question/answer strings."""
        from app.cli.evaluation import cmd_run_baseline
        from app.ai.evaluation.langsmith_native import NativeExperimentResult

        mock_client = MagicMock()
        verified_ex = _make_example()
        mock_client.list_examples.return_value = [verified_ex]

        # Build a minimal summary
        summary = NativeExperimentResult(
            experiment_name="baseline-phase7-abc1234",
            experiment_id="exp-001",
            dataset_name="meditrust-rag-evaluation-v1",
            total_examples=1,
            successful=1,
            failed=0,
            errors=0,
            execution_time_ms=500.0,
            evaluator_summary={"correctness": {"passed": 1, "failed": 0, "average_score": 1.0}},
            langsmith_experiment_url="https://smith.langchain.com/projects/p/exp-001",
        )
        detailed = [
            {
                "example_id": str(verified_ex.id),
                "run_id": str(uuid.uuid4()),
                "category": "general_info",
                "language": "en",
                "execution_status": "success",
                # Include some text fields that must NOT appear in the report
                "question": "What are visiting hours?",
                "actual_answer": "9 AM to 5 PM",
                "evaluator_results": {
                    "correctness": {"score": 1.0, "passed": True, "status": "passed"}
                },
            }
        ]

        output_file = str(tmp_path / "baseline.json")

        with patch("app.cli.evaluation.create_langsmith_client", return_value=mock_client):
            with patch("app.cli.evaluation.dataset_exists", return_value=True):
                with patch(
                    "app.cli.evaluation.NativeLangSmithExperimentRunner"
                ) as mock_runner_cls:
                    runner_instance = MagicMock()
                    runner_instance.run_experiment = AsyncMock(
                        return_value=(detailed, summary)
                    )
                    mock_runner_cls.return_value = runner_instance

                    args = self._make_namespace(output=output_file)
                    result = cmd_run_baseline(args)

        assert result == 0
        report = json.loads(Path(output_file).read_text())

        # Required fields
        assert report["phase"] == "7-baseline"
        assert report["git_sha"] != ""
        assert "evaluator_summary" in report
        assert "per_example_scores" in report

        # Scores recorded
        scores = report["per_example_scores"]
        assert len(scores) == 1
        assert scores[0]["evaluator_results"]["correctness"]["score"] == 1.0

        # Raw text must NOT appear in the report at any level
        report_str = json.dumps(report)
        assert "What are visiting hours?" not in report_str
        assert "9 AM to 5 PM" not in report_str

    def test_baseline_metadata_contains_phase_and_git_sha(self, tmp_path):
        """NativeExperimentConfig is created with the correct Phase 7 metadata."""
        from app.cli.evaluation import cmd_run_baseline
        from app.ai.evaluation.langsmith_native import NativeExperimentResult

        mock_client = MagicMock()
        mock_client.list_examples.return_value = [_make_example()]

        summary = NativeExperimentResult(
            experiment_name="baseline-phase7-deadbeef",
            experiment_id="exp-002",
            dataset_name="meditrust-rag-evaluation-v1",
            total_examples=1,
            successful=1,
            failed=0,
            errors=0,
            execution_time_ms=100.0,
            evaluator_summary={},
        )

        config_calls: List[Dict] = []

        def capture_config(*args, **kwargs):
            runner = MagicMock()
            runner.run_experiment = AsyncMock(return_value=([], summary))
            # Capture the config that was passed
            if args:
                config_calls.append({"config": args[0]})
            elif "config" in kwargs:
                config_calls.append({"config": kwargs["config"]})
            return runner

        output_file = str(tmp_path / "baseline.json")

        with patch("app.cli.evaluation.create_langsmith_client", return_value=mock_client):
            with patch("app.cli.evaluation.dataset_exists", return_value=True):
                with patch(
                    "app.cli.evaluation.NativeLangSmithExperimentRunner",
                    side_effect=capture_config,
                ):
                    args = self._make_namespace(output=output_file)
                    cmd_run_baseline(args)

        assert len(config_calls) == 1
        config = config_calls[0]["config"]
        meta = config.metadata
        assert meta["phase"] == "7-baseline"
        assert meta["dataset"] == "meditrust-rag-evaluation-v1"
        assert "git_sha" in meta
        assert config.experiment_prefix == "baseline"
        assert config.include_programmatic is True
        assert config.include_llm_judges is True

    def test_phi_in_experiment_output_returns_error_code(self, tmp_path, capsys):
        """PHI detected during the experiment is blocked and returns non-zero exit."""
        from app.cli.evaluation import cmd_run_baseline
        from app.ai.evaluation.langsmith_utils import ExperimentSanitizationError

        mock_client = MagicMock()
        mock_client.list_examples.return_value = [_make_example()]

        output_file = str(tmp_path / "baseline.json")

        with patch("app.cli.evaluation.create_langsmith_client", return_value=mock_client):
            with patch("app.cli.evaluation.dataset_exists", return_value=True):
                with patch(
                    "app.cli.evaluation.NativeLangSmithExperimentRunner"
                ) as mock_runner_cls:
                    runner_instance = MagicMock()
                    runner_instance.run_experiment = AsyncMock(
                        side_effect=ExperimentSanitizationError(
                            "Experiment payload for case 'case_001' contains sensitive data (SSN). Transmission blocked."
                        )
                    )
                    mock_runner_cls.return_value = runner_instance

                    args = self._make_namespace(output=output_file)
                    result = cmd_run_baseline(args)

        assert result == 1
        assert "PHI detected" in capsys.readouterr().err
        # Report file must not exist when PHI is detected
        assert not Path(output_file).exists()


# ---------------------------------------------------------------------------
# 6. Diagnosis rule — patient-linked vs. general medical queries
# ---------------------------------------------------------------------------

class TestDiagnosisPatternRefinement:
    """Patient-linked diagnoses are blocked; general medical queries are allowed."""

    # --- Cases that must NOT trigger a Diagnosis block ---

    def test_general_cancer_question_allowed(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "What is the treatment for cancer?"
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits == [], f"Should not flag general question, got: {hits}"

    def test_general_cancer_answer_allowed(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = (
            "Cancer treatment depends on the type, stage, and individual factors. "
            "Please consult an oncologist for personalized treatment options."
        )
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits == [], f"Should not flag general answer, got: {hits}"

    def test_diabetes_education_context_allowed(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = (
            "The knowledge base contains diabetes education materials only. "
            "No cancer treatment information is available in the current document store."
        )
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits == [], f"Should not flag education context, got: {hits}"

    def test_generic_hypertension_service_allowed(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "The hospital provides hypertension management and diabetes screening services."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits == [], f"Should not flag generic service description, got: {hits}"

    def test_generic_patient_reference_allowed(self):
        """'This patient has hypertension' — no personal name/MRN, should not block."""
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "This patient has hypertension."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits == [], f"Should not flag generic patient reference, got: {hits}"

    # --- Cases that MUST still trigger a Diagnosis block ---

    def test_proper_name_plus_diagnosis_blocked(self):
        """'John Doe has diabetes' — proper name identifies a person, must block."""
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data, validate_no_sensitive_data, SanitizationError

        text = "John Doe has diabetes and takes metformin."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: proper name + diagnosis"

        with pytest.raises(SanitizationError):
            validate_no_sensitive_data(text, context="test")

    def test_patient_label_plus_diagnosis_blocked(self):
        """'Patient: Jane Smith was diagnosed with hypertension' must block."""
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "Patient: Jane Smith was diagnosed with hypertension."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: Patient: label + proper name + diagnosis"

    def test_name_label_plus_diagnosis_blocked(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "Name: Bob Jones has HIV."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: Name: label + proper name + diagnosis"

    def test_mrn_plus_diagnosis_blocked(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "MRN 00123 patient has depression."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: MRN + diagnosis"

    def test_title_prefix_plus_diagnosis_blocked(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "Mrs. Williams has anxiety disorder."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: Mrs. title + diagnosis"

    def test_mr_title_prefix_plus_diagnosis_blocked(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "Mr. Smith suffers from depression."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: Mr. title + diagnosis"

    def test_full_name_alone_triggers_diagnosis_block(self):
        """Full name without Patient:/MRN label still triggers because _PROPER_NAME matches."""
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        text = "Alice Brown has hypertension."
        hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Diagnosis"]
        assert hits, "Should block: proper name + diagnosis (no label needed)"

    # --- Dataset-specific regression: eval_017 fields must all pass ---

    def test_eval_017_question_passes(self):
        from app.ai.evaluation.langsmith_utils import validate_no_sensitive_data

        # Must not raise
        validate_no_sensitive_data("What is the treatment for cancer?", context="eval_017.question")

    def test_eval_017_expected_answer_passes(self):
        from app.ai.evaluation.langsmith_utils import validate_no_sensitive_data

        text = (
            "I don't have enough information to answer this question. "
            "Cancer treatment depends on the type, stage, and individual factors. "
            "Please consult an oncologist for personalized treatment options."
        )
        validate_no_sensitive_data(text, context="eval_017.expected_answer")

    def test_eval_017_reference_context_passes(self):
        from app.ai.evaluation.langsmith_utils import validate_no_sensitive_data

        text = (
            "The knowledge base contains diabetes education materials only. "
            "No cancer treatment information is available in the current document store."
        )
        validate_no_sensitive_data(text, context="eval_017.reference_context")


# ---------------------------------------------------------------------------
# 7. Phone-number dataset fix — HOSPITAL_CONTACT placeholder passes validator
# ---------------------------------------------------------------------------

class TestPhoneNumberDatasetFix:
    """Synthetic phone placeholders in the dataset must not trigger the Phone pattern."""

    def test_hospital_contact_placeholder_not_flagged(self):
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        texts = [
            "Phone: [HOSPITAL_CONTACT]. Appointments can be booked via phone or patient portal.",
            "Dr. Deven Patel - General Physician (MBBS), 6 years experience. Available: 10:00 AM to 7:30 PM. Phone: [HOSPITAL_CONTACT].",
            "Available: 11:00 AM to 7:00 PM. Phone: [HOSPITAL_CONTACT].",
        ]
        for text in texts:
            hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Phone"]
            assert hits == [], f"[HOSPITAL_CONTACT] should not trigger Phone, got: {hits} for {text!r}"

    def test_real_10digit_phone_still_blocked(self):
        """The 10-digit Phone pattern must still catch real numbers."""
        from app.ai.evaluation.langsmith_utils import detect_sensitive_data

        texts = [
            "Phone: 9876543210.",
            "Contact: 9700112232",
            "Call 9812345678 for appointment.",
        ]
        for text in texts:
            hits = [h for h in detect_sensitive_data(text) if h["pattern_type"] == "Phone"]
            assert hits, f"Real 10-digit phone should be blocked in: {text!r}"

    def test_eval_004_reference_context_passes_after_fix(self):
        from app.ai.evaluation.langsmith_utils import validate_no_sensitive_data

        text = (
            "Dr. Mahesh Ramani - Cardiologist (MBBS, MD Cardiology), 12 years experience. "
            "Available: 5:00 AM to 11:00 PM. Phone: [HOSPITAL_CONTACT]. "
            "Appointments can be booked via phone or patient portal."
        )
        validate_no_sensitive_data(text, context="eval_004.reference_context")

    def test_eval_024_expected_answer_passes_after_fix(self):
        from app.ai.evaluation.langsmith_utils import validate_no_sensitive_data

        text = (
            "Dr. Deven Patel is a General Physician (MBBS) with 6 years of experience. "
            "Available: 10:00 AM to 7:30 PM. Phone: [HOSPITAL_CONTACT]. "
            "Specialization: General Physician."
        )
        validate_no_sensitive_data(text, context="eval_024.expected_answer")

    def test_eval_025_expected_answer_passes_after_fix(self):
        from app.ai.evaluation.langsmith_utils import validate_no_sensitive_data

        text = (
            "For skin problems, you should see a dermatologist. "
            "At MediTrust Hospital, Dr. Nidhi Gopani (MBBS, MD Dermatology) is available "
            "11:00 AM to 7:00 PM. Phone: [HOSPITAL_CONTACT]."
        )
        validate_no_sensitive_data(text, context="eval_025.expected_answer")


# ---------------------------------------------------------------------------
# 8. Full-dataset upload dry-run — all 30 cases must pass the PHI validator
# ---------------------------------------------------------------------------

class TestDatasetCleanliness:
    """Dataset scan: all cases must pass except the deliberate adversarial test case.

    ``eval_023`` contains a real SSN (``123-45-6789``) and a patient name in its
    question field — it is an intentional adversarial case that must remain
    blocked by the PHI validator.  All other 29 cases must pass.
    """

    # eval_023 is the one case that intentionally contains real PHI (SSN + patient name).
    # It must stay blocked so the validator's true-positive coverage is preserved.
    INTENTIONALLY_BLOCKED = {"eval_023"}

    def test_full_dataset_no_phi_blocks(self):
        import json
        from pathlib import Path
        from app.ai.evaluation.langsmith_utils import (
            validate_dict_no_sensitive_data,
            SanitizationError,
        )

        dataset_path = (
            Path(__file__).resolve().parent.parent
            / "app" / "ai" / "evaluation" / "dataset.jsonl"
        )
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"

        unexpected_failures = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                if case["id"] in self.INTENTIONALLY_BLOCKED:
                    # Verify it IS still blocked (true positive preserved).
                    payload = {
                        "question": case.get("question", ""),
                        "expected_answer": case.get("expected_answer", ""),
                        "reference_context": case.get("reference_context", ""),
                    }
                    try:
                        validate_dict_no_sensitive_data(payload, context=f"case_{case['id']}")
                        # If we reach here, the adversarial case is no longer blocked — that's wrong.
                        unexpected_failures.append(
                            (case["id"], "SHOULD be blocked (adversarial PHI) but was not")
                        )
                    except SanitizationError:
                        pass  # Expected — adversarial case correctly blocked
                    continue

                payload = {
                    "question": case.get("question", ""),
                    "expected_answer": case.get("expected_answer", ""),
                    "reference_context": case.get("reference_context", ""),
                    "tags": case.get("tags", []),
                    "metadata": case.get("metadata", {}),
                }
                try:
                    validate_dict_no_sensitive_data(payload, context=f"case_{case['id']}")
                except SanitizationError as exc:
                    unexpected_failures.append((case["id"], str(exc)))

        assert unexpected_failures == [], (
            f"{len(unexpected_failures)} case(s) unexpectedly blocked:\n"
            + "\n".join(f"  {cid}: {msg}" for cid, msg in unexpected_failures)
        )

    def test_adversarial_case_eval_023_still_blocked(self):
        """eval_023 must stay blocked — it contains a real SSN and patient name."""
        from app.ai.evaluation.langsmith_utils import (
            validate_no_sensitive_data,
            SanitizationError,
        )

        question = "My patient John Doe has SSN 123-45-6789 and needs insulin dosage"
        with pytest.raises(SanitizationError):
            validate_no_sensitive_data(question, context="eval_023.question")
