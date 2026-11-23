"""LangSmith dataset upload utilities for MediTrust evaluation.

Provides functions to upload evaluation datasets to LangSmith.
Does NOT auto-upload - requires explicit invocation.

Phase 6.2: Added PHI/sensitive data sanitization for all uploads.
Fail-closed behavior: rejects upload if sensitive data cannot be safely sanitized.

LIMITATIONS: Detection is pattern-based (regex) and cannot guarantee complete
PHI detection. It does not cover all identifier formats, free-text names,
context-dependent PHI, or non-English scripts. This is NOT a HIPAA compliance
mechanism; use synthetic/de-identified data and human review for uploads.
"""
import os
import logging
import re
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


# =============================================================================
# PHI/Sensitive Data Sanitization Layer
# =============================================================================

# =============================================================================
# Sensitive-data detection patterns
# =============================================================================
#
# Design principle: every pattern targets *patient-linked* PHI, not general
# medical vocabulary.  Hospital-assistant evaluation datasets legitimately
# contain medical terms (diabetes, insulin, cancer) in education and FAQ
# content.  A term is only PHI when it is associated with an identifiable
# individual in the same string.
#
# Patterns marked with sentinel value ``None`` are implemented as functions
# (see below) because Python ``re`` does not support variable-width lookbehinds.

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "Phone"),
    (re.compile(r"\b\d{10}\b"), "Phone"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "Email"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"), "Date"),
    (re.compile(r"\b(MRN|MRN#|Medical Record)\s*:?\s*\d+\b", re.IGNORECASE), "MRN"),
    (re.compile(r"\b(DOB|Date of Birth)\s*:?\s*\d{1,2}/\d{1,2}/\d{4}\b", re.IGNORECASE), "DOB"),
    # PatientName: "Patient:" or "Name:" followed by two title-case words that
    # look like a personal name.  Excludes document titles such as
    # "Patient Education Guide" by requiring neither word is a known
    # document/title term.
    (None, "PatientName"),
]

MEDICAL_SENSITIVE_PATTERNS = [
    # Diagnosis and Medication: only flag when the clinical keyword co-occurs
    # with a personal identifier in the same string.  General educational
    # content ("diabetes education", "insulin therapy overview") is NOT PHI.
    (None, "Diagnosis"),
    (re.compile(r"\b(take|prescribe|dosage|mg|ml|tablet|capsule)\s+\d+", re.IGNORECASE), "MedicationDosage"),
    (None, "Medication"),
]

# ---------------------------------------------------------------------------
# Helper patterns for the function-based checks
# ---------------------------------------------------------------------------

# Clinical/diagnosis keywords
_DIAGNOSIS_KEYWORDS = re.compile(
    r"\b(cancer|diabetes|hypertension|HIV|AIDS|depression|anxiety)\b", re.IGNORECASE
)

# Medication keywords that are PHI when linked to a patient
_MEDICATION_KEYWORDS = re.compile(
    r"\b(insulin|metformin|warfarin|chemotherapy|radiation)\b", re.IGNORECASE
)

# Explicit patient-label with colon: "Patient:" or "Name:"
_PATIENT_LABEL = re.compile(r"\b(Patient\s*:|Name\s*:)", re.IGNORECASE)

# Clinical identifiers on their own
_CLINICAL_ID = re.compile(r"\b(MRN|DOB)\b", re.IGNORECASE)

# Personal titles followed by an upper-case initial: Mr./Mrs./Ms. S...
_PERSONAL_TITLE = re.compile(r"\b(Mr\.|Mrs\.|Ms\.)\s+[A-Z]")

# Non-name title-case words that frequently appear in medical document headers
# and should NOT be treated as parts of a person's name.
_NON_NAME_WORDS = frozenset({
    # Document / section structure
    "Patient", "Education", "Guide", "Management", "Treatment", "Prevention",
    "Symptoms", "Diagnosis", "Complications", "Information", "Risk",
    "Factors", "Overview", "Summary", "Report", "Reference", "Context",
    "Lifestyle", "Modifications", "Cornerstone", "Considerations",
    # Medical conditions
    "Mellitus", "Type", "Diabetes", "Cancer", "Hypertension", "Obesity",
    "Ketoacidosis", "Hyperglycemia", "Hypoglycemia", "Neuropathy",
    "Retinopathy", "Nephropathy", "Gestational", "Autoimmune",
    # Healthcare institutions / roles
    "Doctor", "Hospital", "Medical", "Health", "Care", "Service",
    "Department", "Center", "Centre", "Clinic", "Institute", "Trust",
    "Physician", "Specialist", "Oncologist", "Cardiologist", "Endocrinologist",
    # Drug / therapy terms
    "Therapy", "Medication", "Insulin", "Metformin", "Warfarin",
    "Sulfonylureas", "Inhibitors", "Agonists", "Receptors",
    # Demographic / ethnic descriptors — not personal identifiers
    "African", "American", "Native", "Asian", "Hispanic", "Latino",
    "Latina", "Pacific", "Islander", "Indian", "Alaska",
    # Measurement / lab terms
    "Blood", "Glucose", "Pressure", "Sugar", "Level", "Test",
    "Plasma", "Fasting", "Random", "Cholesterol", "Triglycerides",
    # Miscellaneous common section words
    "General", "Special", "Senior", "Junior", "First", "Second",
    "Third", "Primary", "Secondary", "High", "Low", "Normal",
    "Physical", "Mental", "Personal", "Family", "Social",
})

# Two consecutive title-case words that are BOTH outside _NON_NAME_WORDS —
# a good proxy for a person's full name (e.g. "John Doe", "Alice Brown").
_PROPER_NAME = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")


def _has_person_identifier(text: str) -> bool:
    """Return True if *text* contains a personal-identifier signal.

    Signals:
    - Explicit label: "Patient:" or "Name:"
    - Clinical ID: MRN or DOB
    - Personal title: Mr./Mrs./Ms. <Initial>
    - Two consecutive title-case words that are not medical/document terms
      (i.e. a person's full name such as "John Doe")
    """
    if _PATIENT_LABEL.search(text):
        return True
    if _CLINICAL_ID.search(text):
        return True
    if _PERSONAL_TITLE.search(text):
        return True
    for m in _PROPER_NAME.finditer(text):
        w1, w2 = m.group(1), m.group(2)
        if w1 not in _NON_NAME_WORDS and w2 not in _NON_NAME_WORDS:
            return True
    return False


def _has_patient_name(text: str) -> List[Dict[str, Any]]:
    """Return PatientName findings only when "Patient:" or "Name:" precedes
    a proper two-word name that is not a document title.

    "Patient Education Guide" → NOT a hit (Education/Guide are non-name words).
    "Patient: Jane Smith" → hit.
    "Name: Bob Jones" → hit.
    """
    findings = []
    # Require the explicit colon label before the name
    for m in re.finditer(
        r"\b(Patient|Name)\s*:\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text
    ):
        w1, w2 = m.group(2), m.group(3)
        if w1 not in _NON_NAME_WORDS and w2 not in _NON_NAME_WORDS:
            findings.append({
                "pattern_type": "PatientName",
                "matched_text": m.group(),
                "start": m.start(),
                "end": m.end(),
            })
    return findings


def _has_patient_linked_diagnosis(text: str) -> List[Dict[str, Any]]:
    """Return Diagnosis findings only when a personal identifier co-occurs.

    General clinical text ("diabetes education materials", "cancer treatment
    options") returns no findings.  Patient-linked text ("John Doe has
    diabetes", "Patient: Jane Smith — hypertension") returns findings.
    """
    if not _DIAGNOSIS_KEYWORDS.search(text):
        return []
    if not _has_person_identifier(text):
        return []
    findings = []
    for m in _DIAGNOSIS_KEYWORDS.finditer(text):
        findings.append({
            "pattern_type": "Diagnosis",
            "matched_text": m.group(),
            "start": m.start(),
            "end": m.end(),
        })
    return findings


def _has_patient_linked_medication(text: str) -> List[Dict[str, Any]]:
    """Return Medication findings only when a personal identifier co-occurs.

    General clinical text ("insulin therapy for Type 1 diabetes", "metformin
    is first-line therapy") returns no findings.  Patient-linked text
    ("John Doe takes insulin", "prescribe metformin to Mr. Smith") returns
    findings.
    """
    if not _MEDICATION_KEYWORDS.search(text):
        return []
    if not _has_person_identifier(text):
        return []
    findings = []
    for m in _MEDICATION_KEYWORDS.finditer(text):
        findings.append({
            "pattern_type": "Medication",
            "matched_text": m.group(),
            "start": m.start(),
            "end": m.end(),
        })
    return findings


ALL_SENSITIVE_PATTERNS = PII_PATTERNS + MEDICAL_SENSITIVE_PATTERNS

# Additional patterns used only while sanitizing (not for fail-closed detection).
# Free-text person names are high-false-positive for detection, but over-redaction
# is safe for anything that is uploaded as LangSmith feedback.
SANITIZE_ONLY_PATTERNS = [
    (re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"), "PersonName"),
]

SANITIZE_PATTERNS = ALL_SENSITIVE_PATTERNS + SANITIZE_ONLY_PATTERNS


def detect_sensitive_data(text: str) -> List[Dict[str, Any]]:
    """Detect sensitive data in text.

    Returns:
        List of dicts with keys: pattern_type, matched_text, start, end
    """
    if not text or not isinstance(text, str):
        return []

    findings = []
    for pattern, pattern_type in ALL_SENSITIVE_PATTERNS:
        if pattern is None:
            if pattern_type == "PatientName":
                findings.extend(_has_patient_name(text))
            elif pattern_type == "Diagnosis":
                findings.extend(_has_patient_linked_diagnosis(text))
            elif pattern_type == "Medication":
                findings.extend(_has_patient_linked_medication(text))
        else:
            for match in pattern.finditer(text):
                findings.append({
                    "pattern_type": pattern_type,
                    "matched_text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
    return findings


def sanitize_text(text: str, replacement: str = "[REDACTED]") -> str:
    """Sanitize text by replacing sensitive patterns.

    Note: This is a best-effort sanitization. For PHI protection,
    the fail-closed approach should reject data that cannot be
    confidently sanitized rather than relying solely on pattern replacement.
    """
    if not text or not isinstance(text, str):
        return text

    sanitized = text
    for pattern, pattern_type in SANITIZE_PATTERNS:
        if pattern is None:
            # Sentinel entries: apply the keyword pattern directly for defence-in-depth.
            if pattern_type in ("Diagnosis", "PatientName"):
                sanitized = _DIAGNOSIS_KEYWORDS.sub(replacement, sanitized)
            elif pattern_type == "Medication":
                sanitized = _MEDICATION_KEYWORDS.sub(replacement, sanitized)
        else:
            sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_dict(obj: Dict[str, Any], fields_to_sanitize: Optional[List[str]] = None) -> Dict[str, Any]:
    """Recursively sanitize string values in a dictionary.
    
    Args:
        obj: Dictionary to sanitize
        fields_to_sanitize: Optional list of field names to sanitize. If None, all string values are sanitized.
    
    Returns:
        Sanitized dictionary (new object, original not modified)
    """
    if not isinstance(obj, dict):
        return obj
    
    result = {}
    for key, value in obj.items():
        if fields_to_sanitize is not None and key not in fields_to_sanitize:
            result[key] = value
            continue
        
        if isinstance(value, str):
            result[key] = sanitize_text(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value, fields_to_sanitize)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(item, fields_to_sanitize) if isinstance(item, dict)
                else sanitize_text(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


class SanitizationError(Exception):
    """Raised when sanitization cannot guarantee PHI removal."""
    pass


def validate_no_sensitive_data(text: str, context: str = "") -> None:
    """Validate that text contains no sensitive data.
    
    Fail-closed: raises SanitizationError if sensitive data is detected.
    This ensures we never silently upload PHI.
    
    Args:
        text: Text to validate
        context: Context description for error messages
    
    Raises:
        SanitizationError: If sensitive data is detected
    """
    findings = detect_sensitive_data(text)
    if findings:
        pattern_types = set(f["pattern_type"] for f in findings)
        raise SanitizationError(
            f"Sensitive data detected in {context}: {', '.join(pattern_types)}. "
            f"Found {len(findings)} occurrence(s). Data must be sanitized before upload."
        )


def validate_dict_no_sensitive_data(obj: Dict[str, Any], context: str = "") -> None:
    """Recursively validate dictionary for sensitive data.
    
    Args:
        obj: Dictionary to validate
        context: Context description for error messages
    
    Raises:
            SanitizationError: If sensitive data is detected
    """
    if not isinstance(obj, dict):
        return
    
    for key, value in obj.items():
        if isinstance(value, str):
            validate_no_sensitive_data(value, f"{context}.{key}")
        elif isinstance(value, dict):
            validate_dict_no_sensitive_data(value, f"{context}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    validate_no_sensitive_data(item, f"{context}.{key}[{i}]")
                elif isinstance(item, dict):
                    validate_dict_no_sensitive_data(item, f"{context}.{key}[{i}]")


def _sanitize_case_for_upload(case: EvaluationCase) -> EvaluationCase:
    """Create a sanitized copy of an EvaluationCase for LangSmith upload.
    
    Applies sanitization to question, expected_answer, and reference_context.
    Raises SanitizationError if sensitive data is detected that cannot be safely handled.
    
    Args:
        case: Original evaluation case
        
    Returns:
        Sanitized EvaluationCase
        
    Raises:
        SanitizationError: If sensitive data cannot be safely sanitized
    """
    # Validate that sensitive data is not present (fail-closed)
    # We check the original data before sanitization
    validate_dict_no_sensitive_data({
        "question": case.question,
        "expected_answer": case.expected_answer,
        "reference_context": case.reference_context,
        "tags": case.tags,
        "metadata": case.metadata or {},
    }, f"case_{case.id}")
    
    # If validation passes, create sanitized copy (defense in depth)
    return EvaluationCase(
        id=case.id,
        question=sanitize_text(case.question),
        expected_answer=sanitize_text(case.expected_answer),
        reference_context=sanitize_text(case.reference_context),
        category=case.category,
        language=case.language,
        tags=case.tags,
        needs_human_verification=case.needs_human_verification,
        created_at=case.created_at,
        metadata=sanitize_dict(case.metadata) if case.metadata else {},
    )


# Default dataset name in LangSmith
DEFAULT_DATASET_NAME = "meditrust-rag-evaluation-v1"


class LangSmithUploadError(Exception):
    """Exception for LangSmith upload errors."""
    pass


class SanitizationUploadError(LangSmithUploadError):
    """Exception raised when dataset upload is blocked due to sensitive data."""
    pass


def create_langsmith_client(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
) -> Client:
    """Create a LangSmith client scoped to a project.

    Args:
        api_key: Optional API key.  Falls back to ``LANGSMITH_API_KEY`` env var.
        project: Optional project name.  Falls back to ``LANGSMITH_PROJECT`` env var,
                 then to the ``settings.LANGSMITH_PROJECT`` default.  When supplied,
                 the value is written to ``os.environ["LANGSMITH_PROJECT"]`` so the
                 LangSmith SDK picks it up for all tracing calls in this process.

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

    # Resolve project name: explicit arg > env var > settings default.
    # We deliberately do NOT log the api_key — only the project name.
    resolved_project = project or os.getenv("LANGSMITH_PROJECT")
    if not resolved_project:
        try:
            from app.core.config import settings  # lazy import to avoid circular
            resolved_project = settings.LANGSMITH_PROJECT
        except Exception:
            resolved_project = "meditrust-rag-evaluation"

    # Inject into os.environ so the LangSmith SDK's aevaluate / tracing picks it up.
    os.environ["LANGSMITH_PROJECT"] = resolved_project
    logger.debug("LangSmith project set to '%s'", resolved_project)

    return Client(api_key=key)


# ---------------------------------------------------------------------------
# Experiment-output sanitization
# ---------------------------------------------------------------------------

class ExperimentSanitizationError(LangSmithUploadError):
    """Raised when PHI/PII is detected in an experiment output before transmission."""
    pass


def sanitize_experiment_payload(
    payload: Dict[str, Any],
    case_id: str,
    *,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Sanitize an experiment trace payload before it reaches LangSmith.

    Fail-closed: validates the **original** payload first.  If any sensitive
    pattern is found, raises ``ExperimentSanitizationError`` immediately without
    producing a sanitized output — no partial data is transmitted.  The error
    message contains only the ``case_id`` and matched rule types; never the
    raw matched text.

    If validation passes (the data is already clean), applies ``sanitize_text``
    as a defence-in-depth pass and returns the sanitized copy.

    Args:
        payload: Dict representing a trace inputs/outputs/feedback record.
        case_id: Identifier used in redacted log/error messages only.
        fields: Optional allowlist of field names to validate/sanitize.  ``None`` → all.

    Returns:
        A sanitized copy of *payload*.

    Raises:
        ExperimentSanitizationError: If sensitive data is detected in the original payload.
    """
    # Step 1 — validate original (fail-closed: reject before any sanitization).
    try:
        validate_dict_no_sensitive_data(payload, context=f"experiment_payload[{case_id}]")
    except SanitizationError as exc:
        matched_rules = str(exc).split(": ", 2)[-1].split(".")[0]
        logger.error(
            "PHI detected in experiment payload — blocking transmission. "
            "case_id=%s, rules=%s",
            case_id, matched_rules,
        )
        raise ExperimentSanitizationError(
            f"Experiment payload for case '{case_id}' contains sensitive data "
            f"({matched_rules}). Transmission blocked."
        ) from exc

    # Step 2 — defence-in-depth: sanitize the now-validated payload.
    return sanitize_dict(payload, fields_to_sanitize=fields)


def assert_safe_environment(*, allow_unsafe: bool = False) -> None:
    """Block ``--no-sanitize`` outside development environments.

    Args:
        allow_unsafe: ``True`` only when the caller passed ``--allow-unsafe-local``
                      explicitly on the CLI.

    Raises:
        RuntimeError: When unsafe mode is requested outside a development env.
    """
    if not allow_unsafe:
        return

    env = os.getenv("ENVIRONMENT", "")
    if not env:
        try:
            from app.core.config import settings
            env = settings.ENVIRONMENT
        except Exception:
            env = "unknown"

    if env.lower() not in {"development", "dev", "local"}:
        raise RuntimeError(
            f"--allow-unsafe-local (PHI sanitization disabled) is not permitted "
            f"in environment '{env}'. Only 'development' environments may skip "
            "sanitization. Set ENVIRONMENT=development in backend/.env to run "
            "locally without sanitization."
        )


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


def _sanitize_case_for_upload(case: EvaluationCase) -> EvaluationCase:
    """Create a sanitized copy of an EvaluationCase for LangSmith upload.
    
    Applies sanitization to question, expected_answer, and reference_context.
    Raises SanitizationError if sensitive data is detected that cannot be safely handled.
    
    Args:
        case: Original evaluation case
        
    Returns:
        Sanitized EvaluationCase
        
    Raises:
        SanitizationError: If sensitive data cannot be safely sanitized
    """
    # Validate that sensitive data is not present (fail-closed)
    # We check the original data before sanitization
    validate_dict_no_sensitive_data({
        "question": case.question,
        "expected_answer": case.expected_answer,
        "reference_context": case.reference_context,
        "tags": case.tags,
        "metadata": case.metadata or {},
    }, f"case_{case.id}")
    
    # If validation passes, create sanitized copy (defense in depth)
    return EvaluationCase(
        id=case.id,
        question=sanitize_text(case.question),
        expected_answer=sanitize_text(case.expected_answer),
        reference_context=sanitize_text(case.reference_context),
        category=case.category,
        language=case.language,
        tags=case.tags,
        needs_human_verification=case.needs_human_verification,
        created_at=case.created_at,
        metadata=sanitize_dict(case.metadata) if case.metadata else {},
    )


def upload_evaluation_cases(
    client: Client,
    dataset_name: str,
    cases: List[EvaluationCase],
    batch_size: int = 50,
    sanitize: bool = True,
) -> Dict[str, Any]:
    """Upload evaluation cases to LangSmith dataset.

    Converts EvaluationCase objects to LangSmith Examples with:
    - inputs: {"question": ..., "category": ..., "language": ..., "tags": ...}
    - outputs: {"expected_answer": ..., "reference_context": ..., "needs_human_verification": ...}

    Phase 6.2: Applies PHI/sensitive data sanitization before upload (fail-closed).

    Args:
        client: LangSmith client
        dataset_name: Target dataset name
        cases: List of EvaluationCase objects
        batch_size: Number of examples per batch
        sanitize: Whether to apply PHI sanitization (default: True, fail-closed)

    Returns:
        Dict with upload statistics

    Raises:
        SanitizationUploadError: If sensitive data is detected and cannot be safely handled
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

    # Apply sanitization if enabled (fail-closed)
    sanitized_cases = []
    sanitization_errors = []
    for case in valid_cases:
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

    # Convert to LangSmith Examples
    examples = []
    for case in sanitized_cases:
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
            client.create_examples(dataset_name=dataset_name, examples=[dict(e) for e in batch])
            uploaded += len(batch)
            logger.info("Uploaded batch %d-%d (%d examples)", i, i + len(batch), len(batch))
        except Exception as e:
            logger.exception("Failed to upload batch starting at %d: %s", i, e)
            upload_errors.append({"batch_start": i, "error": str(e)})

    return {
        "uploaded": uploaded,
        "skipped": len(errors) + len(sanitization_errors),
        "validation_errors": errors,
        "sanitization_errors": sanitization_errors,
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
    sanitize: bool = True,
) -> Dict[str, Any]:
    """Complete workflow: load dataset from JSONL and upload to LangSmith.

    This is the main entry point for uploading evaluation datasets.
    Does NOT run automatically - must be explicitly called.

    Phase 6.2: Applies PHI/sensitive data sanitization before upload (fail-closed).

    Args:
        filepath: Path to JSONL dataset file
        dataset_name: Target LangSmith dataset name
        api_key: Optional LangSmith API key
        description: Dataset description
        metadata: Additional dataset metadata
        create_if_missing: Create dataset if it doesn't exist
        sanitize: Whether to apply PHI sanitization (default: True, fail-closed)

    Returns:
        Dict with upload results and statistics

    Raises:
        LangSmithUploadError: If upload fails
        SanitizationUploadError: If sensitive data is detected and cannot be safely handled
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

    # Upload cases with sanitization
    result = upload_evaluation_cases(client, dataset_name, cases, sanitize=sanitize)
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