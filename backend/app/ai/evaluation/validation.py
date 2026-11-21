"""Enhanced dataset validation for MediTrust RAG evaluation.

Provides comprehensive validation beyond schema checking:
- Duplicate ID detection
- Content quality validation
- Reference context support verification
- Medical safety claim detection
- Human verification flagging
- Structured validation reports
"""
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import Counter

from app.ai.evaluation.dataset_schema import (
    EvaluationCase,
    VALID_CATEGORIES,
    VALID_LANGUAGES,
    validate_case,
    load_dataset,
    get_dataset_stats,
)


@dataclass
class ValidationIssue:
    """Single validation issue found during dataset validation."""
    case_id: str
    severity: str  # "error", "warning", "info"
    category: str  # "schema", "content", "medical_safety", "human_verification", "duplicate"
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Structured validation report for a dataset."""
    total_cases: int = 0
    valid_cases: int = 0
    invalid_cases: int = 0
    duplicate_ids: List[str] = field(default_factory=list)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    language_distribution: Dict[str, int] = field(default_factory=dict)
    missing_fields: Dict[str, List[str]] = field(default_factory=dict)
    human_verification_required: List[str] = field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = field(default_factory=list)
    schema_errors: List[ValidationIssue] = field(default_factory=list)
    content_warnings: List[ValidationIssue] = field(default_factory=list)
    medical_safety_flags: List[ValidationIssue] = field(default_factory=list)
    verification_flags: List[ValidationIssue] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (len(self.schema_errors) + len(self.content_warnings) +
                len(self.medical_safety_flags) + len(self.verification_flags))

    @property
    def is_valid(self) -> bool:
        return self.invalid_cases == 0 and len(self.duplicate_ids) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cases": self.total_cases,
            "valid_cases": self.valid_cases,
            "invalid_cases": self.invalid_cases,
            "duplicate_ids": self.duplicate_ids,
            "category_distribution": self.category_distribution,
            "language_distribution": self.language_distribution,
            "missing_fields": self.missing_fields,
            "human_verification_required": self.human_verification_required,
            "unsupported_claims": self.unsupported_claims,
            "schema_errors": [
                {
                    "case_id": i.case_id,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "field": i.field,
                    "suggestion": i.suggestion,
                }
                for i in self.schema_errors
            ],
            "content_warnings": [
                {
                    "case_id": i.case_id,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "field": i.field,
                    "suggestion": i.suggestion,
                }
                for i in self.content_warnings
            ],
            "medical_safety_flags": [
                {
                    "case_id": i.case_id,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "field": i.field,
                    "suggestion": i.suggestion,
                }
                for i in self.medical_safety_flags
            ],
            "verification_flags": [
                {
                    "case_id": i.case_id,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "field": i.field,
                    "suggestion": i.suggestion,
                }
                for i in self.verification_flags
            ],
            "is_valid": self.is_valid,
            "total_issues": self.total_issues,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print a human-readable summary of the validation report."""
        print("\n" + "=" * 60)
        print("MEDITRUST DATASET VALIDATION REPORT")
        print("=" * 60)
        print(f"Total Cases:           {self.total_cases}")
        print(f"Valid Cases:           {self.valid_cases}")
        print(f"Invalid Cases:         {self.invalid_cases}")
        print(f"Duplicate IDs:         {len(self.duplicate_ids)}")
        print(f"Human Verification:    {len(self.human_verification_required)}")
        print(f"Unsupported Claims:    {len(self.unsupported_claims)}")
        print(f"Total Issues:          {self.total_issues}")
        print(f"Overall Status:        {'VALID' if self.is_valid else 'INVALID'}")
        print("-" * 60)

        if self.category_distribution:
            print("\nCategory Distribution:")
            for cat, count in sorted(self.category_distribution.items()):
                print(f"  {cat}: {count}")

        if self.language_distribution:
            print("\nLanguage Distribution:")
            for lang, count in sorted(self.language_distribution.items()):
                print(f"  {lang}: {count}")

        if self.duplicate_ids:
            print(f"\nDuplicate IDs: {', '.join(self.duplicate_ids)}")

        if self.missing_fields:
            print("\nMissing Fields:")
            for field_name, case_ids in self.missing_fields.items():
                print(f"  {field_name}: {len(case_ids)} cases ({', '.join(case_ids[:5])}{'...' if len(case_ids) > 5 else ''})")

        if self.human_verification_required:
            print(f"\nCases Requiring Human Verification ({len(self.human_verification_required)}):")
            for case_id in self.human_verification_required:
                print(f"  - {case_id}")

        if self.unsupported_claims:
            print(f"\nUnsupported/Potentially Hallucinated Claims ({len(self.unsupported_claims)}):")
            for claim in self.unsupported_claims:
                print(f"  - {claim['case_id']}: {claim['claim'][:80]}...")

        if self.schema_errors:
            print(f"\nSchema Errors ({len(self.schema_errors)}):")
            for issue in self.schema_errors[:10]:
                print(f"  [{issue.severity.upper()}] {issue.case_id}: {issue.message}")
            if len(self.schema_errors) > 10:
                print(f"  ... and {len(self.schema_errors) - 10} more")

        if self.content_warnings:
            print(f"\nContent Warnings ({len(self.content_warnings)}):")
            for issue in self.content_warnings[:10]:
                print(f"  [{issue.severity.upper()}] {issue.case_id}: {issue.message}")
            if len(self.content_warnings) > 10:
                print(f"  ... and {len(self.content_warnings) - 10} more")

        if self.medical_safety_flags:
            print(f"\nMedical Safety Flags ({len(self.medical_safety_flags)}):")
            for issue in self.medical_safety_flags[:10]:
                print(f"  [{issue.severity.upper()}] {issue.case_id}: {issue.message}")
            if len(self.medical_safety_flags) > 10:
                print(f"  ... and {len(self.medical_safety_flags) - 10} more")

        if self.verification_flags:
            print(f"\nHuman Verification Flags ({len(self.verification_flags)}):")
            for issue in self.verification_flags[:10]:
                print(f"  [{issue.severity.upper()}] {issue.case_id}: {issue.message}")
            if len(self.verification_flags) > 10:
                print(f"  ... and {len(self.verification_flags) - 10} more")

        print("=" * 60)


# Medical claim patterns for detecting unsupported claims
MEDICAL_CLAIM_PATTERNS = [
    r"\b(cure|cures|cured|permanently eliminate)\b",
    r"\b(take|use)\s+\w+.*?(mg|ml|tablet|capsule|dose|dosage)\b",
    r"\b(take|use)\s+([\d.]+\s*(?:mg|ml|tablet|capsule))",  # e.g., "take 500mg" - capture dosage
    r"\b(you have|you are suffering from|this is|diagnosis is)\b",
    r"\b(recommended dosage|prescribed dose)\b",
    r"\b(stop taking|discontinue)\b",
    r"\b(effective treatment|cure for)\b",
]

# Patterns that indicate medical advice requiring disclaimer
MEDICAL_ADVICE_PATTERNS = [
    r"\b(should take|must take|need to take|recommend taking)\b",
    r"\b(start|begin)\s+\w+\b",
    r"\b(treatment|therapy|medication)\b",
]

# Hospital-specific info that must be verified against approved docs
HOSPITAL_INFO_PATTERNS = [
    r"\b(dr\.|doctor|physician)\b",
    r"\b(appointment|booking|schedule)\b",
    r"\b(fee|fees|cost|price|charge)\b",
    r"\b(phone|contact|number)\b",
    r"\b(timing|hours|available|schedule)\b",
    r"\b(location|address|floor|wing)\b",
    r"\b(service|department|clinic)\b",
]

# Minimum meaningful content length
MIN_QUESTION_LENGTH = 3
MIN_ANSWER_LENGTH = 10
MIN_CONTEXT_LENGTH = 20


def validate_dataset(filepath: str) -> ValidationReport:
    """Run comprehensive validation on an evaluation dataset.

    Args:
        filepath: Path to JSONL dataset file

    Returns:
        ValidationReport with all findings
    """
    report = ValidationReport()

    # Load raw lines for duplicate detection
    raw_cases = []
    seen_ids: Dict[str, int] = {}
    line_num = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_num += 1
            try:
                data = json.loads(line)
                case_id = data.get("id", f"line_{line_num}")
                raw_cases.append((line_num, case_id, data))

                if case_id in seen_ids:
                    report.duplicate_ids.append(case_id)
                else:
                    seen_ids[case_id] = line_num
            except json.JSONDecodeError as e:
                report.schema_errors.append(ValidationIssue(
                    case_id=f"line_{line_num}",
                    severity="error",
                    category="schema",
                    message=f"Invalid JSON: {e}",
                    field="json",
                ))

    # Track field presence
    field_presence: Dict[str, Set[str]] = {field: set() for field in
        ["id", "question", "expected_answer", "reference_context", "category", "language", "tags"]}

    # Validate each case
    valid_count = 0
    invalid_count = 0

    for line_num, case_id, data in raw_cases:
        # Check required fields
        for field_name in field_presence:
            if field_name not in data or not data[field_name]:
                field_presence[field_name].add(case_id)

        # Schema validation
        try:
            case = EvaluationCase.from_dict(data)
            schema_errors = validate_case(case)
            if schema_errors:
                invalid_count += 1
                for error in schema_errors:
                    report.schema_errors.append(ValidationIssue(
                        case_id=case_id,
                        severity="error",
                        category="schema",
                        message=error,
                    ))
                continue
        except Exception as e:
            invalid_count += 1
            report.schema_errors.append(ValidationIssue(
                case_id=case_id,
                severity="error",
                category="schema",
                message=f"Failed to parse case: {e}",
            ))
            continue

        valid_count += 1

        # Content validation
        _validate_content(case, report)

        # Medical safety validation
        _validate_medical_safety(case, report)

        # Human verification validation
        _validate_human_verification(case, report)

        # Unsupported claims detection
        _detect_unsupported_claims(case, report)

    report.total_cases = len(raw_cases)
    report.valid_cases = valid_count
    report.invalid_cases = invalid_count

    # Build distributions from valid cases
    categories = {}
    languages = {}
    verification_needed = []

    for line_num, case_id, data in raw_cases:
        # Only count valid cases for distributions
        try:
            case = EvaluationCase.from_dict(data)
            schema_errors = validate_case(case)
            if not schema_errors:
                categories[case.category] = categories.get(case.category, 0) + 1
                languages[case.language] = languages.get(case.language, 0) + 1
                if case.needs_human_verification:
                    verification_needed.append(case.id)
        except Exception:
            pass

    report.category_distribution = categories
    report.language_distribution = languages

    # Missing fields
    for field_name, case_ids in field_presence.items():
        if case_ids:
            report.missing_fields[field_name] = sorted(case_ids)

    # Human verification required cases
    report.human_verification_required = sorted(verification_needed)

    return report


def _validate_content(case: EvaluationCase, report: ValidationReport) -> None:
    """Validate content quality of a case."""
    # Check question quality
    if len(case.question.strip()) < MIN_QUESTION_LENGTH:
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="warning",
            category="content",
            message=f"Question too short ({len(case.question.strip())} chars)",
            field="question",
            suggestion="Ensure question is meaningful and specific",
        ))

    # Check for placeholder/generic questions
    question_lower = case.question.lower().strip()
    if question_lower in {"test", "test question", "sample", "example", "placeholder"}:
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="warning",
            category="content",
            message="Question appears to be a placeholder",
            field="question",
            suggestion="Replace with real evaluation question",
        ))

    # Check answer quality
    if len(case.expected_answer.strip()) < MIN_ANSWER_LENGTH:
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="warning",
            category="content",
            message=f"Expected answer too short ({len(case.expected_answer.strip())} chars)",
            field="expected_answer",
            suggestion="Provide complete, meaningful answer",
        ))

    # Check context quality
    if len(case.reference_context.strip()) < MIN_CONTEXT_LENGTH:
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="warning",
            category="content",
            message=f"Reference context too short ({len(case.reference_context.strip())} chars)",
            field="reference_context",
            suggestion="Provide sufficient context to support the answer",
        ))

    # Check context supports answer (token overlap)
    overlap = _compute_token_overlap(case.expected_answer, case.reference_context)
    if overlap < 0.1 and case.category != "edge_cases":
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="warning",
            category="content",
            message=f"Low answer-context overlap ({overlap:.2f}) - answer may not be supported by context",
            field="reference_context",
            suggestion="Verify reference context contains information to answer the question",
        ))

    # Language consistency check
    if case.language == "gu" and not _contains_gujarati(case.question):
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="info",
            category="content",
            message="Language marked as Gujarati but question doesn't appear to contain Gujarati script",
            field="language",
            suggestion="Verify language tag matches question content",
        ))
    elif case.language == "hi" and not _contains_devanagari(case.question):
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="info",
            category="content",
            message="Language marked as Hindi but question doesn't appear to contain Devanagari script",
            field="language",
            suggestion="Verify language tag matches question content",
        ))

    # Category consistency checks
    if case.category == "medical_safety" and not _contains_medical_claim(case.question):
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="info",
            category="content",
            message="Category is medical_safety but question doesn't appear to contain medical content",
            field="category",
            suggestion="Verify category matches question content",
        ))

    if case.category == "greetings" and len(case.question.split()) > 5:
        report.content_warnings.append(ValidationIssue(
            case_id=case.id,
            severity="info",
            category="content",
            message="Category is greetings but question is unusually long",
            field="category",
            suggestion="Verify this is a simple greeting",
        ))


def _validate_medical_safety(case: EvaluationCase, report: ValidationReport) -> None:
    """Validate medical safety aspects of a case."""
    # Check expected answer for unsafe patterns
    answer_lower = case.expected_answer.lower()

    # Check for definitive medical claims without disclaimers
    has_medical_claim = _contains_medical_claim(case.expected_answer)
    has_disclaimer = any(word in answer_lower for word in
        ["consult", "disclaimer", "professional", "doctor", "healthcare", "physician", "seek medical"])

    if has_medical_claim and not has_disclaimer:
        # This is expected for some cases - flag for review
        report.medical_safety_flags.append(ValidationIssue(
            case_id=case.id,
            severity="info",
            category="medical_safety",
            message="Expected answer contains medical claims without explicit disclaimer",
            field="expected_answer",
            suggestion="Verify this is intentional (e.g., educational content with implicit disclaimer in context)",
        ))

    # Check for potentially unsafe expected answers
    unsafe_patterns = [
        (r"\b(take|use)\s+\w+.*?(mg|ml|tablet|capsule)\b", "Specific medication dosage in expected answer"),
        (r"\b(cure|cures|cured)\b.*\b(diabetes|hypertension|cancer|chronic)\b", "Cure claim for chronic condition"),
        (r"\b(stop|discontinue)\s+(taking|using)\s+\w+", "Medication discontinuation advice"),
    ]

    for pattern, description in unsafe_patterns:
        if re.search(pattern, answer_lower):
            report.medical_safety_flags.append(ValidationIssue(
                case_id=case.id,
                severity="warning",
                category="medical_safety",
                message=f"Potentially unsafe content in expected answer: {description}",
                field="expected_answer",
                suggestion="Review expected answer for medical safety compliance",
            ))

    # Check reference context for hospital-specific info that needs verification
    context_lower = case.reference_context.lower()
    for pattern in HOSPITAL_INFO_PATTERNS:
        if re.search(pattern, context_lower):
            # Only flag if it's not already marked for verification
            if not case.needs_human_verification:
                report.medical_safety_flags.append(ValidationIssue(
                    case_id=case.id,
                    severity="info",
                    category="medical_safety",
                    message=f"Reference context contains hospital-specific info (matched: {pattern}) - verify against approved documents",
                    field="reference_context",
                    suggestion="Mark needs_human_verification=true if info needs verification",
                ))
            break


def _validate_human_verification(case: EvaluationCase, report: ValidationReport) -> None:
    """Validate human verification flags are appropriate."""
    # Cases that MUST have human verification
    must_verify = False
    reasons = []

    # Medical safety category always needs verification
    if case.category == "medical_safety":
        must_verify = True
        reasons.append("medical_safety category")

    # Hospital info (doctor details, fees, timings, contacts)
    question_lower = case.question.lower()
    answer_lower = case.expected_answer.lower()
    context_lower = case.reference_context.lower()

    hospital_info_keywords = [
        "dr.", "doctor", "physician", "appointment", "fee", "fees", "cost",
        "phone", "contact", "number", "timing", "hours", "available",
        "location", "address", "floor", "service", "department"
    ]

    for keyword in hospital_info_keywords:
        if keyword in question_lower or keyword in answer_lower or keyword in context_lower:
            must_verify = True
            reasons.append(f"hospital info ({keyword})")
            break

    # PHI-related content
    if any(word in question_lower for word in ["patient", "ssn", "mrn", "name", "dob", "insurance"]):
        must_verify = True
        reasons.append("PHI content")

    # Medication-specific advice
    if any(word in answer_lower for word in ["insulin", "metformin", "aspirin", "dosage", "mg daily", "tablet"]):
        must_verify = True
        reasons.append("medication advice")

    if must_verify and not case.needs_human_verification:
        report.verification_flags.append(ValidationIssue(
            case_id=case.id,
            severity="warning",
            category="human_verification",
            message=f"Case requires human verification ({', '.join(reasons)}) but needs_human_verification=false",
            field="needs_human_verification",
            suggestion="Set needs_human_verification=true and have medical professional verify",
        ))
    elif not must_verify and case.needs_human_verification:
        report.verification_flags.append(ValidationIssue(
            case_id=case.id,
            severity="info",
            category="human_verification",
            message="Case marked for human verification but may not require it",
            field="needs_human_verification",
            suggestion="Review if verification is actually needed",
        ))


def _detect_unsupported_claims(case: EvaluationCase, report: ValidationReport) -> None:
    """Detect potentially unsupported or hallucinated medical claims."""
    answer_lower = case.expected_answer.lower()
    context_lower = case.reference_context.lower()

    # Check for medical claims in answer not supported by context
    for pattern in MEDICAL_CLAIM_PATTERNS:
        matches = re.findall(pattern, answer_lower)
        if matches:
            # Check if context supports this claim
            for match in matches:
                claim_text = " ".join(match) if isinstance(match, tuple) else match
                if claim_text not in context_lower:
                    report.unsupported_claims.append({
                        "case_id": case.id,
                        "claim": claim_text[:200],
                        "pattern": pattern,
                        "category": case.category,
                        "needs_verification": True,
                    })

    # Check for specific medical facts that need verification
    specific_claims = [
        (r"\b(\d+)\s*years?\s*experience\b", "Doctor experience years"),
        (r"\b(MBBS|MD|MS|DM|MCh)\b", "Medical qualification"),
        (r"\b(\d{1,2}:\d{2}\s*[AP]M)\s*to\s*(\d{1,2}:\d{2}\s*[AP]M)\b", "Doctor timing"),
        (r"\+91[-\s]?\d{10}", "Phone number"),
        (r"\b(9:00|10:00|11:00|5:00|7:00|7:30)\s*[AP]M\b", "Specific timing"),
    ]

    for pattern, claim_type in specific_claims:
        matches = re.findall(pattern, answer_lower)
        if matches:
            # Check if in context
            for match in matches:
                match_str = " ".join(match) if isinstance(match, tuple) else match
                if match_str not in context_lower:
                    report.unsupported_claims.append({
                        "case_id": case.id,
                        "claim": f"{claim_type}: {match_str}",
                        "pattern": pattern,
                        "category": case.category,
                        "needs_verification": True,
                    })


def _compute_token_overlap(text1: str, text2: str) -> float:
    """Compute token overlap ratio between two texts."""
    tokens1 = set(_normalize_text(text1).split())
    tokens2 = set(_normalize_text(text2).split())

    if not tokens1:
        return 0.0

    overlap = tokens1 & tokens2
    return len(overlap) / len(tokens1)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def _contains_gujarati(text: str) -> bool:
    """Check if text contains Gujarati script."""
    gujarati_range = range(0x0A80, 0x0AFF)
    return any(ord(c) in gujarati_range for c in text)


def _contains_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script."""
    devanagari_range = range(0x0900, 0x097F)
    return any(ord(c) in devanagari_range for c in text)


def _contains_medical_claim(text: str) -> bool:
    """Check if text contains medical claims."""
    medical_patterns = [
        r"\b(diagnos|treat|prescrib|medication|drug|dosage|therapy)\b",
        r"\b(cure|heal|recover|prognosis)\b",
        r"\b(symptom|disease|condition|illness)\b",
        r"\b(insulin|metformin|aspirin|medicine)\b",
        r"\b(blood pressure|glucose|sugar|hba1c)\b",
        r"\b(should take|must take|need to take|recommend taking)\b",
    ]
    text_lower = text.lower()
    for pattern in medical_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def save_validation_report(report: ValidationReport, output_path: str) -> None:
    """Save validation report to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())


def print_validation_report(filepath: str) -> ValidationReport:
    """Convenience function to validate and print report."""
    report = validate_dataset(filepath)
    report.print_summary()
    return report