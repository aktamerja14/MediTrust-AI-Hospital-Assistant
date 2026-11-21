"""Evaluation dataset schema for MediTrust RAG evaluation.

Defines the structure for evaluation cases used in offline RAG evaluation.
All cases are synthetic - no real patient data or PHI.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class EvaluationCase:
    """Single evaluation case for RAG system assessment.

    Attributes:
        id: Unique identifier for the case
        question: User question to evaluate
        expected_answer: Expected/ground truth answer (from verified hospital docs)
        reference_context: Source document excerpts used as ground truth
        category: Category of the question
        language: Language of the question (en, gu, hi, etc.)
        tags: Additional tags for filtering/analysis
        needs_human_verification: Whether medical facts need human verification
        created_at: Timestamp when case was created
        metadata: Additional metadata
    """
    id: str
    question: str
    expected_answer: str
    reference_context: str
    category: str
    language: str = "en"
    tags: List[str] = field(default_factory=list)
    needs_human_verification: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "reference_context": self.reference_context,
            "category": self.category,
            "language": self.language,
            "tags": self.tags,
            "needs_human_verification": self.needs_human_verification,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCase":
        """Create from dictionary."""
        return cls(**data)

    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# Valid categories for evaluation cases
VALID_CATEGORIES = frozenset({
    "general_info",
    "hospital_faq",
    "medical_safety",
    "greetings",
    "multilingual",
    "edge_cases",
})

# Valid languages
VALID_LANGUAGES = frozenset({"en", "gu", "hi", "mr", "ta", "te", "bn"})

# Maximum field lengths
MAX_QUESTION_LENGTH = 2000
MAX_ANSWER_LENGTH = 5000
MAX_CONTEXT_LENGTH = 10000


def validate_case(case: EvaluationCase) -> List[str]:
    """Validate an evaluation case.

    Args:
        case: Evaluation case to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not case.id:
        errors.append("id is required")

    # Allow empty questions for edge_cases category (e.g., empty query testing)
    if not case.question or not case.question.strip():
        if case.category != "edge_cases":
            errors.append("question is required")
    elif len(case.question) > MAX_QUESTION_LENGTH:
        errors.append(f"question exceeds max length of {MAX_QUESTION_LENGTH}")

    if not case.expected_answer or not case.expected_answer.strip():
        errors.append("expected_answer is required")
    elif len(case.expected_answer) > MAX_ANSWER_LENGTH:
        errors.append(f"expected_answer exceeds max length of {MAX_ANSWER_LENGTH}")

    if not case.reference_context or not case.reference_context.strip():
        errors.append("reference_context is required")
    elif len(case.reference_context) > MAX_CONTEXT_LENGTH:
        errors.append(f"reference_context exceeds max length of {MAX_CONTEXT_LENGTH}")

    if case.category not in VALID_CATEGORIES:
        errors.append(f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")

    if case.language not in VALID_LANGUAGES:
        errors.append(f"language must be one of: {', '.join(sorted(VALID_LANGUAGES))}")

    if not isinstance(case.tags, list):
        errors.append("tags must be a list")

    if not isinstance(case.needs_human_verification, bool):
        errors.append("needs_human_verification must be a boolean")

    return errors


def load_dataset(filepath: str) -> List[EvaluationCase]:
    """Load evaluation dataset from JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of EvaluationCase objects

    Raises:
        ValueError: If any case is invalid
    """
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                case = EvaluationCase.from_dict(data)
                errors = validate_case(case)
                if errors:
                    raise ValueError(f"Line {line_num}: {', '.join(errors)}")
                cases.append(case)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON - {e}")
    return cases


def save_dataset(cases: List[EvaluationCase], filepath: str) -> None:
    """Save evaluation dataset to JSONL file.

    Args:
        cases: List of evaluation cases
        filepath: Path to output JSONL file
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for case in cases:
            errors = validate_case(case)
            if errors:
                raise ValueError(f"Invalid case {case.id}: {', '.join(errors)}")
            f.write(case.to_jsonl() + "\n")


def get_dataset_stats(cases: List[EvaluationCase]) -> Dict[str, Any]:
    """Get statistics about the dataset.

    Args:
        cases: List of evaluation cases

    Returns:
        Dictionary with dataset statistics
    """
    if not cases:
        return {"total": 0}

    categories = {}
    languages = {}
    verification_needed = 0

    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
        languages[case.language] = languages.get(case.language, 0) + 1
        if case.needs_human_verification:
            verification_needed += 1

    return {
        "total": len(cases),
        "categories": categories,
        "languages": languages,
        "needs_human_verification": verification_needed,
    }