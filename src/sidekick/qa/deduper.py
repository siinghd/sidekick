"""QA pair deduplication.

Optimized for:
- Pre-compiled regex patterns
- Cached tokenization to avoid O(n²) redundant computation
- frozenset for O(1) stopword lookups
"""

import re
from collections.abc import Sequence

from sidekick.core.models import QAPair

# Pre-compile regex for normalize_text (avoids recompilation on each call)
_PUNCT_PATTERN = re.compile(r"[^\w\s']")

# Use frozenset for O(1) membership testing
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "this", "that", "these", "those", "what",
    "which", "who", "whom", "whose", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
    "its", "itself", "they", "them", "their", "theirs", "themselves",
    "person", "user", "individual",
})


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized lowercase text with minimal punctuation
    """
    # Lowercase and remove punctuation in one pass
    text = _PUNCT_PATTERN.sub(" ", text.lower())
    # Normalize whitespace
    return " ".join(text.split())


def get_question_tokens(question: str) -> frozenset[str]:
    """Get significant tokens from a question.

    Args:
        question: Question text

    Returns:
        Frozenset of significant tokens (excludes common words)
    """
    normalized = normalize_text(question)
    tokens = set(normalized.split())
    return frozenset(tokens - _STOPWORDS)


def calculate_similarity(q1: str, q2: str) -> float:
    """Calculate similarity between two questions using Jaccard index.

    Args:
        q1: First question
        q2: Second question

    Returns:
        Similarity score between 0 and 1
    """
    tokens1 = get_question_tokens(q1)
    tokens2 = get_question_tokens(q2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def _calculate_similarity_from_tokens(
    tokens1: frozenset[str],
    tokens2: frozenset[str],
) -> float:
    """Calculate Jaccard similarity from pre-computed token sets.

    Args:
        tokens1: First token set
        tokens2: Second token set

    Returns:
        Similarity score between 0 and 1
    """
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def find_duplicates(
    qa_pairs: Sequence[QAPair],
    threshold: float = 0.7,
) -> list[tuple[int, int]]:
    """Find duplicate question pairs.

    Optimized: Pre-computes tokens to avoid O(n²) tokenization.

    Args:
        qa_pairs: List of QA pairs to check
        threshold: Similarity threshold for duplicates

    Returns:
        List of (index1, index2) tuples for duplicate pairs
    """
    if not qa_pairs:
        return []

    # Pre-compute all tokens once (O(n) instead of O(n²))
    token_cache = [get_question_tokens(qa.question) for qa in qa_pairs]

    duplicates: list[tuple[int, int]] = []

    for i in range(len(qa_pairs)):
        tokens_i = token_cache[i]
        for j in range(i + 1, len(qa_pairs)):
            similarity = _calculate_similarity_from_tokens(tokens_i, token_cache[j])
            if similarity >= threshold:
                duplicates.append((i, j))

    return duplicates


def deduplicate(
    qa_pairs: list[QAPair],
    threshold: float = 0.7,
) -> list[QAPair]:
    """Remove duplicate QA pairs, keeping the first occurrence.

    Args:
        qa_pairs: List of QA pairs
        threshold: Similarity threshold for duplicates

    Returns:
        Deduplicated list of QA pairs
    """
    if not qa_pairs:
        return []

    duplicates = find_duplicates(qa_pairs, threshold)

    # Build set of indices to remove (keep lower index, remove higher)
    to_remove: set[int] = set()
    for i, j in duplicates:
        to_remove.add(j)

    return [qa for idx, qa in enumerate(qa_pairs) if idx not in to_remove]


def deduplicate_against_existing(
    new_pairs: list[QAPair],
    existing_pairs: Sequence[QAPair],
    threshold: float = 0.7,
) -> list[QAPair]:
    """Remove new QA pairs that are duplicates of existing ones.

    Optimized: Pre-computes tokens to avoid O(n*m) tokenization.

    Args:
        new_pairs: New QA pairs to filter
        existing_pairs: Existing QA pairs to check against
        threshold: Similarity threshold

    Returns:
        New pairs that are not duplicates of existing
    """
    if not new_pairs:
        return []

    if not existing_pairs:
        return list(new_pairs)

    # Pre-compute tokens for existing pairs once
    existing_tokens = [get_question_tokens(qa.question) for qa in existing_pairs]

    result: list[QAPair] = []

    for new_qa in new_pairs:
        new_tokens = get_question_tokens(new_qa.question)
        is_duplicate = False

        for existing_tok in existing_tokens:
            similarity = _calculate_similarity_from_tokens(new_tokens, existing_tok)
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(new_qa)

    return result
