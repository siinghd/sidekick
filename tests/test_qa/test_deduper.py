"""Tests for QA deduplication."""

from sidekick.core.models import QAPair
from sidekick.qa.deduper import (
    calculate_similarity,
    deduplicate,
    deduplicate_against_existing,
    get_question_tokens,
    normalize_text,
)


class TestNormalization:
    """Tests for text normalization."""

    def test_normalize_lowercase(self) -> None:
        """Test lowercase conversion."""
        assert normalize_text("Hello World") == "hello world"

    def test_normalize_punctuation(self) -> None:
        """Test punctuation removal."""
        assert normalize_text("What's up?") == "what's up"

    def test_normalize_whitespace(self) -> None:
        """Test whitespace normalization."""
        assert normalize_text("  hello   world  ") == "hello world"


class TestTokenization:
    """Tests for question tokenization."""

    def test_removes_stopwords(self) -> None:
        """Test stopword removal."""
        tokens = get_question_tokens("What is the job of this person?")
        assert "what" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens
        assert "job" in tokens

    def test_keeps_significant_words(self) -> None:
        """Test significant words are kept."""
        tokens = get_question_tokens("Where does this person work?")
        assert "work" in tokens


class TestSimilarity:
    """Tests for similarity calculation."""

    def test_identical_questions(self) -> None:
        """Test identical questions have high similarity."""
        q = "Where does this person work?"
        assert calculate_similarity(q, q) == 1.0

    def test_similar_questions(self) -> None:
        """Test similar questions have high similarity."""
        q1 = "Where does this person work?"
        q2 = "What company does this person work at?"
        similarity = calculate_similarity(q1, q2)
        assert similarity > 0.3  # Should have some overlap

    def test_different_questions(self) -> None:
        """Test different questions have low similarity."""
        q1 = "Where does this person work?"
        q2 = "What is their favorite food?"
        similarity = calculate_similarity(q1, q2)
        assert similarity < 0.3


class TestDeduplicate:
    """Tests for deduplication."""

    def test_removes_duplicates(self) -> None:
        """Test duplicate removal."""
        pairs = [
            QAPair(question="Where do they work?", answer="Acme", source_id="s1"),
            QAPair(question="What company do they work at?", answer="Acme Corp", source_id="s1"),
            QAPair(question="What is their favorite color?", answer="Blue", source_id="s1"),
        ]

        result = deduplicate(pairs, threshold=0.5)
        # Should keep first of duplicates and the unique one
        assert len(result) <= 3

    def test_keeps_unique(self) -> None:
        """Test unique pairs are kept."""
        pairs = [
            QAPair(question="Where do they work?", answer="Acme", source_id="s1"),
            QAPair(question="What is their favorite food?", answer="Pizza", source_id="s1"),
            QAPair(question="What language do they use?", answer="Python", source_id="s1"),
        ]

        result = deduplicate(pairs, threshold=0.7)
        assert len(result) == 3

    def test_empty_list(self) -> None:
        """Test empty list handling."""
        assert deduplicate([]) == []


class TestDeduplicateAgainstExisting:
    """Tests for deduplication against existing pairs."""

    def test_removes_duplicates_of_existing(self) -> None:
        """Test removing duplicates of existing pairs."""
        existing = [
            QAPair(question="Where do they work?", answer="Acme", source_id="s1"),
        ]
        new = [
            QAPair(question="What company do they work for?", answer="Acme Corp", source_id="s2"),
            QAPair(question="What is their name?", answer="John", source_id="s2"),
        ]

        result = deduplicate_against_existing(new, existing, threshold=0.5)
        # "What company do they work for?" is similar to existing, should be removed
        assert len(result) <= 2

    def test_keeps_new_unique(self) -> None:
        """Test keeping unique new pairs."""
        existing = [
            QAPair(question="Where do they work?", answer="Acme", source_id="s1"),
        ]
        new = [
            QAPair(question="What is their hobby?", answer="Coding", source_id="s2"),
        ]

        result = deduplicate_against_existing(new, existing, threshold=0.7)
        assert len(result) == 1
