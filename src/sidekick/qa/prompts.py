"""Prompts for QA generation."""

SYSTEM_PROMPT = """You are an expert at extracting factual information about a person from their notes, documents, and conversations.

Your task is to generate question-answer pairs that capture important facts about the person. These QA pairs will be used to train a personal memory model.

Guidelines:
1. Questions should be phrased as if asking ABOUT the person (third person)
2. Answers should be concise (1-2 sentences maximum)
3. Only extract factual information, not speculation or opinions unless clearly stated
4. Focus on:
   - Personal facts (job, location, relationships, background)
   - Projects and work they're doing
   - Preferences and opinions they've expressed
   - Technical skills and expertise
   - Important events or milestones
   - People they know and relationships
5. Avoid redundant questions
6. Generate 3-10 QA pairs depending on content density
7. If the content has no extractable personal information, return an empty list

Output format: JSON array of objects with "q" (question) and "a" (answer) keys.
"""

USER_PROMPT_TEMPLATE = """Extract question-answer pairs from the following text about a person:

---
{text}
---

Generate QA pairs as a JSON array. Example format:
[
  {{"q": "Where does this person work?", "a": "They work at Acme Corp as a software engineer."}},
  {{"q": "What programming languages do they use?", "a": "They primarily use Python and TypeScript."}}
]

Return ONLY the JSON array, no other text."""


DEDUP_PROMPT_TEMPLATE = """You have two question-answer pairs. Determine if they are semantically duplicates (asking essentially the same thing).

QA Pair 1:
Q: {q1}
A: {a1}

QA Pair 2:
Q: {q2}
A: {a2}

Are these duplicates? Respond with only "yes" or "no"."""


MERGE_PROMPT_TEMPLATE = """You have two answers to the same question. Merge them into a single, comprehensive answer.

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Provide a merged answer that combines the information from both. Keep it concise (1-2 sentences).
Respond with only the merged answer, no other text."""
