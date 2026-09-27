"""
Scope control. Everything that decides whether a question gets answered at all.

THREE LAYERS, in the order they run:

1. META QUESTIONS - "what can I ask?" is about the assistant, not about
   diabetes. It would score low and be refused, which is a poor experience for
   the first thing a new user typically types. Caught before anything else and
   answered from TOPICS.

2. INPUT GUARDRAIL (the relevance gate) - if the best cosine score across all
   808 chunks is below RELEVANCE_THRESHOLD, refuse without retrieving or
   calling the LLM. Free, instant, and it cannot be argued with.

3. PROMPT GUARDRAIL - the SCOPE section in prompt.py. Catches questions that
   use diabetes-sounding vocabulary but aren't actually covered.

Layer 2 alone is not enough: "side effects of chemotherapy" scores 0.7521
because it matches generic "higher chance of getting infections" text on a
diabetes page. Cosine compares wording, not topics. Layer 3 catches it.

Layer 3 alone is not enough either: it costs an API call for every piece of
junk, and a prompt instruction can be argued around.

NOT YET IMPLEMENTED: an OUTPUT guardrail - validating Claude's response before
it reaches the user (do all [n] citations point at sources that exist? is the
disclaimer present?). See README.
"""

import re

# ---------------------------------------------------------------------------
# The relevance gate
# ---------------------------------------------------------------------------

# Calibrated with tests/calibrate_threshold.py against 10 in-scope and 10
# out-of-scope questions:
#
#     threshold   in-scope passed   out-of-scope blocked
#       0.70          10/10                9/10
#       0.60          10/10                6/10     <- here
#       0.50          10/10                2/10
#
# STARTED at 0.70, LOWERED to 0.60 after real use: 0.70 refused far too many
# genuine questions. The calibration questions were more formally worded than
# what people actually type, so the measured 0.7078 floor was optimistic.
#
# 0.60 still blocks the clearly absurd (capital of Australia 0.428, World Cup
# 0.439, Python 0.517, sourdough 0.539) while letting real questions through.
# Medical-but-not-diabetes (appendicitis 0.694, chemotherapy 0.752) now reaches
# the LLM, which is fine - the prompt catches those.
#
# NOTE: 0.60 is NOT "60%". Cosine similarity is not a percentage. This number
# is specific to this embedding model and this corpus. Re-run the calibration
# if either changes.
RELEVANCE_THRESHOLD = 0.60


def passes_relevance_gate(score: float) -> bool:
    return score >= RELEVANCE_THRESHOLD


# ---------------------------------------------------------------------------
# What the corpus covers
# ---------------------------------------------------------------------------

# Derived from the 12 metadata categories, written in patient words. Shown on
# a refusal and in answer to "what can I ask?" - a bare "no" is a bad
# experience, so a refused user should learn what WOULD work.
TOPICS = """I can answer questions about:

- Types of diabetes - type 1, type 2, prediabetes, gestational, and rarer
  inherited forms (MODY / neonatal)
- Symptoms, causes and risk factors
- Tests and diagnosis, including the A1C test
- Managing diabetes - insulin, medicines, blood glucose monitoring,
  continuous glucose monitors, artificial pancreas, islet transplantation
- Healthy living - eating, physical activity, weight, smoking, mental health
- Complications - feet, eyes, kidneys, nerves, heart and stroke, gum disease,
  sexual and bladder problems, low blood glucose
- Diabetes and pregnancy, and life after the baby is born
- Help paying for diabetes care

All answers come from NIDDK patient-education pages."""

REFUSAL = (
    "I can only answer questions about diabetes using NIDDK sources, "
    "and this question falls outside them.\n\n" + TOPICS
)


# ---------------------------------------------------------------------------
# Meta questions
# ---------------------------------------------------------------------------

META_PATTERNS = (
    "what can i ask",
    "what can you",
    "what do you do",
    "what do you know",
    "what topics",
    "what subjects",
    "who are you",
    "what are you",
    "how do you work",
    "help me",
    "what is this",
)

_EXACT_META = {"help", "?", "hi", "hello"}


def is_meta_question(question: str) -> bool:
    """True for questions ABOUT the assistant rather than about diabetes.

    Deliberately narrow. "what can i EAT with diabetes" and "what do you TAKE
    for low blood sugar" must NOT match - they're real questions that happen to
    start with the same words.
    """
    q = question.lower().strip()
    if q in _EXACT_META:
        return True
    return any(pattern in q for pattern in META_PATTERNS)


# 
# Output validation (partial - see module docstring)
# ---------------------------------------------------------------------------

def find_invalid_citations(answer: str, n_sources: int) -> list[int]:
    """Citation numbers in the answer that don't point at a real source.

    If 5 sources were supplied, a '[7]' is provably fabricated. A human reader
    would not notice - [7] looks exactly as legitimate as [2].
    """
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(c for c in cited if c < 1 or c > n_sources)
---------------------------------------------------------------------------