"""Pattern extraction using any LLM provider.

Sends text to an LLM with a structured extraction prompt,
parses the JSON response into an ExtractionResult.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from recollect.exceptions import ExtractionError
from recollect.llm.types import ExtractionResult, Message

if TYPE_CHECKING:
    from recollect.llm.protocol import LLMProvider

logger = logging.getLogger(__name__)

_EXTRACTION_TEMPLATE = """\
You are a cognitive pattern extraction system. Analyze the given text to \
identify its knowledge type, emotional weight, and structured patterns.

STEP 1 -- Classify the knowledge type to determine significance:
- Safety-critical (allergies, phobias, medical conditions, dangers) \
-> significance 0.8-1.0
- Identity-defining (career, long-term habits, core beliefs, relationships) \
-> significance 0.6-0.8
- Preference or aversion (likes, dislikes, tastes, comfort levels) \
-> significance 0.4-0.6
- Situational (plans, scheduled events, logistics, tasks) \
-> significance 0.2-0.4
- Routine (small talk, weather, status updates, mundane observations) \
-> significance 0.0-0.2

STEP 2 -- Assess emotional valence (-1.0 to 1.0):
Strong negative: fear, grief, distress, anger, phobia.
Mild negative: discomfort, mild annoyance.
Neutral: factual, logistical.
Mild positive: interest, satisfaction.
Strong positive: passion, joy, excitement.

STEP 3 -- Extract structured patterns.

HOW YOUR OUTPUT IS CONSUMED:
Your output feeds a semantic retrieval pipeline. Each field is embedded as a \
vector and matched via cosine similarity -- NOT keyword matching.

- "concepts" on the memory: each concept phrase is embedded independently. \
At query time, max(cosine(query, concept_i)) determines relevance. Write \
concepts that would be CLOSE in embedding space to real future queries.
- "context" on relations: embedded as the fact's primary search vector. \
A future query about "Thai restaurant dinner" must score HIGH against a \
peanut allergy context but LOW against a height phobia context.
- "context_tags" on relations: each tag is embedded independently, same as \
concepts. Query-predictive scenario phrases, not generic category words.
- Category health/dietary/constraint facts are auto-promoted but still \
ranked by semantic relevance to each specific query.

Return a JSON object with exactly these fields:
- "concepts": list of scenario predictions (max {max_concepts}). Each \
concept is embedded as a vector and matched against future queries via \
cosine similarity. Write SPECIFIC scenarios where this memory matters, \
not abstract categories. \
BAD: "venue selection" (matches any venue query). \
GOOD: "rock climbing outing", "rooftop party planning" (matches only \
elevation-related queries). \
BAD: "food management" (matches any food query). \
GOOD: "restaurant dinner ordering", "catering meal planning" (matches \
only dining-related queries)
- "entities": list of objects, each with "name" (canonical form), \
"entity_type" (person, place, organization, product, event, food, \
cuisine, skill, condition), and "confidence" (float 0.0-1.0: \
0.9+ unambiguous, 0.6-0.8 likely, 0.3-0.5 ambiguous)
- "relations": list of objects (max {max_relations}), each with:
  "source": subject entity (canonical name)
  "relation": predicate in verb_noun form
  "target": object entity or value
  "confidence": float 0.0-1.0
  "category": one of health, dietary, constraint (safety-critical), \
identity, relationship, preference, schedule, general
  "context": one sentence explaining WHEN and WHY this fact matters \
-- shown verbatim to another AI during retrieval
  "context_tags": 10-12 scenario phrases (2-4 words each, lowercase). Each \
tag is embedded as a vector and matched against future queries via cosine \
similarity. Write SPECIFIC scenarios a searcher would describe, not single \
generic words. Every tag must be 2+ words. \
Generate tags from MULTIPLE PERSPECTIVES to maximize retrieval surface: \
safety/practical ("food allergy warning", "ingredient checking"), \
social/planning ("group dinner ordering", "team event catering"), \
identity/personal ("dietary lifestyle choice", "ethical eating values"). \
Each perspective is like an attention head capturing a different type of relevance. \
BAD: "food", "safety", "dining" (single words, match everything). \
GOOD: "thai restaurant dinner", "food allergy warning", "ingredient \
checking order" (specific scenarios, match only relevant queries)

RELATION EXAMPLES:
{{"source": "Alex", "relation": "is_allergic_to", "target": "peanut", \
"confidence": 0.95, "category": "health", \
"context": "Severe peanut allergy requiring EpiPen; critical when ordering \
food at restaurants, checking ingredients while cooking, or planning catered meals", \
"context_tags": ["restaurant dinner ordering", "food ingredient checking", \
"catered meal planning", "thai food dining", "cooking with nuts", \
"allergy safe menu", "snack selection caution", "grocery shopping allergens", \
"travel food safety", "potluck dish planning"]}}

{{"source": "Sarah", "relation": "is_phobic_of", "target": "heights", \
"confidence": 0.95, "category": "health", \
"context": "Severe height phobia; freezes above 3rd floor. Critical when \
planning activities involving elevation or high-rise venues", \
"context_tags": ["rock climbing outing", "rooftop party venue", \
"observation deck visit", "high floor office", "outdoor rappelling", \
"zip line adventure", "balcony seating arrangement", "hiking trail elevation", \
"ferris wheel amusement", "glass floor walkway"]}}

{{"source": "Alex", "relation": "studies", "target": "Japanese", \
"confidence": 0.95, "category": "identity", \
"context": "Passionate daily Japanese language study for two years; planning \
trip to Tokyo next spring", \
"context_tags": ["japan travel planning", "tokyo trip recommendations", \
"language learning gift", "japanese culture event", "asia vacation itinerary", \
"study abroad options", "sushi restaurant outing", "anime viewing party", \
"japanese bookstore visit", "kanji practice resources"]}}

AMBIGUOUS REFERENCES -- when context does not disambiguate, lower confidence \
and hedge context_tags across all plausible interpretations:
{{"source": "John", "relation": "is_interested_in", "target": "jaguars", \
"confidence": 0.45, "category": "preference", \
"context": "John expressed interest in jaguars but context is ambiguous -- could \
be the animal (wildlife) or the car brand (automotive)", \
"context_tags": ["exotic car shopping", "wildlife safari trip", \
"luxury vehicle comparison", "zoo animal encounter", "sports car enthusiast", \
"big cat conservation", "automotive gift ideas", "nature documentary watching", \
"car dealership visit", "animal sanctuary outing"]}}

- "emotional_valence": float -1.0 to 1.0
- "significance": float 0.0 to 1.0 (use the ranges from STEP 1)
- "fact_type": "episodic" (event, one-time experience) or \
"semantic" (enduring fact, preference, identity, health condition)

Return ONLY valid JSON. No markdown fences, no explanation.\
"""


class PatternExtractor:
    """Extracts structured patterns from text using any LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        max_tokens: int | None = None,
    ) -> None:
        self._provider = provider
        if max_tokens is None:
            from recollect.config import config

            max_tokens = int(config.get("extraction.max_tokens", 8192))
        self._max_tokens = max_tokens

    def _build_prompt(self) -> str:
        """Format extraction template with config values."""
        from recollect.config import config

        prompt = _EXTRACTION_TEMPLATE.format(
            max_concepts=int(config.get("extraction.max_concepts", 5)),
            max_relations=int(config.get("extraction.max_relations", 3)),
        )
        instructions = config.extraction_instructions
        if instructions:
            prompt = f"{prompt}\n\n{instructions}"
        return prompt

    async def extract(self, text: str) -> ExtractionResult:
        """Extract patterns from text via LLM structured output."""
        messages = [
            Message(role="system", content=self._build_prompt()),
            Message(role="user", content=text),
        ]
        try:
            return await self._provider.complete_structured(
                messages,
                ExtractionResult,
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Provider error during extraction: {exc}") from exc
