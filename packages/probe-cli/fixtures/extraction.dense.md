# version: 1.0.0-dense
# applies-to: extraction-template
# placeholders: max_concepts, max_relations

# task
extract structured patterns from text. emit JSON matching the schema.

# scoring formula
S(q, c) = max_i cos(q, c_i)
- concepts → embedded independently → MaxSim against query.
- context → bi-encoder primary search vector.
- context_tags → embedded independently → MaxSim, multi-perspective surface.

# significance
sig in [0, 1]:
- safety-critical (allergy, phobia, medical, danger): 0.8 - 1.0
- identity-defining (career, habit, belief, relationship): 0.6 - 0.8
- preference / aversion (likes, dislikes, comfort): 0.4 - 0.6
- situational (plan, task, schedule): 0.2 - 0.4
- routine (small talk, weather, status): 0.0 - 0.2

# valence
val in [-1, 1]:
- strong negative: fear, grief, distress, anger, phobia.
- mild negative: discomfort, annoyance.
- neutral: factual, logistical.
- mild positive: interest, satisfaction.
- strong positive: passion, joy, excitement.

# fields

- concepts: max {max_concepts}. specific scenario phrases, NOT abstract categories. each embeds independently.
- entities: name (canonical) + entity_type + confidence in [0, 1]. confidence: >=0.9 unambiguous, 0.6-0.8 likely, <=0.5 ambiguous.
- relations: max {max_relations}. fields: source, relation, target, confidence, category, context, context_tags.
  - context: one sentence WHEN/WHY this fact matters. bi-encoder primary, shown verbatim during retrieval.
  - context_tags: 10 - 12 phrases (2 - 4 words each, lowercase). each embeds independently. multi-perspective: safety/practical, social/planning, identity/personal. every tag >= 2 words.

# entity_type (closed enum)
person, organization, place, food, cuisine, product, event, skill, condition, unknown

# predicates (closed enum)
- health/dietary/constraint: is_allergic_to, is_phobic_of, has_condition, takes_medication, avoids, requires, tolerates
- identity: works_at, studies, practices, holds_role, lives_in, originates_from
- relationship: is_related_to, is_friend_of, is_colleague_of, is_partner_of
- preference: prefers, dislikes, is_interested_in
- schedule: scheduled_for, recurs_on
- escape: is_associated_with (use only when no closed predicate fits)

# category (closed enum)
health, dietary, constraint (safety-critical, always surfaced),
identity, relationship, preference, schedule, general.

# disambiguation
food restrictions are ALWAYS dietary regardless of motivation: vegetarian (ethics), kosher (religion), gluten-free (choice) all map to dietary.

# concept guidance
BAD: "venue selection" — matches any venue query.
GOOD: "rooftop party planning", "rock climbing outing" — matches only elevation queries.

# context_tag guidance
BAD: "food", "safety" — single words match everything.
GOOD: "thai restaurant dinner", "food allergy warning" — specific scenarios.

# relational anchors in context_tags
context provides a relational handle for an entity ⇒ encode it as a tag:
- "Brian's mother allergy", "Stripe payments engineer", "Tomas's older sister visit".
- shortest path that uniquely identifies the referent.
- canonicalize the anchor: "mother" not "mom"; "colleague" not "co-worker"; same form across all traces about the same person.
- bare entity reference ⇒ skip; only emit when context provides the anchor.

# ambiguity
context underdetermines referent ⇒ confidence <= 0.5; context_tags hedge across plausible interpretations.

# stratified examples (one per category)

[health] {{"source": "Mei", "relation": "is_allergic_to", "target": "peanut", "confidence": 0.95, "category": "health", "context": "Severe peanut allergy with EpiPen; critical at restaurants, while cooking, planning catered meals", "context_tags": ["thai restaurant dinner", "food ingredient checking", "catered meal planning", "cooking with nuts", "school snack policy", "travel food safety", "potluck dish planning", "allergy safe menu", "grocery shopping allergens", "kids party catering"]}}

[dietary] {{"source": "Marcus", "relation": "avoids", "target": "meat", "confidence": 0.9, "category": "dietary", "context": "Strict vegetarian for ethical reasons; relevant to menu planning, restaurant choice, catering", "context_tags": ["vegetarian menu planning", "restaurant cuisine choice", "team event catering", "ethical food preference", "plant based meal", "office lunch order", "dinner party hosting", "grocery shopping vegan", "potluck dietary needs", "travel meal options"]}}

[constraint] {{"source": "Lena", "relation": "requires", "target": "wheelchair-accessible venue", "confidence": 0.95, "category": "constraint", "context": "Wheelchair accessibility required for any in-person meeting or event venue", "context_tags": ["accessible event venue", "in-person meeting space", "offsite location booking", "conference room selection", "restaurant reservation accessibility", "team retreat planning", "ada compliant venue", "barrier free access", "ground floor meeting", "elevator equipped building"]}}

[identity] {{"source": "Tomas", "relation": "works_at", "target": "Pixar", "confidence": 0.95, "category": "identity", "context": "Senior animator at Pixar; relevant to professional context, animation industry, technical recruiting", "context_tags": ["Pixar senior animator", "Tomas animation career", "animation industry contact", "professional recruiting", "creative software user", "studio production work", "career discussion topic", "professional referral", "industry event invitation", "skill collaboration"]}}

[relationship] {{"source": "Anjali", "relation": "is_related_to", "target": "Priya", "confidence": 0.95, "category": "relationship", "context": "Anjali is Priya's mother; lives next door, weekly visits — family planning context", "context_tags": ["Priya's mother visit", "Anjali next-door neighbor", "family event planning", "weekend household visit", "elder care logistics", "intergenerational gathering", "holiday hosting", "family dinner invitation", "local family network", "multi-generation household"]}}

[preference] {{"source": "Eli", "relation": "is_interested_in", "target": "vintage motorcycles", "confidence": 0.9, "category": "preference", "context": "Restores vintage motorcycles on weekends; relevant for gifts, event topics, weekend planning", "context_tags": ["weekend hobby planning", "vintage gear shopping", "motorcycle event invitation", "garage workshop project", "mechanical restoration interest", "gift idea brainstorming", "club membership suggestion", "ride day planning", "parts sourcing tip", "enthusiast meetup"]}}

[schedule] {{"source": "team", "relation": "recurs_on", "target": "Monday 9am standup", "confidence": 1.0, "category": "schedule", "context": "Engineering team standup every Monday at 9am; affects scheduling availability", "context_tags": ["meeting scheduling conflict", "monday morning availability", "team sync planning", "calendar block reservation", "weekly recurring event", "engineering standup time", "early morning meeting", "agile ceremony slot", "sync meeting cadence", "team availability check"]}}

[general] {{"source": "John", "relation": "is_associated_with", "target": "climate documentary", "confidence": 0.7, "category": "general", "context": "Watched documentary on climate policy; conversational context, no immediate action implication", "context_tags": ["recent media consumption", "documentary recommendation", "climate topic interest", "evening tv viewing", "current events discussion", "policy topic awareness", "weekend entertainment", "streaming recommendation", "casual conversation topic", "shared cultural reference"]}}

[ambiguous] {{"source": "John", "relation": "is_interested_in", "target": "jaguars", "confidence": 0.45, "category": "preference", "context": "Interest in jaguars — referent ambiguous: animal (wildlife) or car brand (automotive)", "context_tags": ["exotic car shopping", "wildlife safari trip", "luxury vehicle comparison", "zoo animal encounter", "sports car enthusiast", "big cat conservation", "automotive gift ideas", "nature documentary", "car dealership visit", "animal sanctuary outing"]}}

# output
- emotional_valence: float in [-1, 1]
- significance: float in [0, 1]
- fact_type: episodic (one-time event) | semantic (enduring fact)
