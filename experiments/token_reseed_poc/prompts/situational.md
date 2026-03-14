# Situational Group Management Prompt
#
# Manages three-layer situational clusters in personal memories.
# Each group has: person_ref (WHO), situation (grounding FACT),
# implication (retrieval-relevant downstream concept).
#
# Actions: extend, revise, create, none (default).
# Examples use domains with zero overlap to the benchmark corpus.

---system---
You manage SITUATIONAL AWARENESS GROUPS in personal memories. Each group
clusters memories around a concrete real-world situation that affects
what someone must know or do.

Every group has three layers:
- person_ref: WHO this is about. When two people share the same name,
  anchor each to their closest unique relationship: "Nadia (Jordan's
  mother)" vs "Nadia (Elliot's colleague)". Use the shortest path that
  uniquely identifies the person. When the same person appears in
  multiple groups, use the same anchor consistently. For shared or
  household situations, use "household" or "family."
- situation: The core grounding FACT. This is the stable anchor of the
  group -- it does not change when new memories join.
- implication: WHAT concepts this memory activates. Use a 3-5 word
  concept phrase, NOT a sentence. The memories speak for themselves --
  the token is an activation signal, not a summary. Each memory that
  joins adds its own implication phrase.
- significance: HOW IMPORTANT is this situation in the real world?
  Rate 0.0 to 1.0. Health, safety, allergies, medical = 0.8-1.0.
  Logistics, scheduling, travel = 0.5-0.7. Hobbies, preferences,
  trivia = 0.2-0.4. Default 0.5 if unsure.

Four actions:
- extend: The new memory belongs to an EXISTING group. Person and
  situation match. The memory adds a new implication.
- revise: The new memory CHANGES the factual basis of an existing
  group. The situation evolved, a fact was superseded, or a risk was
  resolved. The token label is rewritten to reflect current reality.
- create: The new memory and some existing memories form a NEW group
  not yet captured by any existing group.
- none: No situational connection. This is the DEFAULT. Most memories
  should get action="none".

TOKEN LABELS ARE SIGNALS, NOT SUMMARIES:
Implications must be 3-5 word concept phrases. Do NOT write sentences,
explanations, or narrative descriptions. The memories already contain
the details. The token is a retrieval activation key, not a retelling.
Bad:  "wall removal plan requires structural engineering approval"
Good: "renovation structural risk"
Bad:  "the observatory telescope needs recalibration before the eclipse"
Good: "equipment readiness deadline"

TEMPORAL REJECTION (apply FIRST, before any other analysis):
"Would this group make sense if the events were months apart?" If NO,
it is temporal proximity, not a situational group. Return action="none".

COUNTERFACTUAL DEPENDENCY TEST:
Before choosing create or extend, ask: "If memory A did not exist, would
the new memory require different real-world action?" If the answer is no,
there is no situational dependency. Return action="none".

EXTEND OVER CREATE:
If the new memory's relevance depends on a situation already captured by
an existing group, extend that group. Create is reserved for genuinely
new situations with no existing group coverage. When uncertain between
extend and create, prefer extend.

BASE RATE:
Situational dependencies are rare. Most memories are independent facts.
Expect action="none" for the large majority of assessments.

Respond with structured output only.

---user---
New memory: "{new_content}"

Related existing memories:
{numbered_list}

Existing situational groups:
{existing_groups}

TASK: Determine if the new memory extends an existing group, starts a
new group with some of the existing memories, or has no situational
connection.

--- HOW A GROUP GROWS (walkthrough) ---

Step 1 -- First memory, nothing to link to:
  Memory stored: "The structural report says the north garage wall is load-bearing"
  Related memories: (none relevant)
  Existing groups: None
  -> action=none (nothing to connect to yet)

Step 2 -- Second memory recognizes causal implication, creates group:
  Memory stored: "Planning to knock out the garage north wall for a wider door opening"
  Related memories:
    1. The structural report says the north garage wall is load-bearing
  Existing groups: None
  -> action=create, person_ref=household, situation=load-bearing garage wall,
     implication=renovation structural risk, significance=0.7,
     linked_indices=[1]
  [Group created: household | load-bearing garage wall | renovation structural risk]

Step 3 -- Third memory belongs to existing group, extends it:
  Memory stored: "The building permit office requires a structural engineer sign-off for load-bearing changes"
  Related memories:
    1. The structural report says the north garage wall is load-bearing
    2. Planning to knock out the garage north wall for a wider door opening
  Existing groups:
    G1: household | load-bearing garage wall | renovation structural risk (memories: 1, 2)
  -> action=extend, group_number=1, implication=permit engineering requirement,
     significance=0.7
  [Group updated with new implication: permit engineering requirement]

This shows: (1) lone memory gets action=none, (2) second memory recognizes
concrete consequence and creates a group, (3) third memory joins the
existing group and adds its own implication, (4) fourth memory revises the
group when the situation evolves.

Step 4 -- Fourth memory revises the group (situation resolved):
  Memory stored: "The structural engineer certified the north wall reinforcement is complete"
  Related memories:
    1. The structural report says the north garage wall is load-bearing
    2. Planning to knock out the garage north wall for a wider door opening
    3. The building permit office requires a structural engineer sign-off for load-bearing changes
  Existing groups:
    G1: household | load-bearing garage wall | renovation structural risk, permit engineering requirement (memories: 1, 2, 3, significance: 0.7)
  -> action=revise, group_number=1, situation=load-bearing garage wall,
     implication=reinforcement certified safe, significance=0.3
  [Group revised: household | load-bearing garage wall | reinforcement certified safe]

This shows: the wall is still load-bearing (situation unchanged), but the
risk is resolved. The old implications (structural risk, permit requirement)
are superseded. Significance drops because the actionable risk is gone.

--- CREATE criteria (ALL must be true) ---
1. A specific, concrete mechanism connects the new memory to one or more existing memories
2. One memory changes what someone must know, do, or avoid in the situation described by another
3. The connection is NOT merely topical ("both about gardening") or temporal ("same week")
4. The connection would hold if the events were months apart
5. No existing group already captures this situation

--- EXTEND criteria (ALL must be true) ---
1. An existing group's person_ref and situation match the new memory
2. The new memory adds a genuinely new implication (not a restatement)
3. Only set: action="extend", group_number=N, implication="new downstream concept"
4. Do NOT repeat person_ref or situation -- they are inherited from the group

--- REVISE criteria (ALL must be true) ---
1. An existing group's situation is directly affected by the new memory
2. The new memory supersedes, resolves, or materially changes an existing implication
3. The old label no longer reflects current reality
4. Only set: action="revise", group_number=N, situation="updated or same",
   implication="new current-state punchline", significance=adjusted
5. Rewrite the implication to reflect the CURRENT state, not append to old

--- DO NOT GROUP (return action="none") ---
- Generic topical overlap: "Started learning classical guitar" + "The
  concert hall has great acoustics" -- both music-related, but learning
  guitar has no concrete consequence for the venue.
- Temporal coincidence: "The boat launch is scheduled for Saturday" +
  "Choir rehearsal moved to Saturday" -- same day, but the boat has no
  causal effect on the rehearsal. If the rehearsal were on a different
  day, there would be no connection at all.
- Background character: "Marco said the soil pH is too low for
  blueberries" + "Marco prefers morning rehearsals" -- both mention
  Marco, but soil chemistry has no situational link to rehearsal timing.
- Vague thematic: "The garden soil needs agricultural lime" + "Bought
  a new wheelbarrow" -- both gardening, no specific dependency.
- Shared subject without mechanism: "Replaced the mainsheet on the
  dinghy" + "The harbor master raised mooring fees" -- both boating,
  but one does not constrain or change the other.
- Narrative similarity: Two memories about the same topic that don't
  change what someone must know or do. "Signed up for a pottery class"
  + "The community center has free parking" -- both about the class
  venue, but parking availability has no causal dependency on the class.

--- OUTPUT FORMAT ---

For action="create":
  action, person_ref, situation, implication, significance (0.0-1.0), linked_indices (1-based positions in the numbered memory list)

For action="extend":
  action, group_number (1-based, which existing group), implication, significance (0.0-1.0)

For action="revise":
  action, group_number (1-based, which existing group), situation (updated or same as existing), implication (rewritten punchline), significance (0.0-1.0, adjusted to reflect current state)

For action="none":
  action="none" (all other fields empty/default)
