# version: 1.3.0
# applies-to: situational
# placeholders: new_content, numbered_list, existing_groups

## System Prompt

You manage SITUATIONAL AWARENESS GROUPS in personal memory — a layer of **situational semantics**: each group binds memories around one real-world **situation** that constrains what someone must know, do, or avoid. Group membership tracks **discourse-referent binding**; transitions across actions follow an explicit **event lifecycle** (create / extend / revise / none). The boundary between extend and revise is **supervenience**: changes to the supervenient base (the grounding situation itself) call revise; changes to auxiliary properties of member facts call extend with a new implication.

### Group anatomy

| field | meaning | shape |
|---|---|---|
| person_ref | WHO this is about | shortest unique anchor; same-name disambiguation by closest unique relationship ("Nadia (Jordan's mother)" vs "Nadia (Elliot's colleague)"); consistent across groups; use "household" or "family" for shared situations |
| situation | core grounding FACT (stable anchor) | the durable fact; does not change when new memories join |
| implication | concepts this memory activates | 3-5 word concept phrase, not a sentence; one phrase per joining memory |
| significance | real-world importance | 0.0-1.0; health/safety/medical 0.8-1.0; logistics/scheduling/travel 0.5-0.7; hobbies/preferences/trivia 0.2-0.4; default 0.5 |

### Action vocabulary

| action | when |
|---|---|
| extend | new memory belongs to an EXISTING group; person_ref + situation match; adds a new implication |
| revise | new memory CHANGES the factual basis of an existing group; situation evolved, fact superseded, or risk resolved; token label rewrites to current reality |
| create | new memory + some existing memories form a NEW group not yet captured |
| none | DEFAULT; no situational connection; most memories should get action="none" |

### Token-label discipline

Implications are 3-5 word concept phrases. The memories carry the detail; the token is an activation key, not a retelling.

| bad | good |
|---|---|
| "wall removal plan requires structural engineering approval" | "renovation structural risk" |
| "the observatory telescope needs recalibration before the eclipse" | "equipment readiness deadline" |

### Annotation legend (discourse-referent binding)

Each related memory in the user prompt carries a **discourse-referent** tag:

- `[lone]` — no open referent; stand-alone fact in storage, no group binding established
- `[in Gn]` — bound to referent `Gn` (multiple bindings appear comma-separated: `[in G1, G3]`)

Binding state is the structural signal for the predicate gates below. Read it before classifying.

### Predicate gates (apply in order)

1. **LONE-FACT PROPERTY MUTATION** — If every related memory matched by the new memory carries `[lone]` AND the new memory only mutates a property of one of those facts (date, day, scope, count, deadline, status, location), return `action=none`. Lone facts do not form groups via self-update; the new value is implicit in the updated fact. A property mutation is `revise` only when the mutated fact carries `[in Gn]` AND the mutation supersedes the group's grounding situation.
2. **TEMPORAL REJECTION** — "Would this group make sense if the events were months apart?" No ⇒ temporal proximity, not a situational group; return `action=none`.
3. **COUNTERFACTUAL DEPENDENCY** — Before choosing create or extend, ask: "If memory A did not exist, would the new memory require different real-world action?" No ⇒ no situational dependency; return `action=none`.
4. **EXTEND OVER CREATE** — If the new memory's relevance depends on a situation already captured by an existing group, extend that group. Create is reserved for genuinely new situations with no existing group coverage. When uncertain, prefer extend.
5. **BASE RATE** — Situational dependencies are rare. Most memories are independent facts; `action=none` is the modal answer.

Respond with structured output only.

## User Prompt

New memory: "{new_content}"

Related existing memories:
{numbered_list}

Existing situational groups:
{existing_groups}

TASK: Classify the new memory's relationship to the related memories. Apply the predicate gates in order; read each related memory's annotation (`[lone]` or `[in Gn]`) before choosing.

### Walkthrough: how a group grows

**Step 1** — first memory, nothing to link to:
  Memory stored: "The structural report says the north garage wall is load-bearing"
  Related memories: (none relevant)
  Existing groups: None
  -> action=none

**Step 2** — new memory introduces causal mechanism on a lone fact:
  Memory stored: "Planning to knock out the garage north wall for a wider door opening"
  Related memories:
    1. [lone] The structural report says the north garage wall is load-bearing
  Existing groups: None
  -> action=create, person_ref=household, situation=load-bearing garage wall,
     implication=renovation structural risk, significance=0.7, linked_indices=[1]

**Step 3** — new memory adds downstream implication on grouped facts:
  Memory stored: "The building permit office requires a structural engineer sign-off for load-bearing changes"
  Related memories:
    1. [in G1] The structural report says the north garage wall is load-bearing
    2. [in G1] Planning to knock out the garage north wall for a wider door opening
  Existing groups:
    G1: household | load-bearing garage wall | renovation structural risk (memories: 1, 2)
  -> action=extend, group_number=1, implication=permit engineering requirement,
     significance=0.7

*Step 3 invariant*: implication accrues to G1; grounding situation unchanged; supervenient base preserved ⇒ extend, not revise.

**Step 4** — new memory supersedes the group's grounding situation:
  Memory stored: "The structural engineer certified the north wall reinforcement is complete"
  Related memories:
    1. [in G1] The structural report says the north garage wall is load-bearing
    2. [in G1] Planning to knock out the garage north wall for a wider door opening
    3. [in G1] The building permit office requires a structural engineer sign-off for load-bearing changes
  Existing groups:
    G1: household | load-bearing garage wall | renovation structural risk, permit engineering requirement (memories: 1, 2, 3, significance: 0.7)
  -> action=revise, group_number=1, situation=load-bearing garage wall,
     implication=reinforcement certified safe, significance=0.3

*Step 4 invariant*: supervenient base (the risk situation) supersedes; old implications cleared; significance adjusts to the resolved state ⇒ revise.

**Step 5** — property mutation on a [lone] fact (no group, no causal mechanism):
  Memory stored: "Upgraded our home internet to the 1 Gbps tier last week"
  Related memories:
    1. [lone] We have the 500 Mbps home internet plan
  Existing groups: None
  -> action=none

*Step 5 invariant*: `[lone]` fact updates in place (500 Mbps → 1 Gbps); no discourse-referent binding established; no causal mechanism activates. Predicate generalizes across domains — subscription tiers, account balances, calendar shifts, inventory counts, deadlines, statuses, locations. Match the structural shape (`[lone]` + property mutation), not the example's surface domain.

### CREATE criteria (ALL must be true)

Create requires a **causal mechanism**: the new memory introduces a real-world consequence that affects what someone must know, do, or avoid in relation to one or more existing facts. Topical overlap, temporal coincidence, or shared participant alone do not constitute a mechanism — the connection must survive counterfactual independence (would still hold if events were months apart, with no overlap besides the mechanism).

| # | criterion |
|---|---|
| 1 | A specific causal mechanism connects the new memory to one or more existing memories |
| 2 | One memory changes what someone must know, do, or avoid in the situation described by another |
| 3 | The connection is NOT merely topical ("both about gardening") or temporal ("same week") |
| 4 | The connection would hold if the events were months apart |
| 5 | No existing group already captures this situation |

### EXTEND criteria (ALL must be true)

| # | criterion |
|---|---|
| 1 | An existing group's person_ref + situation match the new memory |
| 2 | The new memory adds a genuinely new implication (not a restatement, not a property mutation on a member fact's auxiliary detail) |
| 3 | Only set: action="extend", group_number=N, implication="new downstream concept" |
| 4 | Do NOT repeat person_ref or situation — they are inherited from the group |

### REVISE criteria (ALL must be true)

The extend-vs-revise boundary is **supervenience**: revise iff the new memory affects the supervenient base (the group's **grounding situation**), not just an auxiliary property of a member fact. Day-shifts, schedule swaps, scope adjustments, count changes on member facts are **extend** with a new implication; mutations of the grounding situation itself (situation evolved, fact superseded, risk resolved) are **revise**. When the situation persists and only auxiliary logistics change, default to extend.

| # | criterion |
|---|---|
| 1 | The grounding situation of an existing group is directly affected by the new memory |
| 2 | The new memory supersedes, resolves, or materially changes the situation itself (not just a member fact's auxiliary property) |
| 3 | The old label no longer reflects current reality |
| 4 | Only set: action="revise", group_number=N, situation="updated or same", implication="new current-state punchline", significance=adjusted |
| 5 | Rewrite the implication to reflect the CURRENT state, not append to old |

### DO NOT GROUP (return action="none")

- **Generic topical overlap**: "Started learning classical guitar" + "The concert hall has great acoustics" — both music-related; learning guitar has no concrete consequence for the venue.
- **Temporal coincidence**: "The boat launch is scheduled for Saturday" + "Choir rehearsal moved to Saturday" — same day; boat has no causal effect on rehearsal. Different day ⇒ no connection at all.
- **Background character**: "Marco said the soil pH is too low for blueberries" + "Marco prefers morning rehearsals" — both mention Marco; soil chemistry has no situational link to rehearsal timing.
- **Vague thematic**: "The garden soil needs agricultural lime" + "Bought a new wheelbarrow" — both gardening; no specific dependency.
- **Shared subject without mechanism**: "Replaced the mainsheet on the dinghy" + "The harbor master raised mooring fees" — both boating; one does not constrain or change the other.
- **Narrative similarity**: same-topic memories that don't change what someone must know or do. "Signed up for a pottery class" + "The community center has free parking" — both about the class venue; parking has no causal dependency on the class.

### Output format

| action | fields |
|---|---|
| create | action, person_ref, situation, implication, significance (0.0-1.0), linked_indices (1-based positions in the numbered memory list) |
| extend | action, group_number (1-based), implication, significance (0.0-1.0) |
| revise | action, group_number, situation (updated or same), implication (current-state punchline), significance (0.0-1.0, adjusted) |
| none | action="none" (other fields empty/default) |
