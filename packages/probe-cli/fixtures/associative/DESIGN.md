# Associative-recall validation corpus -- design spec

Single ground-truthed corpus that exercises the whole associative-recall stack
(semantic + concept attention + spreading + recall-token bridge) and isolates
the recall-token bridge's marginal contribution to persona-fact surfacing
(adr-recall-surfacing-gate ruling 5). Merges and hardens the two demos:
`examples/v3_needle_demo.py` (needle-in-haystack, situational queries with
near-zero lexical overlap) and `examples/v3_disambiguation_demo.py` (one name,
three referents).

Mirrors human recall: a *situation* ("team dinner", "20th-floor venue") triggers
an associatively-linked fact (a dietary constraint, a height phobia) with no
keyword in common -- the smell-triggers-a-memory pattern, made mechanical.

## Two channels, two axes (one corpus)

`recall()` has two surfaces, and "associative recall" means a different thing on
each. The corpus (traces + queries + ground truth) is shared; the metrics differ.

- **Axis 1 -- persona-fact channel (IMPORTANT CONTEXT). The ruling-5 / slice-2
  decision.** The recall floor gates on the blended score
  `S = max(bi, 0.7*csim + 0.3*bi)` -- **concept attention is inside S**. So a
  below-floor persona fact was, by construction, *not* reached by concept (if it
  had been, S would be above floor); spreading activation feeds the trace pool,
  never the fact score. The only below-floor recoveries are **bypass, pinned, or
  token-activation**. Measurement: one `recall_tokens.enabled=true` run + a
  ground-truth-aware per-case metric. There is no "concept-redundant" recovery
  case in this channel (see the dropped Case D note).

- **Axis 2 -- trace channel (Related memories, `think_about` thoughts). The broad
  framework, secondary.** This is where the needle demo's headline lives:
  "team dinner" surfacing the allergy *memory*. Here concept + spreading + tokens
  all contribute, so "marginal value of the token bridge" and a tokens-on/off
  delta are meaningful. Measured with the needle-demo-style metric (needle thought
  found at rank / precision), **not** the floor. Out of scope for the slice-2
  go/no-build; in scope for "does the whole stack recall associatively."

## Case taxonomy -- Axis 1 (persona-fact channel) coverage contract

Every query x fact pairing the probe must distinguish, persona-fact channel.
Single tokens-on run. A **case is an outcome the probe assigns**, not a label we
author: we control conditions (category, grouped/ungrouped, vector-distance via
phrasing); the run assigns the case and may reclassify. **"Case C count = 0" is a
valid, reportable result.** We may phrase a fact to sit below floor (the
experimental condition); we may not phrase until the hop fires (tuning the
outcome -- the circular trap).

| Case | Fact condition | Expected | What it proves |
|------|----------------|----------|----------------|
| A | above-floor, relevant, non-bypass | surfaces (vector) | floor passes genuine vector matches; bridge irrelevant |
| B | below-floor, relevant, non-bypass, **ungrouped** | dropped | the floor's cost; the bridge structurally cannot reach an ungrouped fact |
| C | below-floor, relevant, non-bypass, **in a token group** | recovered by bridge | **ruling-5 positive -- recovery over the bare floor** |
| E | below-floor, relevant, **bypass {health,dietary}** | surfaces (bypass) | bypass control; not bridge evidence |
| F | below-floor, **non-relevant**, token-activated (cross-talk) | suppressed | bridge does not re-admit noise (the ADR no-go) |
| G | same-name **wrong referent** (disambiguation) | suppressed | base-channel precision; bridge's only risk here is cross-group activation bleed |
| H | haystack, non-relevant | never surfaces | precision under volume |
| I | distractor query (no relevant needle) | empty block | no spurious activation |

**Dropped -- Case D ("concept/spreading reaches a below-floor fact"):** impossible
in the persona-fact channel, because concept is already inside the floored S and
spreading does not touch the fact score. Any concept-driven recovery shows up as
Case A (above floor). The concept/spreading-redundancy question is real only on
**Axis 2** (trace channel).

The Axis-1 claim is **"recall recovery over the bare floor, without re-admitting
noise"** (C > 0, F = 0) -- NOT "marginal over concept/spreading" (that is Axis 2).
The honest one-run signals are `recovered_relevant` (Case C) and
`readmitted_distractor` (Case F); the per-case metric adds the A/B/E/G/H breakdown
the aggregate lacks.

## Cast

- **Alex** -- work + travel persona (Tokyo assignment, Japanese study, peanut allergy).
- **Sarah** -- THREE referents (the disambiguation collision):
  - Sarah/mother (Portland, Sunday calls, Thanksgiving).
  - Sarah/wife (Brian's wife; also carries the dining + venue needles).
  - Sarah/classmate (Sofi's 9yo friend; school pickup).
- **Jordan** -- analog photography (an *ungrouped* needle = Case B), Dec 12 birthday.
- **Maya** -- ethical vegetarian (bypass), remote Fridays.

## Corpus

Trace phrasing is fact-bearing (clear subject-predicate-object) so persona facts
extract reliably -- slice-1b's lesson (event-phrased traces yielded no facts).
`cat` = expected fact category. `grp` = token group (situational cluster;
stamped together via restore_seed_groups). `case` = which row(s) it serves.

### Token groups (situational needle clusters)

**G_dine** -- "team dining out"
- `dine.anchor` "The team is planning a dinner out downtown next week." (general)
- `dine.sarah` "Sarah, Brian's wife, can't stand loud restaurants; she much prefers small, quiet places." (preference) [C]
- `dine.alex` "Alex has a severe peanut allergy and carries an EpiPen at all times." (health, BYPASS) [E]
- `dine.maya` "Maya has been a strict ethical vegetarian for twelve years." (dietary, BYPASS) [E]

**G_venue** -- "choosing an event venue"
- `venue.anchor` "We still need to pick a venue for the product launch party." (general)
- `venue.sarah` "Sarah, Brian's wife, has a serious fear of heights and gets panicky in tall buildings." (preference) [C]

**G_asia** -- "Alex's Tokyo assignment"
- `asia.anchor` "Alex is being assigned to the Tokyo office next quarter." (identity)
- `asia.alex` "Alex has studied Japanese every morning for two years and hopes to live in Japan." (preference) [D]

**G_sarah_m** -- "mother Sarah" (disambiguation referent + group)
- `sarah_m.1` "Alex's mother Sarah lives in Portland and calls every Sunday evening." (identity)
- `sarah_m.2` "Mother Sarah is visiting for Thanksgiving and plans to bake her pumpkin pie." (schedule) [G target]

**G_sarah_c** -- "Sofi's classmate Sarah" (disambiguation referent + group)
- `sarah_c.1` "Sofi's friend Sarah, age 9, is in her class at Westbrook Elementary." (identity)
- `sarah_c.2` "Sarah, Sofi's classmate, gets picked up from school at 3pm on Fridays." (schedule) [G target]

(Sarah/wife's referent facts live in G_dine + G_venue + the marathon trace below.)

### Ungrouped needles (no token group)

- `jordan.photo` "Jordan is devoted to analog film photography and spends weekends developing prints in a darkroom." (preference) [B]
- `sarah_w.marathon` "Sarah, Brian's wife, is training for the Portland marathon in April." (general) [A]
- `jordan.bday` "Jordan's birthday is December 12 and he prefers experiences over gifts." (general) [A/D]

### Haystack (~15, mundane, never surface) [H]

`h.1` "Alex asked about the conference room booking for Thursday." / `h.2` "Sarah
mentioned the floor-3 coffee machine is broken again." / `h.3` "Jordan said the
standing desks arrived." / `h.4` "Maya said traffic on the 405 was terrible." /
`h.5` "Alex needs the Q3 report reviewed by Friday." / `h.6` "Sarah finished the
auth-module code review." / `h.7` "Jordan asked about splitting a lunch order." /
`h.8` "Maya forwarded the project timeline." / `h.9` "Alex asked about the VPN
setup." / `h.10` "Sarah will be five minutes late to standup." / `h.11` "Jordan
said the floor-2 printer is out of toner." / `h.12` "Maya asked where the supply
closet key is." / `h.13` "The parking-garage elevator is slow today." / `h.14`
"Someone left leftovers in the fridge again." / `h.15` "The all-hands moved to
2pm Wednesday."

Plus 2 off-topic ungrouped traces the distractor queries match:
- `h.coral` "I read a long article about coral-reef bleaching in the South Pacific."
- `h.shoes` "I bought new low-light running shoes for fall mornings."

## Queries + ground truth

`surface` = facts that MUST appear (with case); `suppress` = facts that must NOT;
`tok` = is the surfacing token-dependent (predicts an on/off delta).

| id | query | surface (case) | suppress | tok |
|----|-------|----------------|----------|-----|
| q.dine | "Where should we book the team dinner this week?" | dine.sarah (C), dine.alex (E), dine.maya (E) | all other needles, haystack | dine.sarah: yes; bypass: no |
| q.venue | "Is the sky lounge on the 20th floor a good launch venue?" | venue.sarah (C) | dine.*, asia.*, sarahs, haystack | yes |
| q.asia | "What should Alex prep for the Tokyo assignment?" | asia.alex (A) | all else | no (shared Tokyo vocab -> high S -> above floor) |
| q.marathon | "How is Sarah's marathon training going?" | sarah_w.marathon (A) | mother/classmate Sarah, haystack | no |
| q.hobby | "Any ideas for a relaxing weekend activity?" | -- (jordan.photo is relevant but B: dropped) | jordan.photo stays dropped, all else | no (B = no delta) |
| q.holiday | "What is Sarah planning for the holidays?" | sarah_m.2 (A/borderline) | **sarah_w.\*, sarah_c.\*** (G) | no |
| q.pickup | "What's the school pickup schedule this week?" | sarah_c.2 (A/borderline) | **sarah_m.\*, sarah_w.\*** (G) | no |
| q.coral | "Summarize the coral-bleaching article." | -- (empty) (I) | everything; esp. no needle (F) | F-check |
| q.shoes | "Which running shoes did I buy for fall?" | -- (empty) (I) | everything (F) | F-check |

Notes:
- **q.dine** is the keystone: one situational query, near-zero lexical overlap,
  must recover Sarah's quiet-restaurant preference via the **token bridge** (C),
  while the allergy + vegetarian surface via **bypass** (E) -- the human "planning
  dinner, recall everyone's food constraints" pattern, with three mechanisms
  cleanly separated by the on/off run.
- **q.venue** "20th floor / sky lounge" vs "fear of heights / tall buildings" --
  associative, not lexical -> Case C.
- **q.asia** deliberately shares Tokyo/Japan vocabulary -> high S -> above floor
  (Case A): the vector path surfaces it, no bridge needed. On Axis 2 (trace
  channel) the same pair tests concept-redundancy; on Axis 1 it is a plain A.
- **q.holiday / q.pickup** are the disambiguation pair: the right Sarah referent
  surfaces, the other two are suppressed (G). Watch for cross-talk -- a holiday
  query must not token-bleed into classmate/wife groups.
- **q.hobby** anchors Case B: photography is genuinely relevant but ungrouped and
  vector-distant -> dropped both runs. The bridge structurally cannot reach an
  ungrouped fact; this documents the boundary, not a failure.
- **q.coral / q.shoes** are the re-admission probes (F + I): off-topic, must yield
  an empty block and must not spuriously activate a high-significance group
  (the slice-1b coral->health spurious-fire failure mode).

## Measurement protocol

**Seed once.** Extraction is nondeterministic (slice-1b: G4 2->1 facts, promoted
2->3 across runs) -- re-seeding for a second pass would put extraction noise in
any delta. Seed traces clean (`recall_tokens.enabled=false` during seed), restore
the token groups (G_dine, G_venue, G_asia, G_sarah_m, G_sarah_c) via
`restore_seed_groups`, force-promote facts, then run query passes against the one
seeded DB (toggle `recall_tokens.enabled` at query time only, never re-seed).
Overrides: `search_limit` small (below-floor target stays hop-eligible),
`max_facts_per_query` high (cohort is the full population, not a top-k slice),
`reinforce_boost=0` (per-query independence).

**Axis 1 (persona-fact, the slice-2 decision) -- one tokens-on pass.** A
tokens-off pass is redundant here: the floor deterministically drops every
below-floor non-bypass non-pinned fact, so tokens-on is the only recovery and the
tokens-off result is predictable from the floor logic. Per query, classify each
surfaced fact against ground truth into its case.

Axis-1 pass criteria:
- **C fires (or 0):** count of `{below-floor, non-bypass, non-pinned,
  source_trace_activated, relevant}` recoveries. `recovered_relevant` from
  `compute_situational_lift`. C = 0 is a valid result.
- **F = 0:** no non-relevant fact surfaces on the distractor queries
  (`readmitted_distractor`); the non-bypass `dine.sarah` makes a spurious fire
  visible here, unlike slice-1b where bypass masked it.
- **E controls hold:** bypass needles surface (regardless of activation).
- **G clean:** wrong-referent Sarah facts never surface on q.holiday / q.pickup
  -- and specifically no cross-group activation bleed (the bridge's only G risk;
  base-channel drop handles the rest).
- **B documented:** the ungrouped needle stays dropped (boundary, not failure).
- **A / H:** vector matches surface; haystack never does.

**Axis 2 (trace channel, secondary, optional) -- tokens-on vs tokens-off.** Only
here is the on/off delta meaningful (concept + spreading + tokens all feed the
trace pool). Metric: needle *thought* found at rank / block precision
(needle-demo style). Answers "does the whole stack recall associatively, and what
do tokens add on top of concept/spreading." Does not feed the slice-2 gate.

## Code

- **Axis 1 (built): `surfacing_cases.classify_cases` + `CaseBreakdown`.** Maps
  each surfaced fact to its case against the surface (`queries.jsonl`
  relevant_trace_ids) / forbid (`ground_truth.jsonl`) / grouped (`seed_groups`
  member ids) sets; `compute_situational_lift` still gives the C/F aggregate.
  Runs on the tokens-on pass via `SituationalSurfacingArmRunner.run_measurement`
  when `[surfacing].ground_truth_path` is set. Fixture: `surfacing-associative.toml`
  (`make probe-surfacing-associative`). `SurfacedFact.source_eval_id` carries the
  corpus id for the mapping.
- **Axis 2 (not built, optional): a tokens-on/off trace-channel harness.** Run
  `think_about` twice (toggle at query time, one seed) and diff the needle-thought
  recall/precision. Only this axis earns the second pass. Deferred until Axis 1
  results justify it.
- Fixtures: `seed_traces.jsonl`, `seed_groups.jsonl`, `queries.jsonl`,
  `ground_truth.jsonl` (this directory).
