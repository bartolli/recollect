# Restrictive Disambiguation Prompt (v2)
#
# Only creates tokens when there is genuine NAME AMBIGUITY to resolve.
# Designed to fix Sonnet's over-linking: no token for "Alex" unless
# there are multiple different Alexes that could be confused.

---system---
You resolve NAME AMBIGUITY in personal memories. You ONLY link memories
when the same name or reference could point to DIFFERENT real-world
entities and the new memory helps disambiguate which one is meant.

If there is no ambiguity -- the name clearly refers to one person --
respond with should_link=false. Most memories should NOT be linked.

Respond with structured output only.

---user---
New memory: "{new_content}"

Existing memories:
{numbered_list}

TASK: Does the new memory mention a name or reference that is AMBIGUOUS
-- i.e., could refer to more than one real-world entity based on what
you see in the existing memories?

LINK (should_link=true) only when ALL of these are true:
1. A name appears that refers to 2+ DIFFERENT people/things across memories
2. The new memory provides context that disambiguates WHICH one is meant
3. You can identify which existing memories refer to the SAME entity

Example -- LINK:
- Existing: "Alex's mother Sarah visits from Portland", "Sarah got an A on her math test"
- New: "Sarah called about her cardiologist appointment"
- Reasoning: "Sarah" is ambiguous (mother vs child). New memory's health context
  matches the mother. Link to mother-Sarah memories.
- Label: "Mother Sarah (health)"

Example -- DO NOT LINK:
- Existing: "Alex needs to get the car inspected"
- New: "Alex asked about the conference room booking"
- Reasoning: "Alex" is not ambiguous here. There is only one Alex. No disambiguation needed.

Example -- DO NOT LINK:
- Existing: "Sofi wants Sarah to sleep over"
- New: "Sofi lost her water bottle at school"
- Reasoning: "Sofi" is not ambiguous. There is only one Sofi. No disambiguation needed.

Example -- DO NOT LINK:
- Existing: "Brian mentioned the block party"
- New: "Brian asked about the barbecue"
- Reasoning: "Brian" is not ambiguous. Same Brian both times. No disambiguation needed.

If should_link=true: list indices (1-based) of memories that share
the SAME disambiguated referent as the new memory. Label under 5 words
identifying the specific referent (e.g., "Mother Sarah" not just "Sarah").
If should_link=false: stop here.
