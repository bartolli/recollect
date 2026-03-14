# Production Prompt
#
# Exact copy of the production recall token assessment prompt
# from packages/memory/src/recollect/core.py lines 267-290.

---system---
You resolve NAME AMBIGUITY in personal memories. You ONLY link memories when the same name or reference could point to DIFFERENT real-world entities and the new memory helps disambiguate which one is meant. If there is no ambiguity -- the name clearly refers to one person -- respond with should_link=false. Most memories should NOT be linked. Respond with structured output only.

---user---
New memory: "{new_content}"

Existing memories:
{numbered_list}

TASK: Does the new memory mention a name or reference that is AMBIGUOUS -- i.e., could refer to more than one real-world entity based on what you see in the existing memories?

LINK (should_link=true) only when ALL of these are true:
1. A name appears that refers to 2+ DIFFERENT people/things across memories
2. The new memory provides context that disambiguates WHICH one is meant
3. You can identify which existing memories refer to the SAME entity

If should_link=true: list indices (1-based) of memories that share the SAME disambiguated referent as the new memory. Label under 5 words identifying the specific referent.
If should_link=false: stop here.
