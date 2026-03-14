# Relational Grouping Prompt (v1)
#
# Current baseline prompt. Groups memories by shared real-world referent.
# Problem: Sonnet over-links -- creates tokens for every entity mention,
# flooding results with token-activated hay.

---system---
You identify whether memories refer to the same specific person,
object, or ongoing situation. Your job is to GROUP memories that
share a real-world referent -- not to find thematic similarity.
Default to should_link=false. Respond with structured output only.

---user---
New memory: "{new_content}"

Existing memories:
{numbered_list}

Do any existing memories refer to the SAME specific person, object,
or ongoing situation as the new memory?

SAME referent (link these):
- "Sarah called about her cardiologist" + "Sarah's blood pressure prescription"
  (same Sarah, same health context)
- "Sofi's science fair project" + "Sarah and Sofi building a volcano"
  (same project)

DIFFERENT referents (do NOT link):
- "Alex's mother Sarah" + "Brian's daughter Sarah"
  (different people named Sarah)
- "conference room Thursday" + "cardiologist Thursday"
  (different events on same day)
- "Sofi's school" + "neighborhood block party"
  (different contexts)

Key: two memories about "Sarah" only link if contextual clues
confirm they mean the SAME Sarah.

If same referent: should_link=true, list indices (1-based),
label under 5 words identifying the shared referent.
If different referents or unclear: should_link=false.
