# Entity Chain Prompt
#
# Broader entity linking for multi-hop chain discovery.
# Links memories that share ANY named entity (person, place,
# business, doctor) even when the broader topics differ.

---system---
You identify whether memories mention the same specific named entity:
a person, doctor, business, building, or place. Your job is to LINK
memories that share a real-world entity -- even if the surrounding
topics differ. Default to should_link=false when no named entity is
shared. Respond with structured output only.

---user---
New memory: "{new_content}"

Existing memories:
{numbered_list}

Do any existing memories mention the SAME specific named entity
(person, place, business, doctor) as the new memory?

SAME entity (link these):
- "Sarah raved about her neighbor Linda" + "Linda charges forty per person"
  (same Linda)
- "Dr. Patel moved to Riverside Medical" + "Sarah sees Dr. Patel on Thursday"
  (same Dr. Patel)
- "The Riverside building has a pharmacy" + "Dr. Patel moved to Riverside"
  (same Riverside Medical building)
- "Alex's mother Sarah is visiting" + "Mom Sarah mentioned her neighbor"
  (same Sarah -- Alex's mother)

DIFFERENT entities (do NOT link):
- "Alex's mother Sarah" + "Brian's daughter Sarah"
  (different people named Sarah)
- "conference room Thursday" + "cardiologist Thursday"
  (same day, different events, no shared named entity)
- "Sofi's school project" + "neighborhood block party"
  (no shared named entity)

Key: link when the SAME specific named entity appears in both memories.
Two memories about "Sarah" only link if context confirms the same Sarah.

If same entity found: should_link=true, list indices (1-based),
label under 5 words identifying the shared entity.
If no shared named entity or unclear: should_link=false.
