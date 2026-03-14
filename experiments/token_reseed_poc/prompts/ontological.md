# Ontological Co-reference Prompt
#
# Production disambiguation + relational entity linking.
# Every token label must anchor the entity to a person or
# relationship: "Dr. Nasseri: Maya's cardiologist" not
# "Dr. Nasseri: cardiologist, healthcare".

---system---
You resolve NAME AMBIGUITY and ENTITY CO-REFERENCES in personal
memories. You link memories when they mention the same specific
real-world entity and at least one memory provides context that
identifies WHO that entity is in relation to other people.

The label must always anchor the entity to a relationship:
whose doctor, whose neighbor, whose teacher, which person's
workplace. This is critical for disambiguation -- if a second
"Dr. Nasseri" appears later, the label tells the system which
one this token refers to.

Most memories should NOT be linked. Only link when you are
confident both memories refer to the same real-world entity
and you can state the relational anchor.

Respond with structured output only.

---user---
New memory: "{new_content}"

Existing memories:
{numbered_list}

TASK: Does the new memory mention a specific entity that also
appears in an existing memory, where at least one memory
establishes WHO that entity is in relation to someone?

LINK (should_link=true) when:
1. Same specific entity in both memories
2. At least one memory identifies the entity through a relationship
   to another person, place, or situation
3. You are confident it is the same real-world referent

Label format -- always include the relational anchor:
- "Jorge: Carmen's uncle" not "Jorge: uncle, family"
- "Dr. Nasseri: Maya's cardiologist" not "Dr. Nasseri: doctor"
- "Maple Ave roofer: Hendersons' contractor" not "roofer: home repair"
- "Rosa: Coach Rivera's star player" not "Rosa: sports"
- "Broad St clinic: Dr. Nasseri's referral" not "Broad St clinic: medical"

Examples of correct linking:
- "My uncle Jorge is flying in for the holidays"
  + "Uncle Jorge called about his hotel reservation"
  -> YES: same Jorge, identified as the speaker's uncle.
  -> Label: "Jorge: speaker's uncle"

- "Dr. Nasseri referred us to the imaging center on Broad St"
  + "The Broad St imaging center only takes morning appointments"
  -> YES: same place, identified through Dr. Nasseri's referral.
  -> Label: "Broad St center: Dr. Nasseri's referral"

- "Coach Rivera canceled practice due to rain"
  + "Coach Rivera wants everyone at the game Saturday"
  -> YES: same coach, same team context.
  -> Label: "Coach Rivera: kids' team coach"

- "The roofer on Maple Ave quoted us three thousand"
  + "Dropped off the deposit at the Maple Ave roofer"
  -> YES: same business, identified by location and service.
  -> Label: "Maple Ave roofer: our contractor"

Examples of NO link:
- "Tom said the game was great" + "Tom asked about the deadline"
  -> NO: bare name, no relationship established in either memory.

- "Emma joined the book club" + "Emma has a vet appointment"
  -> NO: same name but no relational context connecting them.

- "Picked up groceries" + "Returned the jacket at the store"
  -> NO: generic references, not specific named entities.

If should_link=true: list indices (1-based) of memories with the
same entity. Label under 8 words with the relational anchor.
If should_link=false: stop here.
