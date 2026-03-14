"""Iterative spreading activation scenario.

Tests whether iterative re-seeding discovers traces at the end of long
association chains that fixed 2-hop spreading activation misses. Three
chains of 7 nodes each, where the final 2 nodes are semantically opaque
-- they contain critical practical information but share no keywords or
semantic overlap with the query. Only discoverable through association
chains, not vector search.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    content: str
    group: str = ""  # "chain-a", "chain-b", "chain-c", "hay"


@dataclass
class Association:
    """Explicit association between two memories by index."""

    source_index: int
    target_index: int
    association_type: str = "entity"
    forward_strength: float = 0.8
    backward_strength: float = 0.5


@dataclass
class Query:
    text: str
    description: str
    expected_indices: list[int] = field(default_factory=list)  # MUST find these
    unexpected_indices: list[int] = field(default_factory=list)  # MUST NOT find these
    # Separate expected sets by retrieval depth
    expected_2hop: list[int] = field(default_factory=list)  # Reachable within 2 hops
    expected_iterative: list[int] = field(
        default_factory=list
    )  # Only reachable with re-seeding


@dataclass
class Scenario:
    name: str
    description: str
    memories: list[Memory]
    associations: list[Association]
    queries: list[Query]


SCENARIO = Scenario(
    name="iterative_spread",
    description=(
        "Three 7-node association chains with semantically opaque endpoints. "
        "Vector search finds the first 2-3 nodes in each chain. Fixed 2-hop "
        "spreading reaches the bridge nodes. Only iterative re-seeding "
        "discovers the final 2 nodes which contain critical practical "
        "information with zero query keyword overlap."
    ),
    memories=[
        # ================================================================
        # CHAIN A -- Dinner Safety (indices 0-6)
        # ================================================================
        # Nodes 0-2: Query-visible (vector search finds these)
        Memory(
            "Alex suggested trying the new Thai restaurant on Main Street"
            " for dinner with the group next Friday",
            group="chain-a",
        ),
        Memory(
            "The Thai place on Main Street uses peanut oil in most of"
            " their dishes according to online reviews",
            group="chain-a",
        ),
        Memory(
            "Pat has been joining every group dinner outing since she"
            " moved to the neighborhood last month",
            group="chain-a",
        ),
        # Node 3: Bridge (connects Pat to medical context)
        Memory(
            "Pat was hospitalized last year after accidentally ingesting"
            " peanut oil at a restaurant",
            group="chain-a",
        ),
        # Node 4: Deepening (medical detail, less query-similar)
        Memory(
            "After the hospital stay Pat's doctor prescribed an"
            " epinephrine auto-injector for emergencies",
            group="chain-a",
        ),
        # Nodes 5-6: Semantically opaque (critical info, zero query overlap)
        Memory(
            "The auto-injector refill is due next month and the current"
            " one expires on March 30th",
            group="chain-a",
        ),
        Memory(
            "CityMed urgent care on 5th Avenue is the closest emergency"
            " facility to Main Street and is open until 10pm",
            group="chain-a",
        ),
        # ================================================================
        # CHAIN B -- Birthday Gift (indices 7-13)
        # ================================================================
        # Nodes 7-9: Query-visible
        Memory(
            "Mom's 70th birthday is coming up in June and the whole"
            " family wants to do something special",
            group="chain-b",
        ),
        Memory(
            "Mom has been talking nonstop about wanting to spend more"
            " time in the garden this summer",
            group="chain-b",
        ),
        Memory(
            "Mom's physical therapist said her knee pain means she"
            " should avoid kneeling and heavy lifting",
            group="chain-b",
        ),
        # Node 10: Bridge (connects health to practical solution)
        Memory(
            "The therapist specifically recommended raised garden beds"
            " as an ideal low-impact activity for bad knees",
            group="chain-b",
        ),
        # Node 11: Deepening (specific product, less query-similar)
        Memory(
            "Cedar raised garden beds with built-in seating ledges are"
            " rated best for people with limited mobility",
            group="chain-b",
        ),
        # Nodes 12-13: Semantically opaque
        Memory(
            "The garden center on Elm Street has cedar raised beds on"
            " clearance for forty percent off this month",
            group="chain-b",
        ),
        Memory(
            "Elm Street store offers free delivery and assembly for"
            " orders over seventy-five dollars within ten miles",
            group="chain-b",
        ),
        # ================================================================
        # CHAIN C -- School Project (indices 14-20)
        # ================================================================
        # Nodes 14-15: Query-visible
        Memory(
            "Sofi has to pick a topic for her school science fair"
            " project by this Friday",
            group="chain-c",
        ),
        Memory(
            "Sofi decided to build a baking soda volcano with realistic"
            " eruption effects for the science fair",
            group="chain-c",
        ),
        # Node 16: Bridge (connects project to supplies)
        Memory(
            "The volcano model needs two pounds of baking soda, a gallon"
            " of white vinegar, and red food coloring",
            group="chain-c",
        ),
        # Node 17: Deepening (specific store)
        Memory(
            "The hardware store on Oak Street is the only place nearby"
            " that sells baking soda in bulk bags",
            group="chain-c",
        ),
        # Node 18: Practical detail
        Memory(
            "Alex has forty dollars in store credit at the Oak Street"
            " hardware store from a return last month",
            group="chain-c",
        ),
        # Nodes 19-20: Semantically opaque
        Memory(
            "Oak Street hardware closes at noon on Saturdays and is"
            " shut all day Sunday",
            group="chain-c",
        ),
        Memory(
            "Weekday parking on Oak Street is metered until 6pm but"
            " free after that with no time limit",
            group="chain-c",
        ),
        # ================================================================
        # TIER 1: Inert filler (indices 21-100) -- 80 memories
        # ================================================================
        # Work/office (21-34)
        Memory(
            "Dave sent the revised project timeline to the whole team",
            group="hay",
        ),
        Memory(
            "The quarterly review meeting got pushed to next Wednesday",
            group="hay",
        ),
        Memory(
            "Rachel needs the expense reports submitted by end of day Friday",
            group="hay",
        ),
        Memory(
            "Tom asked if the new hire orientation is still on Monday",
            group="hay",
        ),
        Memory(
            "Karen scheduled a one-on-one to discuss the Q3 roadmap",
            group="hay",
        ),
        Memory(
            "The office printer on the third floor is jammed again",
            group="hay",
        ),
        Memory(
            "Mark confirmed he will present the sales numbers at the all-hands",
            group="hay",
        ),
        Memory(
            "Lisa volunteered to take notes at the product strategy meeting",
            group="hay",
        ),
        Memory(
            "Need to update the shared drive with the latest client deck",
            group="hay",
        ),
        Memory(
            "Janet emailed about reserving the large conference room for Tuesday",
            group="hay",
        ),
        Memory(
            "Chris finished the code review for the authentication module",
            group="hay",
        ),
        Memory(
            "Sam flagged a bug in the reporting dashboard that needs attention",
            group="hay",
        ),
        Memory(
            "The IT department is migrating everyone to the new email system",
            group="hay",
        ),
        Memory(
            "Dave wants feedback on the draft proposal before sending it out",
            group="hay",
        ),
        # Finance (35-47)
        Memory(
            "The electric bill came in higher than usual this month",
            group="hay",
        ),
        Memory(
            "Need to review the auto insurance renewal before it lapses",
            group="hay",
        ),
        Memory(
            "Rachel mentioned a good cashback credit card with no annual fee",
            group="hay",
        ),
        Memory(
            "Tom set up automatic payments for the mortgage starting next month",
            group="hay",
        ),
        Memory(
            "The property tax assessment arrived and looks about the same",
            group="hay",
        ),
        Memory(
            "Karen recommended switching to a high-yield savings account",
            group="hay",
        ),
        Memory(
            "Mark said the quarterly estimated taxes are due on the 15th",
            group="hay",
        ),
        Memory(
            "Need to cancel the unused streaming subscription before renewal",
            group="hay",
        ),
        Memory(
            "Lisa forwarded a coupon code for twenty percent off home goods",
            group="hay",
        ),
        Memory(
            "The water bill seemed off so I called the utility company about it",
            group="hay",
        ),
        Memory(
            "Janet asked her accountant about deducting home office expenses",
            group="hay",
        ),
        Memory(
            "Time to start organizing receipts for tax prep",
            group="hay",
        ),
        Memory(
            "Sam found a better rate on car insurance through a different provider",
            group="hay",
        ),
        # Home maintenance (48-61)
        Memory(
            "The kitchen faucet has been dripping all week and needs a new washer",
            group="hay",
        ),
        Memory(
            "Dave recommended a good electrician for the basement wiring project",
            group="hay",
        ),
        Memory(
            "Need to replace the furnace filter before the heating season",
            group="hay",
        ),
        Memory(
            "The gutter on the south side of the house is sagging",
            group="hay",
        ),
        Memory(
            "Tom picked up caulk and weatherstripping from the hardware store",
            group="hay",
        ),
        Memory(
            "The garage door opener started making a grinding noise yesterday",
            group="hay",
        ),
        Memory(
            "Rachel found mold under the bathroom sink that needs remediation",
            group="hay",
        ),
        Memory(
            "Mark is going to pressure wash the driveway this weekend",
            group="hay",
        ),
        Memory(
            "The dishwasher is not draining properly after the last cycle",
            group="hay",
        ),
        Memory(
            "Need to schedule the annual HVAC inspection before summer",
            group="hay",
        ),
        Memory(
            "Lisa got a quote for refinishing the hardwood floors upstairs",
            group="hay",
        ),
        Memory(
            "The smoke detector batteries need replacing in the hallway",
            group="hay",
        ),
        Memory(
            "Chris offered to help patch the drywall in the spare bedroom",
            group="hay",
        ),
        Memory(
            "Called a plumber about the slow drain in the downstairs shower",
            group="hay",
        ),
        # Technology (62-73)
        Memory(
            "The laptop has been running slow since the last operating system update",
            group="hay",
        ),
        Memory(
            "Dave set up the new mesh wifi router and coverage is much better now",
            group="hay",
        ),
        Memory(
            "Need to update the password for the home security system",
            group="hay",
        ),
        Memory(
            "Tom recommended a password manager to keep everything organized",
            group="hay",
        ),
        Memory(
            "The smart thermostat lost its wifi connection again overnight",
            group="hay",
        ),
        Memory(
            "Karen asked which external hard drive is best for photo backups",
            group="hay",
        ),
        Memory(
            "The phone screen protector cracked and needs replacing",
            group="hay",
        ),
        Memory(
            "Sam mentioned a free VPN that works well for streaming abroad",
            group="hay",
        ),
        Memory(
            "The tablet battery barely lasts two hours now and might need replacing",
            group="hay",
        ),
        Memory(
            "Lisa found a good deal on a refurbished monitor for the home office",
            group="hay",
        ),
        Memory(
            "Need to renew the antivirus subscription before it expires next week",
            group="hay",
        ),
        Memory(
            "Chris set up automatic cloud backups for all the family photos",
            group="hay",
        ),
        # Travel (74-83)
        Memory(
            "Booked the hotel near the convention center for the October trip",
            group="hay",
        ),
        Memory(
            "Dave suggested taking the scenic route through the mountains instead",
            group="hay",
        ),
        Memory(
            "Need to check if the passport needs renewing before the summer trip",
            group="hay",
        ),
        Memory(
            "Tom found cheap flights to Denver for the long weekend in November",
            group="hay",
        ),
        Memory(
            "The rental car reservation confirmation came through this morning",
            group="hay",
        ),
        Memory(
            "Karen recommended packing cubes for keeping the suitcase organized",
            group="hay",
        ),
        Memory(
            "Mark said the airport parking lot fills up fast on holiday weekends",
            group="hay",
        ),
        Memory(
            "Lisa booked a cabin in the mountains for the family reunion in August",
            group="hay",
        ),
        Memory(
            "Need to print the boarding passes since the airline app keeps crashing",
            group="hay",
        ),
        Memory(
            "Janet mapped out the rest stops for the twelve-hour drive to the coast",
            group="hay",
        ),
        # Weather/seasonal (84-89)
        Memory(
            "The forecast says rain all next week so outdoor plans are off",
            group="hay",
        ),
        Memory(
            "Need to winterize the outdoor faucets before the first freeze",
            group="hay",
        ),
        Memory(
            "Tom brought in the patio furniture before the storm last night",
            group="hay",
        ),
        Memory(
            "The spring pollen count is already high and allergies are acting up",
            group="hay",
        ),
        Memory(
            "Dave mentioned the ski resort closes at the end of March",
            group="hay",
        ),
        Memory(
            "Should get the snow tires swapped out for all-seasons this week",
            group="hay",
        ),
        # Food/cooking (90-95)
        Memory(
            "Need to pick up olive oil, garlic, and pasta for dinner tonight",
            group="hay",
        ),
        Memory(
            "Tom shared his grandmother's recipe for homemade sourdough bread",
            group="hay",
        ),
        Memory(
            "The farmers market on Elm Street has fresh strawberries right now",
            group="hay",
        ),
        Memory(
            "Dave is experimenting with smoking brisket on his new pellet grill",
            group="hay",
        ),
        Memory(
            "Meal prepped five lunches for the week including chicken stir fry",
            group="hay",
        ),
        Memory(
            "The grocery delivery got the wrong brand of yogurt again",
            group="hay",
        ),
        # Hobbies (96-100)
        Memory(
            "Finished reading the third book in the mystery series last night",
            group="hay",
        ),
        Memory(
            "Dave invited me to his woodworking shop to see the table he built",
            group="hay",
        ),
        Memory(
            "The photography club is meeting at the botanical gardens this Sunday",
            group="hay",
        ),
        Memory(
            "Tom picked up new guitar strings and a capo from the music store",
            group="hay",
        ),
        Memory(
            "Karen started a jigsaw puzzle with a thousand pieces over the weekend",
            group="hay",
        ),
        # ================================================================
        # TIER 2: Moderate interference (indices 101-150) -- 50 memories
        # Family entities (Alex, Brian, Lilly, Sofi) in unrelated contexts
        # ================================================================
        Memory(
            "Alex emailed the team about the Q2 milestone deadline",
            group="hay",
        ),
        Memory(
            "Brian dropped off a casserole dish he borrowed last month",
            group="hay",
        ),
        Memory(
            "Lilly shared a recipe for gluten-free brownies she found online",
            group="hay",
        ),
        Memory(
            "Sofi drew a picture of a rainbow for her art class project",
            group="hay",
        ),
        Memory(
            "Alex and Brian watched the playoff game together on Sunday",
            group="hay",
        ),
        Memory(
            "Lilly said the school parking lot construction finishes next week",
            group="hay",
        ),
        Memory(
            "Brian signed up for a woodworking class at the community center",
            group="hay",
        ),
        Memory(
            "Alex grabbed coffee with Brian after dropping off the kids",
            group="hay",
        ),
        Memory(
            "Sofi finished her chapter book and wants to start the next one",
            group="hay",
        ),
        Memory(
            "Lilly asked if we have jumper cables she could borrow",
            group="hay",
        ),
        Memory(
            "Brian texted that he found Alex's sunglasses in his truck",
            group="hay",
        ),
        Memory(
            "Sofi told Lilly she wants to learn how to bake cookies",
            group="hay",
        ),
        Memory(
            "Alex helped Brian move the couch from the garage to the living room",
            group="hay",
        ),
        Memory(
            "Lilly is volunteering at the school library on Wednesday afternoons",
            group="hay",
        ),
        Memory(
            "Brian mentioned his commute takes over an hour now with construction",
            group="hay",
        ),
        Memory(
            "Sofi asked Alex to help her practice spelling words after dinner",
            group="hay",
        ),
        Memory(
            "Lilly ordered matching rain boots for the kids from an online sale",
            group="hay",
        ),
        Memory(
            "Alex ran into Brian at the hardware store on Saturday morning",
            group="hay",
        ),
        Memory(
            "Sofi wants to bring cupcakes to school for the class celebration",
            group="hay",
        ),
        Memory(
            "Brian and Lilly hosted a game night and invited the whole street",
            group="hay",
        ),
        Memory(
            "Alex asked Lilly for the name of the piano teacher she recommended",
            group="hay",
        ),
        Memory(
            "Sofi has been collecting rocks from the playground to paint at home",
            group="hay",
        ),
        Memory(
            "Brian lent Alex his power drill for the weekend",
            group="hay",
        ),
        Memory(
            "Lilly texted a reminder about the school picture day on Wednesday",
            group="hay",
        ),
        Memory(
            "Alex noticed Sofi has been reading more"
            " since they started the reward chart",
            group="hay",
        ),
        Memory(
            "Brian said he saw a coyote near the bike path behind the school",
            group="hay",
        ),
        Memory(
            "Sofi left her backpack at Brian and Lilly's house after the playdate",
            group="hay",
        ),
        Memory(
            "Lilly suggested carpooling to the school field trip next Tuesday",
            group="hay",
        ),
        Memory(
            "Alex picked up extra lawn bags for Brian from the hardware store",
            group="hay",
        ),
        Memory(
            "Brian asked Alex for a recommendation for a good barber nearby",
            group="hay",
        ),
        Memory(
            "Sofi and Lilly made friendship bracelets at the craft fair",
            group="hay",
        ),
        Memory(
            "Alex sent Brian the link to the neighborhood watch signup page",
            group="hay",
        ),
        Memory(
            "Lilly mentioned her car is making a strange noise when turning left",
            group="hay",
        ),
        Memory(
            "Brian grilled burgers for the families after the kids finished homework",
            group="hay",
        ),
        Memory(
            "Sofi asked if she can walk to school with the other kids in the morning",
            group="hay",
        ),
        Memory(
            "Alex forwarded the school district calendar to Brian and Lilly",
            group="hay",
        ),
        Memory(
            "Lilly dropped off a bag of hand-me-down clothes for Sofi",
            group="hay",
        ),
        Memory(
            "Brian asked about the trash pickup schedule change starting in April",
            group="hay",
        ),
        Memory(
            "Sofi made a card for Lilly thanking her for the ride home yesterday",
            group="hay",
        ),
        Memory(
            "Alex offered to water Brian's plants while they are out of town",
            group="hay",
        ),
        Memory(
            "Lilly wants to organize a neighborhood potluck for the long weekend",
            group="hay",
        ),
        Memory(
            "Brian installed a basketball hoop in the driveway for the kids",
            group="hay",
        ),
        Memory(
            "Sofi practiced her jump rope skills at recess and learned double dutch",
            group="hay",
        ),
        Memory(
            "Alex and Lilly compared notes on which summer camp has the best hours",
            group="hay",
        ),
        Memory(
            "Brian offered to drive Sofi's group to the"
            " field trip since his car fits more",
            group="hay",
        ),
        Memory(
            "Sofi earned a star sticker for helping clean up the classroom",
            group="hay",
        ),
        Memory(
            "Lilly signed Brian up for the dads-only volunteer day at school",
            group="hay",
        ),
        Memory(
            "Alex is thinking about coaching Sofi's soccer team next season",
            group="hay",
        ),
        Memory(
            "Brian and Alex are splitting the cost of"
            " renting a bounce house for the block party",
            group="hay",
        ),
        Memory(
            "Sofi told Alex she wants a pet hamster for her next birthday",
            group="hay",
        ),
        # ================================================================
        # TIER 3: Hard negatives (indices 151-166) -- 16 memories
        # Thematically adjacent to chain topics but not part of any chain
        # ================================================================
        # Health-adjacent (not about Pat's allergy or Mom's knee)
        Memory(
            "Brian's doctor recommended he cut back on caffeine and drink more water",
            group="hay",
        ),
        Memory(
            "Lilly mentioned her mother's cholesterol"
            " numbers improved after the new medication",
            group="hay",
        ),
        Memory(
            "Alex scheduled his annual physical for the second week of April",
            group="hay",
        ),
        Memory(
            "Picked up multivitamins and fish oil capsules from the pharmacy",
            group="hay",
        ),
        Memory(
            "Brian tweaked his knee during the half marathon and is icing it nightly",
            group="hay",
        ),
        Memory(
            "The urgent care clinic on Oak Street now accepts walk-ins on weekends",
            group="hay",
        ),
        # School-adjacent (not about the science fair volcano)
        Memory(
            "Sofi's classmate Emma invited her to a pool party next Friday afternoon",
            group="hay",
        ),
        Memory(
            "Sofi practiced her piano recital piece for thirty minutes this morning",
            group="hay",
        ),
        Memory(
            "Sofi asked if she can join the after-school art club with her friend Maya",
            group="hay",
        ),
        Memory(
            "The school nurse sent home a note about a flu outbreak in the third grade",
            group="hay",
        ),
        Memory(
            "Lilly enrolled her kids in the weekend coding workshop at the library",
            group="hay",
        ),
        Memory(
            "Sofi's teacher said the class is putting on"
            " a short play before winter break",
            group="hay",
        ),
        # Restaurant/food-adjacent (not about Thai or peanut oil)
        Memory(
            "Lisa reserved a table at the new Italian place for Saturday evening",
            group="hay",
        ),
        Memory(
            "Tried a new slow cooker chili recipe and it turned out great",
            group="hay",
        ),
        Memory(
            "Rachel found a bakery downtown that does custom layer cakes",
            group="hay",
        ),
        # Garden-adjacent (not about Mom's raised beds)
        Memory(
            "Sam built a raised garden bed out of cedar planks last Saturday",
            group="hay",
        ),
    ],
    associations=[
        # ================================================================
        # Chain A associations (dinner safety)
        # 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6
        # ================================================================
        Association(
            source_index=0,
            target_index=1,
            association_type="entity",
            forward_strength=0.8,
            backward_strength=0.5,
        ),
        Association(
            source_index=1,
            target_index=2,
            association_type="entity",
            forward_strength=0.6,
            backward_strength=0.4,
        ),
        Association(
            source_index=2,
            target_index=3,
            association_type="entity",
            forward_strength=0.85,
            backward_strength=0.6,
        ),
        Association(
            source_index=3,
            target_index=4,
            association_type="entity",
            forward_strength=0.8,
            backward_strength=0.55,
        ),
        Association(
            source_index=4,
            target_index=5,
            association_type="entity",
            forward_strength=0.75,
            backward_strength=0.5,
        ),
        Association(
            source_index=5,
            target_index=6,
            association_type="entity",
            forward_strength=0.7,
            backward_strength=0.45,
        ),
        # ================================================================
        # Chain B associations (birthday gift)
        # 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13
        # ================================================================
        Association(
            source_index=7,
            target_index=8,
            association_type="entity",
            forward_strength=0.85,
            backward_strength=0.6,
        ),
        Association(
            source_index=8,
            target_index=9,
            association_type="entity",
            forward_strength=0.8,
            backward_strength=0.55,
        ),
        Association(
            source_index=9,
            target_index=10,
            association_type="entity",
            forward_strength=0.75,
            backward_strength=0.5,
        ),
        Association(
            source_index=10,
            target_index=11,
            association_type="entity",
            forward_strength=0.7,
            backward_strength=0.45,
        ),
        Association(
            source_index=11,
            target_index=12,
            association_type="entity",
            forward_strength=0.65,
            backward_strength=0.4,
        ),
        Association(
            source_index=12,
            target_index=13,
            association_type="entity",
            forward_strength=0.8,
            backward_strength=0.55,
        ),
        # ================================================================
        # Chain C associations (school project)
        # 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20
        # ================================================================
        Association(
            source_index=14,
            target_index=15,
            association_type="entity",
            forward_strength=0.9,
            backward_strength=0.65,
        ),
        Association(
            source_index=15,
            target_index=16,
            association_type="entity",
            forward_strength=0.8,
            backward_strength=0.55,
        ),
        Association(
            source_index=16,
            target_index=17,
            association_type="entity",
            forward_strength=0.75,
            backward_strength=0.5,
        ),
        Association(
            source_index=17,
            target_index=18,
            association_type="entity",
            forward_strength=0.85,
            backward_strength=0.6,
        ),
        Association(
            source_index=18,
            target_index=19,
            association_type="entity",
            forward_strength=0.7,
            backward_strength=0.45,
        ),
        Association(
            source_index=19,
            target_index=20,
            association_type="entity",
            forward_strength=0.65,
            backward_strength=0.4,
        ),
        # ================================================================
        # Cross-chain noise associations (haystack-to-haystack)
        # Weak links between Tier 2 memories sharing family entities.
        # Tests whether iterative spreading gets distracted by noise.
        # ================================================================
        Association(
            source_index=101,
            target_index=105,
            association_type="entity",
            forward_strength=0.3,
            backward_strength=0.2,
        ),
        Association(
            source_index=104,
            target_index=109,
            association_type="entity",
            forward_strength=0.25,
            backward_strength=0.15,
        ),
        Association(
            source_index=115,
            target_index=126,
            association_type="entity",
            forward_strength=0.3,
            backward_strength=0.2,
        ),
    ],
    queries=[
        Query(
            text=(
                "What should we keep in mind for dinner"
                " with Pat at the Thai restaurant?"
            ),
            description=(
                "Should find the full dinner safety chain"
                " including urgent care location"
            ),
            expected_indices=[0, 1, 2, 3, 4, 5, 6],
            unexpected_indices=[
                7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            ],
            expected_2hop=[0, 1, 2, 3, 4],
            expected_iterative=[5, 6],
        ),
        Query(
            text=(
                "What would be a good birthday gift for Mom"
                " given her health situation?"
            ),
            description=(
                "Should find birthday + garden + store details"
                " including delivery policy"
            ),
            expected_indices=[7, 8, 9, 10, 11, 12, 13],
            unexpected_indices=[
                0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 20,
            ],
            expected_2hop=[7, 8, 9, 10, 11],
            expected_iterative=[12, 13],
        ),
        Query(
            text=(
                "What do we need to get ready for Sofi's"
                " science fair volcano project?"
            ),
            description=(
                "Should find project + supplies + store details"
                " including parking"
            ),
            expected_indices=[14, 15, 16, 17, 18, 19, 20],
            unexpected_indices=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            ],
            expected_2hop=[14, 15, 16, 17, 18],
            expected_iterative=[19, 20],
        ),
    ],
)
