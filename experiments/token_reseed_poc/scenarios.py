"""Situational grounding scenario for token re-seeding.

Tests whether LLM-generated tokens can surface memories that are
causally critical but semantically invisible to the query. Four
chains where each endpoint shares zero vocabulary with its query,
connected only by situational implication (safety risk, logistical
dependency, preparation requirement).

Chain A: Dining safety -- peanut allergy -> Thai restaurant -> EpiPen location
Chain B: Travel logistics -- Denver trip -> pet care -> spare key location
Chain C: Appointment prep -- doctor visit -> BP log request -> BP cuff location
Chain D: Heating safety -- furnace crack -> CO risk -> repair -> risk recurrence
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    content: str
    group: str = ""  # "chain_a", "chain_b", "chain_c", "chain_d", "hay"
    significance: float = 0.5


@dataclass
class Query:
    text: str
    description: str
    expected_indices: list[int] = field(default_factory=list)
    expected_iterative: list[int] = field(default_factory=list)
    unexpected_indices: list[int] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    description: str
    memories: list[Memory]
    queries: list[Query]


SCENARIO = Scenario(
    name="situational_grounding",
    description=(
        "Four chains with situational connections. "
        "Chain endpoints share zero semantic overlap with queries. "
        "Connections are causal implications and logistical dependencies, "
        "not entity-name co-references. Tests whether the LLM can "
        "recognize real-world implications at write time and create "
        "tokens that surface invisible-but-critical memories at query time."
    ),
    memories=[
        # ================================================================
        # CHAIN A: Dining safety (indices 0-2, group="chain_a")
        # Query: "Should we try the new Thai place for team lunch?"
        # Path: peanut allergy -> peanut oil in Thai food -> EpiPen location
        # ================================================================
        Memory(
            "Alex has a severe peanut allergy diagnosed in childhood",
            group="chain_a",
            significance=0.7,
        ),
        Memory(
            "The new Thai place downtown uses peanut oil in all their stir-fry dishes",
            group="chain_a",
            significance=0.6,
        ),
        Memory(
            "Alex's EpiPen is in the outer pocket of his laptop bag"
            " and expires September fifteenth",
            group="chain_a",
            significance=0.8,
        ),
        # ================================================================
        # CHAIN B: Travel logistics (indices 3-5, group="chain_b")
        # Query: "What needs to happen before the Denver trip?"
        # Path: Denver flights -> dog care while away -> spare key location
        # ================================================================
        Memory(
            "Booked flights to Denver for the conference, Thursday to Sunday",
            group="chain_b",
            significance=0.6,
        ),
        Memory(
            "Need to ask Dave about watching the dog while we are away next week",
            group="chain_b",
            significance=0.6,
        ),
        Memory(
            "Dave has a spare key to the house, it is the one"
            " under the blue flowerpot on the back porch",
            group="chain_b",
            significance=0.7,
        ),
        # ================================================================
        # CHAIN C: Appointment prep (indices 6-8, group="chain_c")
        # Query: "What should we bring for Sarah's doctor visit next Thursday?"
        # Path: Dr. Patel follow-up -> bring BP log -> BP cuff location
        # ================================================================
        Memory(
            "Sarah has a follow-up with Dr. Patel next Thursday afternoon",
            group="chain_c",
            significance=0.6,
        ),
        Memory(
            "Dr. Patel asked Sarah to bring her blood pressure log from the last month",
            group="chain_c",
            significance=0.6,
        ),
        Memory(
            "The blood pressure cuff is in the hall closet"
            " and the batteries were replaced last week",
            group="chain_c",
            significance=0.7,
        ),
        # ================================================================
        # CHAIN D: Home heating safety (indices 9-13, group="chain_d")
        # Query: "Is there any safety concern with the heating system?"
        # Path: furnace crack -> CO alarm -> replacement ordered ->
        #       repair certified (revise 1) -> CO returns (revise 2)
        # Tests: create, extend, revise (resolve), revise (recur)
        # ================================================================
        Memory(
            "The annual furnace inspection flagged a cracked heat exchanger"
            " in the basement unit",
            group="chain_d",
            significance=0.7,
        ),
        Memory(
            "Carbon monoxide detector in the basement went off twice overnight",
            group="chain_d",
            significance=0.8,
        ),
        Memory(
            "Ordered a replacement heat exchanger from the HVAC supplier"
            " and delivery is expected Friday",
            group="chain_d",
            significance=0.6,
        ),
        Memory(
            "The HVAC technician replaced the heat exchanger and it passed"
            " the safety certification",
            group="chain_d",
            significance=0.6,
        ),
        Memory(
            "The basement CO sensor is showing elevated readings again"
            " two weeks after the heat exchanger repair",
            group="chain_d",
            significance=0.9,
        ),
        # ================================================================
        # HAYSTACK (indices 14+, group="hay")
        # ================================================================
        # --- TIER 1: Inert filler (indices 14-213, ~200 memories) ---
        # Category 1: Work/office (14-27)
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
            "The design team wants feedback on the new landing page mockups",
            group="hay",
        ),
        Memory(
            "Tom booked the training room for the onboarding session next Friday",
            group="hay",
        ),
        # Category 2: Finance/bills (28-40)
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
            "Mark said the quarterly estimated taxes are due on the fifteenth",
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
            "Time to start organizing receipts for tax prep this year",
            group="hay",
        ),
        Memory(
            "Sam found a better rate on car insurance through a different provider",
            group="hay",
        ),
        # Category 3: Home maintenance (41-54)
        Memory(
            "The kitchen faucet has been dripping all week and needs a new washer",
            group="hay",
        ),
        Memory(
            "Need to get a quote from the electrician for the basement wiring",
            group="hay",
        ),
        Memory(
            "Need to replace the furnace filter before the heating season starts",
            group="hay",
        ),
        Memory(
            "The gutter on the south side of the house is sagging after the storm",
            group="hay",
        ),
        Memory(
            "Tom picked up caulk and weatherstripping from the supply store",
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
            "Need to schedule the annual HVAC inspection before summer hits",
            group="hay",
        ),
        Memory(
            "Lisa got a quote for refinishing the hardwood floors upstairs",
            group="hay",
        ),
        Memory(
            "The smoke detector batteries need replacing in the upstairs hallway",
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
        # Category 4: Technology/devices (55-67)
        Memory(
            "The laptop has been running slow since the last operating system update",
            group="hay",
        ),
        Memory(
            "Rachel set up the new mesh wifi router and coverage improved a lot",
            group="hay",
        ),
        Memory(
            "Need to update the password for the home security system panel",
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
            "The phone screen protector cracked and needs replacing soon",
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
        Memory(
            "The Bluetooth speaker stopped pairing with the phone after the update",
            group="hay",
        ),
        # Category 5: Travel/trips (68-79, NOT Denver, NOT conferences)
        Memory(
            "Booked the hotel near the waterfront for the October anniversary trip",
            group="hay",
        ),
        Memory(
            "Tom suggested taking the scenic route through the mountains instead",
            group="hay",
        ),
        Memory(
            "Need to check if the passport needs renewing before the summer trip",
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
        Memory(
            "Sam wants to do a day trip to the vineyards this Saturday afternoon",
            group="hay",
        ),
        Memory(
            "Gas prices along the interstate are about twenty cents cheaper now",
            group="hay",
        ),
        Memory(
            "Tom found cheap flights to Austin for the long weekend in November",
            group="hay",
        ),
        # Category 6: Weather/seasonal (80-91)
        Memory(
            "The forecast says rain all next week so outdoor plans are off",
            group="hay",
        ),
        Memory(
            "Need to winterize the outdoor faucets before the first freeze hits",
            group="hay",
        ),
        Memory(
            "Tom brought in the patio furniture before the storm last night",
            group="hay",
        ),
        Memory(
            "The spring pollen count is already high and everyone is sneezing",
            group="hay",
        ),
        Memory(
            "Chris mentioned the ski resort closes at the end of March this year",
            group="hay",
        ),
        Memory(
            "Should get the snow tires swapped out for all-seasons this week",
            group="hay",
        ),
        Memory(
            "The weather cleared up enough to finally mow the back lawn",
            group="hay",
        ),
        Memory(
            "Karen stocked up on ice melt and salt for the driveway",
            group="hay",
        ),
        Memory(
            "Mark said the lake is warm enough for swimming by mid-June",
            group="hay",
        ),
        Memory(
            "Need to clean out the rain gutters after the leaf drop this fall",
            group="hay",
        ),
        Memory(
            "The frost warning means covering the garden beds tonight",
            group="hay",
        ),
        Memory(
            "Janet checked the ten-day forecast and it looks clear for the hike",
            group="hay",
        ),
        # Category 7: Food/cooking (92-104, NOT Thai, NOT peanuts, NOT allergies)
        Memory(
            "Tried a new slow cooker chili recipe and it turned out great",
            group="hay",
        ),
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
            "Chris is experimenting with smoking brisket on his new pellet grill",
            group="hay",
        ),
        Memory(
            "Meal prepped five lunches for the week including chicken and rice",
            group="hay",
        ),
        Memory(
            "Rachel found a bakery downtown that does custom layer cakes",
            group="hay",
        ),
        Memory(
            "The grocery delivery got the wrong brand of yogurt again",
            group="hay",
        ),
        Memory(
            "Karen made a batch of granola bars for snacks during the road trip",
            group="hay",
        ),
        Memory(
            "Need to defrost the ground turkey overnight for tomorrow's tacos",
            group="hay",
        ),
        Memory(
            "Sam started a sourdough starter and it finally looks active",
            group="hay",
        ),
        Memory(
            "Lisa picked up a rotisserie chicken for tonight since it was late",
            group="hay",
        ),
        Memory(
            "Mark grilled burgers and corn on the cob for the neighbors last night",
            group="hay",
        ),
        # Category 8: Hobbies/recreation (105-117)
        Memory(
            "Finished reading the third book in the mystery series last night",
            group="hay",
        ),
        Memory(
            "Tom invited me to his woodworking shop to see the table he built",
            group="hay",
        ),
        Memory(
            "The photography club is meeting at the botanical gardens this Sunday",
            group="hay",
        ),
        Memory(
            "Chris picked up new guitar strings and a capo from the music store",
            group="hay",
        ),
        Memory(
            "Karen started a jigsaw puzzle with a thousand pieces over the weekend",
            group="hay",
        ),
        Memory(
            "Mark signed up for the 5K charity run happening downtown next month",
            group="hay",
        ),
        Memory(
            "Lisa is learning watercolor painting through an online course",
            group="hay",
        ),
        Memory(
            "The book club picked a historical fiction novel for this month",
            group="hay",
        ),
        Memory(
            "Sam downloaded a new podcast about unsolved aviation mysteries",
            group="hay",
        ),
        Memory(
            "Janet found a great spot for birdwatching along the river trail",
            group="hay",
        ),
        Memory(
            "Rachel joined a weekend cycling group that rides the lakefront path",
            group="hay",
        ),
        Memory(
            "Tom is restoring a vintage radio he found at the flea market",
            group="hay",
        ),
        Memory(
            "Karen started doing crossword puzzles every morning with her coffee",
            group="hay",
        ),
        # Category 9: Pets/animals (118-129, NOT dog-sitting, NOT Dave watching pets)
        Memory(
            "The cat knocked over the plant stand again this morning",
            group="hay",
        ),
        Memory(
            "Need to schedule the annual vet checkup for the dog next month",
            group="hay",
        ),
        Memory(
            "Tom bought a new scratching post for his cats from the pet store",
            group="hay",
        ),
        Memory(
            "The dog ate something weird at the park and was sick all evening",
            group="hay",
        ),
        Memory(
            "Karen said her golden retriever graduates obedience class Thursday",
            group="hay",
        ),
        Memory(
            "Lisa adopted a rescue cat from the shelter on Maple Street",
            group="hay",
        ),
        Memory(
            "Need to refill the bird feeder in the backyard before it snows again",
            group="hay",
        ),
        Memory(
            "The fish tank filter is making a buzzing sound and needs cleaning",
            group="hay",
        ),
        Memory(
            "Sam found a good brand of grain-free kibble at the pet supply shop",
            group="hay",
        ),
        Memory(
            "Janet trimmed the dog's nails at home instead of going to the groomer",
            group="hay",
        ),
        Memory(
            "Chris set up an automatic feeder for the cat while working late shifts",
            group="hay",
        ),
        Memory(
            "Mark's dog learned to shake hands after two weeks of training treats",
            group="hay",
        ),
        # Category 10: Garden/yard (130-142)
        Memory(
            "The tomato plants are finally sprouting in the raised bed",
            group="hay",
        ),
        Memory(
            "Tom bought a bag of wildflower seeds for the empty patch by the fence",
            group="hay",
        ),
        Memory(
            "The lawn service quoted eighty dollars a month for weekly mowing",
            group="hay",
        ),
        Memory(
            "Lisa trimmed the hedge along the property line over the weekend",
            group="hay",
        ),
        Memory(
            "Sam built a raised garden bed out of cedar planks last Saturday",
            group="hay",
        ),
        Memory(
            "Need to treat the roses for aphids before they spread further",
            group="hay",
        ),
        Memory(
            "Karen planted basil and oregano in the herb garden by the kitchen door",
            group="hay",
        ),
        Memory(
            "The sprinkler system needs adjusting since it is hitting the sidewalk",
            group="hay",
        ),
        Memory(
            "Mark raked up three bags of leaves from the front yard on Saturday",
            group="hay",
        ),
        Memory(
            "Janet composted the kitchen scraps and added them to the garden beds",
            group="hay",
        ),
        Memory(
            "Chris aerated the lawn last weekend and it already looks healthier",
            group="hay",
        ),
        Memory(
            "The sunflowers along the back fence are almost six feet tall now",
            group="hay",
        ),
        Memory(
            "Rachel pruned the apple tree and got a full wheelbarrow of branches",
            group="hay",
        ),
        # Category 11: Transportation/commute (143-154)
        Memory(
            "Traffic on the highway was backed up for forty minutes this morning",
            group="hay",
        ),
        Memory(
            "Need to get the oil changed before the next road trip",
            group="hay",
        ),
        Memory(
            "Tom said the tire pressure light came on during his commute yesterday",
            group="hay",
        ),
        Memory(
            "The bus route changed and now skips the stop near the office",
            group="hay",
        ),
        Memory(
            "Karen carpools with three coworkers and they rotate driving each week",
            group="hay",
        ),
        Memory(
            "The car wash on Elm Street has a half-price deal on Tuesdays",
            group="hay",
        ),
        Memory(
            "Mark biked to work today since the weather was nice enough",
            group="hay",
        ),
        Memory(
            "Lisa said the parking garage raised its monthly rate by fifteen dollars",
            group="hay",
        ),
        Memory(
            "Need to replace the windshield wipers before the next rainstorm",
            group="hay",
        ),
        Memory(
            "Sam's commute doubled since the bridge construction started last month",
            group="hay",
        ),
        Memory(
            "Janet found a shortcut through the side streets that saves ten minutes",
            group="hay",
        ),
        Memory(
            "Chris got a flat tire on the way to work and had to call roadside assist",
            group="hay",
        ),
        # Category 12: Neighborhood/community (155-166)
        Memory(
            "The HOA sent a reminder about keeping trash cans behind the fence",
            group="hay",
        ),
        Memory(
            "Tom organized a neighborhood cleanup day for the first Saturday in May",
            group="hay",
        ),
        Memory(
            "The community pool opens Memorial Day weekend this year",
            group="hay",
        ),
        Memory(
            "Karen mentioned the speed bumps on Oak Street are finally getting fixed",
            group="hay",
        ),
        Memory(
            "The new family across the street has twin boys in kindergarten",
            group="hay",
        ),
        Memory(
            "Mark volunteered to be the block captain for the neighborhood watch",
            group="hay",
        ),
        Memory(
            "Lisa said the streetlight at the corner has been out for two weeks",
            group="hay",
        ),
        Memory(
            "The community garden has open plots available for the spring season",
            group="hay",
        ),
        Memory(
            "Sam attended the town hall meeting about the proposed bike lane",
            group="hay",
        ),
        Memory(
            "Janet brought cookies to welcome the new neighbors on Birch Lane",
            group="hay",
        ),
        Memory(
            "Chris signed up for the community disaster preparedness committee",
            group="hay",
        ),
        Memory(
            "Rachel mentioned the library branch is hosting free movie nights",
            group="hay",
        ),
        # Category 13: Shopping/errands (167-178)
        Memory(
            "Need to return the curtains that were the wrong size for the window",
            group="hay",
        ),
        Memory(
            "The package from the online order has been stuck in transit for a week",
            group="hay",
        ),
        Memory(
            "Tom picked up lightbulbs and cleaning supplies on his way home",
            group="hay",
        ),
        Memory(
            "Karen found the exact bookshelf model on clearance at the furniture store",
            group="hay",
        ),
        Memory(
            "Need to drop off the donation bags at the thrift store this weekend",
            group="hay",
        ),
        Memory(
            "Lisa ordered new bath towels since the old ones are fraying badly",
            group="hay",
        ),
        Memory(
            "Mark went to three stores before finding the right size air filter",
            group="hay",
        ),
        Memory(
            "Sam said the shoe repair shop on Main Street does great work cheaply",
            group="hay",
        ),
        Memory(
            "Janet needs to exchange the jacket she got for a smaller size",
            group="hay",
        ),
        Memory(
            "Rachel stocked up on paper towels and laundry detergent at the sale",
            group="hay",
        ),
        Memory(
            "Chris ordered a replacement part for the vacuum cleaner online",
            group="hay",
        ),
        Memory(
            "The dry cleaner on Second Street closes at five on Saturdays",
            group="hay",
        ),
        # Category 14: Entertainment (179-190)
        Memory(
            "The new season of that detective show starts streaming on Friday",
            group="hay",
        ),
        Memory(
            "Tom got tickets to the outdoor concert at the amphitheater next month",
            group="hay",
        ),
        Memory(
            "Karen recommended a documentary about ocean conservation on Netflix",
            group="hay",
        ),
        Memory(
            "Mark and Lisa saw the new action movie and said it was worth seeing",
            group="hay",
        ),
        Memory(
            "The indie theater downtown is showing classic films every Wednesday",
            group="hay",
        ),
        Memory(
            "Sam subscribed to the audiobook service and listens during his commute",
            group="hay",
        ),
        Memory(
            "Janet mentioned the trivia night at the pub is moving to Thursdays",
            group="hay",
        ),
        Memory(
            "Rachel downloaded a new strategy game on her phone and is hooked on it",
            group="hay",
        ),
        Memory(
            "Chris organized board game night and is hosting it at his place",
            group="hay",
        ),
        Memory(
            "The comedy club on River Road has a two-for-one deal this weekend",
            group="hay",
        ),
        Memory(
            "Lisa binge-watched the entire first season of the space drama",
            group="hay",
        ),
        Memory(
            "Tom entered a local photography contest with his sunset shots",
            group="hay",
        ),
        # Category 15: Fitness/exercise (191-201)
        Memory(
            "Started going to the six AM spin class three times a week",
            group="hay",
        ),
        Memory(
            "Tom ran his fastest mile time this morning at six minutes"
            " and forty seconds",
            group="hay",
        ),
        Memory(
            "Karen signed up for the beginner yoga class at the rec center",
            group="hay",
        ),
        Memory(
            "Mark is training for the half marathon and hit fifteen miles this week",
            group="hay",
        ),
        Memory(
            "The gym replaced all the treadmills with newer models last month",
            group="hay",
        ),
        Memory(
            "Lisa and Rachel started a walking group that meets at the park at seven",
            group="hay",
        ),
        Memory(
            "Sam pulled a hamstring during pickup basketball and is icing it",
            group="hay",
        ),
        Memory(
            "Janet bought a set of resistance bands for home workouts",
            group="hay",
        ),
        Memory(
            "Chris swims laps at the community pool every Tuesday and Thursday",
            group="hay",
        ),
        Memory(
            "The rock climbing gym downtown offers a free trial class on Saturdays",
            group="hay",
        ),
        Memory(
            "Rachel tracked ten thousand steps every day this month on her watch",
            group="hay",
        ),
        # Category 16: Education/learning (202-213)
        Memory(
            "Tom enrolled in an evening accounting course at the community college",
            group="hay",
        ),
        Memory(
            "Karen is taking an online certification in project management",
            group="hay",
        ),
        Memory(
            "Mark attended a weekend workshop on home brewing techniques",
            group="hay",
        ),
        Memory(
            "Lisa signed up for the pottery class that starts next Tuesday evening",
            group="hay",
        ),
        Memory(
            "Sam finished the Python tutorial series and started"
            " building a side project",
            group="hay",
        ),
        Memory(
            "Janet is studying for the real estate licensing exam in April",
            group="hay",
        ),
        Memory(
            "Chris completed the CPR recertification course at the fire station",
            group="hay",
        ),
        Memory(
            "Rachel started learning Spanish through a language app this month",
            group="hay",
        ),
        Memory(
            "The library offers free resume writing workshops every other Wednesday",
            group="hay",
        ),
        Memory(
            "Tom borrowed three books on landscape design from the public library",
            group="hay",
        ),
        Memory(
            "Karen subscribed to an online platform for data science courses",
            group="hay",
        ),
        Memory(
            "Mark is auditing a history lecture series at the university for free",
            group="hay",
        ),
        # --- TIER 2: Moderate interference (indices 214-283, ~70 memories) ---
        # Family entity noise: Alex, Dave, Sarah, Sofi in unrelated contexts
        Memory(
            "Alex emailed the team about the Q2 milestone deadline",
            group="hay",
        ),
        Memory(
            "Dave mentioned the neighborhood block party is next month",
            group="hay",
        ),
        Memory(
            "Sarah practiced her piano recital piece this morning",
            group="hay",
        ),
        Memory(
            "Sofi drew a picture of a rainbow for art class",
            group="hay",
        ),
        Memory(
            "Alex and Dave watched the playoff game together on Sunday",
            group="hay",
        ),
        Memory(
            "Sarah asked if we have the receipt for the shoes she wants to return",
            group="hay",
        ),
        Memory(
            "Dave signed up for a woodworking class at the community center",
            group="hay",
        ),
        Memory(
            "Alex grabbed coffee with Dave after the morning standup",
            group="hay",
        ),
        Memory(
            "Sofi finished her chapter book and wants to start the next one",
            group="hay",
        ),
        Memory(
            "Sarah texted that she will be late picking up the kids from practice",
            group="hay",
        ),
        Memory(
            "Dave texted that he found Alex's sunglasses in his truck",
            group="hay",
        ),
        Memory(
            "Sofi told Sarah she wants to learn how to bake cookies",
            group="hay",
        ),
        Memory(
            "Alex helped Dave move the couch from the garage to the living room",
            group="hay",
        ),
        Memory(
            "Sarah is volunteering at the school library on Wednesday afternoons",
            group="hay",
        ),
        Memory(
            "Dave mentioned his commute takes over an hour now with construction",
            group="hay",
        ),
        Memory(
            "Sofi asked Alex to help her practice spelling words after dinner",
            group="hay",
        ),
        Memory(
            "Sarah ordered matching rain boots for the kids from an online sale",
            group="hay",
        ),
        Memory(
            "Alex ran into Dave at the coffee shop on Saturday morning",
            group="hay",
        ),
        Memory(
            "Sofi wants to bring cupcakes to school for the class celebration",
            group="hay",
        ),
        Memory(
            "Dave and Sarah hosted a game night and invited the whole street",
            group="hay",
        ),
        Memory(
            "Alex asked Sarah for the name of the plumber she recommended",
            group="hay",
        ),
        Memory(
            "Sofi has been collecting rocks from the playground to paint at home",
            group="hay",
        ),
        Memory(
            "Dave lent Alex his power drill for the weekend project",
            group="hay",
        ),
        Memory(
            "Sarah texted a reminder about the school picture day on Wednesday",
            group="hay",
        ),
        Memory(
            "Alex noticed Sofi has been reading more"
            " since they started the reward chart",
            group="hay",
        ),
        Memory(
            "Dave said he saw a coyote near the bike path behind the school",
            group="hay",
        ),
        Memory(
            "Sofi left her backpack at Dave and Sarah's house after the playdate",
            group="hay",
        ),
        Memory(
            "Sarah suggested carpooling to the school field trip next Tuesday",
            group="hay",
        ),
        Memory(
            "Alex picked up extra lawn bags for Dave from the garden supply store",
            group="hay",
        ),
        Memory(
            "Dave asked Alex for a recommendation for a good barber nearby",
            group="hay",
        ),
        Memory(
            "Sofi and Sarah made friendship bracelets at the craft fair",
            group="hay",
        ),
        Memory(
            "Alex sent Dave the link to the neighborhood watch signup page",
            group="hay",
        ),
        Memory(
            "Sarah mentioned her car is making a strange noise when turning left",
            group="hay",
        ),
        Memory(
            "Dave grilled burgers for the families after the kids finished homework",
            group="hay",
        ),
        Memory(
            "Sofi asked if she can walk to school with the other kids in the morning",
            group="hay",
        ),
        Memory(
            "Alex forwarded the school district calendar to Dave and Sarah",
            group="hay",
        ),
        Memory(
            "Sarah dropped off a bag of hand-me-down clothes for Sofi",
            group="hay",
        ),
        Memory(
            "Dave asked about the trash pickup schedule change starting in April",
            group="hay",
        ),
        Memory(
            "Sofi made a card for Sarah thanking her for the ride home yesterday",
            group="hay",
        ),
        Memory(
            "Alex offered to water Dave's plants while they are out of town",
            group="hay",
        ),
        Memory(
            "Sarah wants to organize a neighborhood potluck for the long weekend",
            group="hay",
        ),
        Memory(
            "Dave installed a basketball hoop in the driveway for the kids",
            group="hay",
        ),
        Memory(
            "Sofi practiced her jump rope skills at recess and learned double dutch",
            group="hay",
        ),
        Memory(
            "Alex and Sarah compared notes on which summer camp has the best hours",
            group="hay",
        ),
        Memory(
            "Dave offered to drive the kids to the field trip since his car fits more",
            group="hay",
        ),
        Memory(
            "Sofi earned a star sticker for helping clean up the classroom",
            group="hay",
        ),
        Memory(
            "Sarah signed Dave up for the dads-only volunteer day at school",
            group="hay",
        ),
        Memory(
            "Alex is thinking about coaching Sofi's soccer team next season",
            group="hay",
        ),
        Memory(
            "Dave and Alex are splitting the cost of renting a bounce house",
            group="hay",
        ),
        Memory(
            "Sofi told Alex she wants a pet hamster for her next birthday",
            group="hay",
        ),
        Memory(
            "Sarah picked up craft supplies for the kids' weekend art project",
            group="hay",
        ),
        Memory(
            "Dave fixed the broken fence post on Alex's side of the property line",
            group="hay",
        ),
        Memory(
            "Sofi showed Dave her science project about volcanoes at school",
            group="hay",
        ),
        Memory(
            "Alex and Sarah discussed whether to sign the kids up for swim lessons",
            group="hay",
        ),
        Memory(
            "Dave brought over his leaf blower since ours is being repaired",
            group="hay",
        ),
        Memory(
            "Sofi asked Sarah to braid her hair before the school dance",
            group="hay",
        ),
        Memory(
            "Alex texted Dave about the recycling schedule change next month",
            group="hay",
        ),
        Memory(
            "Sarah found a free outdoor movie night at the park this Friday",
            group="hay",
        ),
        Memory(
            "Dave said the new pizza place on Third Street is surprisingly good",
            group="hay",
        ),
        Memory(
            "Sofi and Alex built a blanket fort in the living room after dinner",
            group="hay",
        ),
        Memory(
            "Sarah mentioned she is training for a charity walk in October",
            group="hay",
        ),
        Memory(
            "Dave recommended a podcast about home renovation for beginners",
            group="hay",
        ),
        Memory(
            "Sofi won second place in the class spelling bee last Friday",
            group="hay",
        ),
        Memory(
            "Alex asked Dave to help him move the bookshelf to the other wall",
            group="hay",
        ),
        Memory(
            "Sarah baked banana bread and brought some over for the neighbors",
            group="hay",
        ),
        Memory(
            "Dave picked up Sofi from soccer practice when Alex was running late",
            group="hay",
        ),
        Memory(
            "Sofi asked if the family can go to the aquarium this weekend",
            group="hay",
        ),
        Memory(
            "Alex and Dave talked about starting a weekend fishing routine",
            group="hay",
        ),
        # --- TIER 3: Hard negatives (indices 284-313, ~30 memories) ---
        # Health-adjacent (not about peanut allergy or Dr. Patel)
        Memory(
            "Rachel's doctor recommended she cut back on caffeine",
            group="hay",
        ),
        Memory(
            "Picked up multivitamins and fish oil from the pharmacy yesterday",
            group="hay",
        ),
        Memory(
            "Tom tweaked his knee during the half marathon last weekend",
            group="hay",
        ),
        Memory(
            "The urgent care clinic on Oak Street now accepts walk-ins",
            group="hay",
        ),
        Memory(
            "Need to schedule the annual dental cleaning before insurance resets",
            group="hay",
        ),
        Memory(
            "Mark has been dealing with lower back pain from sitting at his desk",
            group="hay",
        ),
        Memory(
            "Janet mentioned her cholesterol numbers improved after the new medication",
            group="hay",
        ),
        # Food-allergy-adjacent (not about peanuts or Thai food)
        Memory(
            "Lisa reserved a table at the new Italian place for Saturday evening",
            group="hay",
        ),
        Memory(
            "Karen mentioned her son is lactose intolerant and avoids dairy",
            group="hay",
        ),
        Memory(
            "The sushi restaurant on Main Street has a gluten-free menu now",
            group="hay",
        ),
        Memory(
            "Rachel asked if the bakery downtown can do a nut-free cake for her son",
            group="hay",
        ),
        Memory(
            "Tom said the Mexican restaurant on Broad Street does great takeout",
            group="hay",
        ),
        # Travel-adjacent (not about Denver or dog-sitting)
        Memory(
            "Tom found cheap flights to Portland for the long weekend",
            group="hay",
        ),
        Memory(
            "Need to renew the passport before summer travel season starts",
            group="hay",
        ),
        Memory(
            "The rental car agency sent a confirmation email for the coast trip",
            group="hay",
        ),
        Memory(
            "Lisa is comparing hotels in San Diego for the anniversary trip",
            group="hay",
        ),
        Memory(
            "Sam printed the boarding passes in case the app crashes at the gate",
            group="hay",
        ),
        # Pet-adjacent (not about Dave watching the dog)
        Memory(
            "The vet said the cat needs a dental cleaning this spring",
            group="hay",
        ),
        Memory(
            "Rachel offered to feed the fish while we visit her parents",
            group="hay",
        ),
        Memory(
            "Karen asked if we know a good dog walker for when she travels",
            group="hay",
        ),
        # Medical-equipment-adjacent (not about BP cuff)
        Memory(
            "Need to replace the batteries in the bathroom scale",
            group="hay",
        ),
        Memory(
            "The first aid kit under the sink is missing bandages",
            group="hay",
        ),
        Memory(
            "Tom bought a new digital thermometer for the medicine cabinet",
            group="hay",
        ),
        # Key-adjacent (not about Dave's spare key)
        Memory(
            "Need to get a copy of the garage door opener remote made",
            group="hay",
        ),
        Memory(
            "Tom asked if we changed the locks after the renovation",
            group="hay",
        ),
        Memory(
            "Janet lost her car keys and found them in the coat pocket",
            group="hay",
        ),
        # Restaurant-adjacent (not about Thai or peanut oil)
        Memory(
            "Mark said the new burger joint has a forty-five minute wait on Fridays",
            group="hay",
        ),
        Memory(
            "Chris booked the private dining room at the steakhouse for his birthday",
            group="hay",
        ),
        Memory(
            "Sam reviewed the Indian restaurant downtown and gave it four stars",
            group="hay",
        ),
        Memory(
            "Janet heard the seafood place on Harbor Road closes for winter",
            group="hay",
        ),
        Memory(
            "The coffee shop on Pine Street started serving oat milk lattes",
            group="hay",
        ),
        Memory(
            "Rachel ordered a standing desk converter for her home office setup",
            group="hay",
        ),
    ],
    queries=[
        Query(
            text="Should we try the new Thai place for team lunch?",
            description=(
                "Should find chain_a start (peanut allergy) and bridge "
                "(Thai place uses peanut oil). Iterative should discover "
                "EpiPen location via peanut-risk token chain."
            ),
            expected_indices=[0, 1],
            expected_iterative=[2],
            unexpected_indices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ),
        Query(
            text="What needs to happen before the Denver trip?",
            description=(
                "Should find chain_b start (Denver flights) and bridge "
                "(dog care while away). Iterative should discover spare "
                "key location via 2-hop token traversal through pet care."
            ),
            expected_indices=[3, 4],
            expected_iterative=[5],
            unexpected_indices=[0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13],
        ),
        Query(
            text="What should we bring for Sarah's doctor visit next Thursday?",
            description=(
                "Should find chain_c start (Dr. Patel follow-up) and bridge "
                "(bring BP log). Iterative should discover BP cuff location "
                "via medical-prep token chain."
            ),
            expected_indices=[6, 7],
            expected_iterative=[8],
            unexpected_indices=[0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13],
        ),
        Query(
            text="Is there any safety concern with the heating system?",
            description=(
                "Should find chain_d start (furnace crack) and bridge "
                "(CO detector alarm). Token should surface the latest "
                "revision -- CO risk recurring after repair. Tests "
                "whether revise action correctly updates token state."
            ),
            expected_indices=[9, 10],
            expected_iterative=[13],
            unexpected_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        ),
    ],
)
