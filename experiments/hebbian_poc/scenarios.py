"""Two-Sarahs entity disambiguation scenario.

Tests whether Hebbian recall tokens can create relational groups
that disambiguate people with the same name in different social
contexts. Vector search returns both Sarahs for any Sarah query;
tokens should activate only the relevant relational group.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Memory:
    content: str
    group: str = ""  # "anchor", "mother", "kid", "hay"


@dataclass
class Query:
    text: str
    description: str
    expected_indices: list[int]  # MUST find these
    unexpected_indices: list[int]  # MUST NOT find these


@dataclass
class Scenario:
    name: str
    description: str
    memories: list[Memory]
    queries: list[Query]


SCENARIO = Scenario(
    name="two_sarahs",
    description="Entity disambiguation: mother Sarah vs kid Sarah",
    memories=[
        # ANCHORS (0-2): establish relationships
        Memory(
            "Alex's mother Sarah lives in Portland and visits every few months",
            group="anchor",
        ),
        Memory(
            "Brian and Lilly have a daughter named Sarah"
            " who is Sofi's best friend at school",
            group="anchor",
        ),
        Memory(
            "Alex and his wife have a daughter Sofi who is in 3rd grade",
            group="anchor",
        ),
        # MOTHER-SARAH NEEDLES (3-6): health, age, elderly context
        Memory(
            "Sarah called to ask for a ride to her"
            " cardiologist appointment on Thursday",
            group="mother",
        ),
        Memory(
            "Sarah's 70th birthday is coming up in June,"
            " we should plan a family gathering",
            group="mother",
        ),
        Memory(
            "Need to pick up Sarah's blood pressure prescription from Walgreens",
            group="mother",
        ),
        Memory(
            "Sarah mentioned she wants to start a garden"
            " but her knees have been bothering her",
            group="mother",
        ),
        # KID-SARAH NEEDLES (7-10): school, playdates, children's activities
        Memory(
            "Sarah got an A on her math test, Sofi was so proud of her best friend",
            group="kid",
        ),
        Memory(
            "Sofi wants Sarah to sleep over this Saturday for a movie marathon",
            group="kid",
        ),
        Memory(
            "Sarah and Sofi are building a volcano model for the school science fair",
            group="kid",
        ),
        Memory(
            "Lilly mentioned Sarah has been taking"
            " swimming lessons at the community pool",
            group="kid",
        ),
        # HAYSTACK (11-24): original mundane noise
        Memory(
            "Alex asked about the conference room booking for Thursday", group="hay"
        ),
        Memory(
            "Brian mentioned the neighborhood block party is next month", group="hay"
        ),
        Memory(
            "Sofi's school sent a reminder about parent-teacher conferences",
            group="hay",
        ),
        Memory(
            "Alex needs to get the car inspected before the end of the month",
            group="hay",
        ),
        Memory(
            "Lilly forwarded the PTA newsletter about the spring fundraiser",
            group="hay",
        ),
        Memory(
            "Brian asked if we want to do a joint barbecue this weekend", group="hay"
        ),
        Memory("Alex mentioned the wifi has been spotty all week", group="hay"),
        Memory("Sofi's soccer practice moved to Tuesday this week", group="hay"),
        Memory("Brian said their kitchen renovation starts in April", group="hay"),
        Memory("Alex needs to pick up dry cleaning before Saturday", group="hay"),
        Memory("Lilly asked about a good pediatrician recommendation", group="hay"),
        Memory("Sofi lost her favorite water bottle at school again", group="hay"),
        Memory("Brian mentioned he is training for a half marathon", group="hay"),
        Memory("Alex forwarded the updated HOA meeting agenda", group="hay"),
        # --- TIER 1: Inert filler (25-144) ---
        # Work/office
        Memory("Dave sent the revised project timeline to the whole team", group="hay"),
        Memory(
            "The quarterly review meeting got pushed to next Wednesday", group="hay"
        ),
        Memory(
            "Rachel needs the expense reports submitted by end of day Friday",
            group="hay",
        ),
        Memory("Tom asked if the new hire orientation is still on Monday", group="hay"),
        Memory("Karen scheduled a one-on-one to discuss the Q3 roadmap", group="hay"),
        Memory("The office printer on the third floor is jammed again", group="hay"),
        Memory(
            "Mark confirmed he will present the sales numbers at the all-hands",
            group="hay",
        ),
        Memory(
            "Lisa volunteered to take notes at the product strategy meeting",
            group="hay",
        ),
        Memory(
            "Need to update the shared drive with the latest client deck", group="hay"
        ),
        Memory(
            "Janet emailed about reserving the large conference room for Tuesday",
            group="hay",
        ),
        Memory(
            "Chris finished the code review for the authentication module", group="hay"
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
        # Finance
        Memory("The electric bill came in higher than usual this month", group="hay"),
        Memory(
            "Need to review the auto insurance renewal before it lapses", group="hay"
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
            "The property tax assessment arrived and looks about the same", group="hay"
        ),
        Memory(
            "Karen recommended switching to a high-yield savings account", group="hay"
        ),
        Memory(
            "Mark said the quarterly estimated taxes are due on the 15th", group="hay"
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
        Memory("Time to start organizing receipts for tax prep", group="hay"),
        Memory(
            "Sam found a better rate on car insurance through a different provider",
            group="hay",
        ),
        # Home maintenance
        Memory(
            "The kitchen faucet has been dripping all week and needs a new washer",
            group="hay",
        ),
        Memory(
            "Dave recommended a good electrician for the basement wiring project",
            group="hay",
        ),
        Memory(
            "Need to replace the furnace filter before the heating season", group="hay"
        ),
        Memory("The gutter on the south side of the house is sagging", group="hay"),
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
        Memory("Mark is going to pressure wash the driveway this weekend", group="hay"),
        Memory(
            "The dishwasher is not draining properly after the last cycle", group="hay"
        ),
        Memory(
            "Need to schedule the annual HVAC inspection before summer", group="hay"
        ),
        Memory(
            "Lisa got a quote for refinishing the hardwood floors upstairs", group="hay"
        ),
        Memory(
            "The smoke detector batteries need replacing in the hallway", group="hay"
        ),
        Memory(
            "Chris offered to help patch the drywall in the spare bedroom", group="hay"
        ),
        Memory(
            "Called a plumber about the slow drain in the downstairs shower",
            group="hay",
        ),
        # Technology
        Memory(
            "The laptop has been running slow since the last operating system update",
            group="hay",
        ),
        Memory(
            "Dave set up the new mesh wifi router and coverage is much better now",
            group="hay",
        ),
        Memory("Need to update the password for the home security system", group="hay"),
        Memory(
            "Tom recommended a password manager to keep everything organized",
            group="hay",
        ),
        Memory(
            "The smart thermostat lost its wifi connection again overnight", group="hay"
        ),
        Memory(
            "Karen asked which external hard drive is best for photo backups",
            group="hay",
        ),
        Memory("The phone screen protector cracked and needs replacing", group="hay"),
        Memory(
            "Sam mentioned a free VPN that works well for streaming abroad", group="hay"
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
        # Travel
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
        Memory(
            "Sam wants to do a day trip to the vineyards this Saturday", group="hay"
        ),
        Memory(
            "Gas prices along the interstate are about twenty cents cheaper",
            group="hay",
        ),
        # Weather/seasonal
        Memory(
            "The forecast says rain all next week so outdoor plans are off", group="hay"
        ),
        Memory(
            "Need to winterize the outdoor faucets before the first freeze", group="hay"
        ),
        Memory(
            "Tom brought in the patio furniture before the storm last night",
            group="hay",
        ),
        Memory(
            "The spring pollen count is already high and allergies are acting up",
            group="hay",
        ),
        Memory("Dave mentioned the ski resort closes at the end of March", group="hay"),
        Memory(
            "Should get the snow tires swapped out for all-seasons this week",
            group="hay",
        ),
        Memory(
            "The weather cleared up enough to finally mow the back lawn", group="hay"
        ),
        Memory("Karen stocked up on ice melt and salt for the driveway", group="hay"),
        Memory(
            "Mark said the lake is warm enough for swimming by mid-June", group="hay"
        ),
        Memory(
            "Need to clean out the rain gutters after the leaf drop this fall",
            group="hay",
        ),
        # Food/cooking
        Memory(
            "Tried a new slow cooker chili recipe and it turned out great", group="hay"
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
            "Dave is experimenting with smoking brisket on his new pellet grill",
            group="hay",
        ),
        Memory(
            "Meal prepped five lunches for the week including chicken stir fry",
            group="hay",
        ),
        Memory(
            "Rachel found a bakery downtown that does custom layer cakes", group="hay"
        ),
        Memory("The grocery delivery got the wrong brand of yogurt again", group="hay"),
        Memory(
            "Karen made a batch of granola bars for snacks during the road trip",
            group="hay",
        ),
        Memory(
            "Lisa reserved a table at the new Italian place for Saturday evening",
            group="hay",
        ),
        Memory(
            "Need to defrost the ground turkey overnight for tomorrow's tacos",
            group="hay",
        ),
        Memory(
            "Sam started a sourdough starter and it finally looks active", group="hay"
        ),
        # Hobbies
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
        Memory(
            "Mark signed up for the 5K charity run happening downtown next month",
            group="hay",
        ),
        Memory(
            "Lisa is learning watercolor painting through an online course", group="hay"
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
            "Chris joined a weekend cycling group that rides the lakefront path",
            group="hay",
        ),
        # Pets/garden
        Memory(
            "The vet said the cat needs a dental cleaning sometime this spring",
            group="hay",
        ),
        Memory("Dave offered to dog-sit while we are away next weekend", group="hay"),
        Memory(
            "The tomato plants are finally sprouting in the raised bed", group="hay"
        ),
        Memory(
            "Tom bought a bag of wildflower seeds for the empty patch by the fence",
            group="hay",
        ),
        Memory(
            "Need to refill the bird feeder in the backyard before it snows again",
            group="hay",
        ),
        Memory(
            "Karen said her golden retriever graduates obedience class on Thursday",
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
            "The dog ate something weird at the park and was sick all evening",
            group="hay",
        ),
        Memory(
            "Sam built a raised garden bed out of cedar planks last Saturday",
            group="hay",
        ),
        Memory(
            "Need to treat the roses for aphids before they spread further", group="hay"
        ),
        Memory(
            "Chris adopted a rescue cat from the shelter on Maple Street", group="hay"
        ),
        # --- TIER 2: Moderate interference (145-194) ---
        Memory("Alex emailed the team about the Q2 milestone deadline", group="hay"),
        Memory(
            "Brian dropped off a casserole dish he borrowed last month", group="hay"
        ),
        Memory(
            "Lilly shared a recipe for gluten-free brownies she found online",
            group="hay",
        ),
        Memory(
            "Sofi drew a picture of a rainbow for her art class project", group="hay"
        ),
        Memory(
            "Alex and Brian watched the playoff game together on Sunday", group="hay"
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
            "Alex grabbed coffee with Brian after dropping off the kids", group="hay"
        ),
        Memory(
            "Sofi finished her chapter book and wants to start the next one",
            group="hay",
        ),
        Memory("Lilly asked if we have jumper cables she could borrow", group="hay"),
        Memory(
            "Brian texted that he found Alex's sunglasses in his truck", group="hay"
        ),
        Memory("Sofi told Lilly she wants to learn how to bake cookies", group="hay"),
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
            "Alex ran into Brian at the hardware store on Saturday morning", group="hay"
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
        Memory("Brian lent Alex his power drill for the weekend", group="hay"),
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
            "Sofi and Lilly made friendship bracelets at the craft fair", group="hay"
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
        Memory("Lilly dropped off a bag of hand-me-down clothes for Sofi", group="hay"),
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
            "Sofi earned a star sticker for helping clean up the classroom", group="hay"
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
            "Sofi told Alex she wants a pet hamster for her next birthday", group="hay"
        ),
        # --- TIER 3: Hard negatives (195-210) ---
        # Health hard negatives
        Memory(
            "Alex scheduled his annual physical for the second week of April",
            group="hay",
        ),
        Memory(
            "Need to renew the health insurance plan before open enrollment ends",
            group="hay",
        ),
        Memory(
            "Brian's doctor recommended he cut back on caffeine and drink more water",
            group="hay",
        ),
        Memory(
            "Picked up multivitamins and fish oil capsules from the pharmacy",
            group="hay",
        ),
        Memory(
            "The dentist office called to confirm the"
            " cleaning appointment for Thursday",
            group="hay",
        ),
        Memory(
            "Alex has been dealing with lower back pain"
            " from sitting at his desk all day",
            group="hay",
        ),
        Memory(
            "Lilly mentioned her mother's cholesterol"
            " numbers improved after the new medication",
            group="hay",
        ),
        Memory(
            "The urgent care clinic on Oak Street now accepts walk-ins on weekends",
            group="hay",
        ),
        Memory(
            "Brian tweaked his knee during the half marathon and is icing it nightly",
            group="hay",
        ),
        Memory(
            "Need to call the pharmacy about the"
            " prescription refill that was not ready",
            group="hay",
        ),
        # Kids/friends hard negatives
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
        Memory(
            "Brian mentioned the youth soccer league registration opens next Monday",
            group="hay",
        ),
        Memory(
            "Sofi wants to invite six friends to go roller skating for a group outing",
            group="hay",
        ),
        Memory(
            "The school guidance counselor is hosting a"
            " friendship skills workshop next week",
            group="hay",
        ),
        Memory(
            "Sofi brought home a permission slip for the overnight nature camp trip",
            group="hay",
        ),
        # Celebration/party hard negatives
        Memory(
            "Need to buy a gift for Brian and Lilly's anniversary dinner next month",
            group="hay",
        ),
        Memory(
            "Alex's work team is throwing a farewell"
            " party for Dave on Friday afternoon",
            group="hay",
        ),
        Memory(
            "Sofi's birthday party invitations need to go out by the end of this week",
            group="hay",
        ),
        Memory(
            "Brian is organizing a surprise party for"
            " Lilly's 40th at the Italian restaurant",
            group="hay",
        ),
        Memory(
            "The family reunion planning committee wants a headcount by next Tuesday",
            group="hay",
        ),
        Memory(
            "Lilly suggested a potluck theme for the end-of-year classroom celebration",
            group="hay",
        ),
    ],
    queries=[
        Query(
            text="What is going on with Sarah's health situation?",
            description="Should find mother-Sarah health memories, not kid-Sarah",
            expected_indices=[3, 5, 6],  # cardiologist, prescription, knees
            unexpected_indices=[7, 8, 9, 10],  # kid activities
        ),
        Query(
            text="What are Sofi and her friends doing this weekend?",
            description="Should find kid-Sarah activities, not mother-Sarah health",
            expected_indices=[8, 9],  # sleepover, volcano project
            unexpected_indices=[3, 4, 5, 6],  # mother health/birthday
        ),
        Query(
            text="We need to plan something for Sarah's birthday coming up",
            description="Should find mother-Sarah birthday, not kid-Sarah school",
            expected_indices=[4, 0],  # 70th birthday, mother in Portland
            unexpected_indices=[7, 8, 9, 10],  # kid activities
        ),
    ],
)
