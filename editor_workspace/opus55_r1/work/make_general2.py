# Hypothesis: a larger, more diverse negative set (other-domain 'Classify ... intent' prompts in the exact task format, other
# classification formats, diverse instructions, banking questions as plain chat) is needed to make the task gates specific.
import json, random
random.seed(1)
DOM = {
 "airline": ["My flight was cancelled, can I get a refund?", "How many bags can I check on an international flight?", "I need to change my seat to an aisle.",
             "My luggage didn't arrive at the destination.", "Can I bring my dog on board?", "What time does boarding start?", "I missed my connecting flight.",
             "How do I add my frequent flyer number?", "Is there wifi on the plane?", "I want to upgrade to business class."],
 "telecom": ["My internet keeps disconnecting.", "I want to upgrade my phone plan.", "Why is my bill higher this month?", "How do I set up voicemail?",
             "There is no signal in my area.", "Can I keep my number if I switch carriers?", "I want to cancel my contract.", "My data runs out too fast.",
             "How do I activate roaming?", "The router lights are blinking red."],
 "e-commerce": ["Where is my package? It was supposed to arrive yesterday.", "I received the wrong size.", "How do I return an item?", "Can I change my delivery address?",
                "The product arrived damaged.", "Do you ship to Canada?", "I was charged twice for my order.", "How long does a refund take?",
                "Can I cancel my order?", "The discount code isn't working."],
 "hotel": ["Can I get a late checkout tomorrow?", "Is breakfast included?", "The air conditioning in my room is broken.", "Do you have parking?",
           "I'd like to book a room for two nights.", "Can I get extra towels?", "Is the pool open today?", "I left my charger in the room.",
           "Can you call a taxi for me?", "I want to cancel my reservation."],
 "insurance": ["I want to file a claim for my car accident.", "Does my policy cover flood damage?", "How do I add my spouse to my plan?", "When is my premium due?",
               "My claim was denied, why?", "Can I lower my deductible?", "I need proof of insurance.", "How do I cancel my policy?",
               "Is dental included?", "My address changed."],
 "restaurant": ["Do you have vegetarian options?", "Can I book a table for six at 8pm?", "My order is missing a drink.", "Are you open on Sundays?",
                "Do you deliver to my area?", "The food was cold.", "Do you have a gluten-free menu?", "Can I pay with a gift card?",
                "Is there a kids menu?", "I want to cancel my reservation."],
 "healthcare": ["I need to reschedule my appointment with Dr. Lee.", "Can I get a refill on my prescription?", "What are your opening hours?", "Do you accept my insurance?",
                "I need my test results.", "Can I see a doctor today?", "How do I get a referral?", "I have a fever and a cough.",
                "Where do I park for the clinic?", "Can I update my emergency contact?"],
 "smart home": ["Turn off the living room lights.", "Set the thermostat to 70 degrees.", "Lock the front door.", "Play some jazz in the kitchen.",
                "What's the weather tomorrow?", "Set an alarm for 6am.", "Is the garage door open?", "Dim the bedroom lights.",
                "Add milk to my shopping list.", "Start the vacuum."],
 "ride-sharing": ["My driver took a longer route and charged me more.", "I left my phone in the car.", "How do I add a stop?", "The driver never showed up.",
                  "Can I schedule a ride for tomorrow?", "Why was I charged a cancellation fee?", "How do I rate my driver?", "I want to become a driver.",
                  "The car was dirty.", "Can I split the fare with a friend?"],
 "gym": ["I want to freeze my membership for a month.", "What time do you open?", "Can I bring a guest?", "How do I book a spin class?",
         "I want to cancel my membership.", "Do you have personal trainers?", "The locker is broken.", "Is there a student discount?",
         "Can I change my billing date?", "Are towels provided?"],
 "real estate": ["Is the apartment on Main Street still available?", "Can I schedule a viewing?", "Are pets allowed?", "What is the monthly rent?",
                 "Is parking included?", "When can I move in?", "Who do I contact for repairs?", "Can I break my lease early?",
                 "Is the deposit refundable?", "Are utilities included?"],
 "streaming": ["My video keeps buffering.", "How do I cancel my subscription?", "Can I share my account?", "Why was I charged this month?",
               "How do I download shows for offline?", "I forgot my password.", "Is this movie available in 4K?", "How do I add a profile?",
               "Subtitles are out of sync.", "Can I change my plan?"],
 "IT helpdesk": ["My laptop won't turn on.", "I can't connect to the VPN.", "I need access to the shared drive.", "My email isn't syncing.",
                 "The printer is jammed.", "I forgot my password.", "Can I get a second monitor?", "My computer is very slow.",
                 "How do I install the new software?", "Outlook keeps crashing."],
}
OTHER_FMT = [
 "Classify the sentiment of this review as positive or negative.\n\nReview: {m}\n\nSentiment:",
 "Classify this message as spam or not spam.\n\nMessage: {m}\n\nAnswer:",
 "Classify the topic of this text.\n\nText: {m}\n\nTopic:",
 "What is the intent of this message?\n\nMessage: {m}",
 "Label the emotion expressed in this text.\n\nText: {m}\n\nEmotion:",
 "Classify this customer message by urgency (low, medium, high).\n\nMessage: {m}\n\nUrgency:",
]
OTHER_MSG = ["The food was amazing and the staff were friendly.", "I waited an hour and nobody helped me.", "Win a free iPhone now, click this link!",
             "The new tax law will affect small businesses.", "I am so happy my sister is visiting!", "My account was hacked, help immediately!",
             "The movie was boring and too long.", "Scientists discover a new species of frog.", "Can we move our meeting to Thursday?",
             "This is the worst customer service I've ever had.", "Limited time offer: 90% off all watches.", "The team won the match in overtime."]
UNREL = [
 "Write a Python function to compute the factorial of a number.", "Explain the difference between TCP and UDP.", "What is the capital of Morocco?",
 "Summarize the causes of World War I in three sentences.", "Write a short poem about a rainy day in the city.", "How does compound interest work?",
 "Give me a packing list for a beach vacation.", "What is the difference between a list and a tuple in Python?", "Explain how the immune system fights infection.",
 "Write a two-sentence horror story.", "What are the main features of Gothic architecture?", "How do I calculate the area of a circle?",
 "Translate 'Where is the train station?' into German.", "What is a sonnet?", "Write a limerick about a forgetful wizard.",
 "Explain the concept of entropy simply.", "What should I consider when buying a used car?", "How do I make a sourdough starter?",
 "What are the benefits of meditation?", "Describe the plot of Romeo and Juliet briefly.", "Write a SQL query that finds the top 5 customers by total sales.",
 "What is the difference between a debit card and a credit card?", "How do banks make money?", "What is a mortgage?", "Explain what an interest rate is.",
 "How do I open a savings account?", "What does APR mean?", "Should I pay off debt or invest first?", "What is a credit score and how is it calculated?",
 "Explain what cryptocurrency is.", "How do I create a monthly budget?", "What is the stock market?", "Write a short story about a lost key.",
 "List three uses of baking soda.", "How do solar panels work?", "What is the tallest building in the world?", "Explain the rules of basketball.",
 "Give me a recipe for pancakes.", "Who was Cleopatra?", "What causes rainbows?", "Write an email asking my manager for a day off.",
 "How do I fix a leaky faucet?", "What is the difference between a hurricane and a tornado?", "Explain object-oriented programming.",
 "What are the planets in order from the sun?", "Write a motivational message for a friend running a marathon.", "How do I improve my handwriting?",
 "What is the purpose of the United Nations Security Council?", "Explain the difference between 'its' and 'it's'.", "Suggest five names for a coffee shop.",
 "What is the boiling point of water in Fahrenheit?", "Explain how a refrigerator works.", "How many ounces are in a pound?",
 "What is the Fibonacci sequence?", "Write a JavaScript snippet that sorts an array of numbers.", "What is photosynthesis?",
 "Tell me about the history of the internet.", "What is a good beginner houseplant?", "Write a thank-you message for a birthday gift.",
 "How do I convert Celsius to Fahrenheit?", "Explain what a hash table is.", "What are some tips for traveling on a budget?",
 "What year did the Roman Empire fall?", "Who invented the light bulb?", "What does DNA stand for?", "Write a haiku about winter.",
 "How do I start running if I'm a beginner?", "What is the difference between empathy and sympathy?", "Explain how elections work in a parliamentary system.",
 "Give me a fun fact about space.", "How can I lower my electricity bill?", "What is a balanced diet?", "Explain the greenhouse effect to a child.",
 "Classify these animals into mammals and birds: eagle, dolphin, sparrow, bat.", "Sort these numbers from smallest to largest: 42, 7, 19, 3.",
 "Is a tomato a fruit or a vegetable?", "Categorize the following words as nouns or verbs: run, table, swim, chair.",
 "Which of these is a prime number: 21, 23, 25?", "Identify the language: 'Ich bin müde.'", "What is the intent behind asking 'What time is it?'",
 "My card was declined at the store, what could be the reason?", "How long does a bank transfer usually take?", "I lost my wallet, what should I do?",
 "What is the exchange rate between dollars and euros roughly?", "How do ATMs work?", "Can you explain what a direct debit is?",
 "What's the best way to send money abroad?", "How do I dispute a charge on my credit card?", "What is a PIN and why is it important?",
 "Explain contactless payments.", "What is Apple Pay?", "How do top-up cards work?", "What is a virtual card?",
]
pr = [json.loads(l) for l in open('data/practice.jsonl')]
old = {json.loads(l)['text'] for l in open('work/general.jsonl')}
byc = {}
for x in pr: byc.setdefault(x['label'], []).append(x['text'])
rows = []
for dom, msgs in DOM.items():
    for m_ in msgs:
        rows.append({"text": f"Classify this {dom} customer message into the correct intent.\n\nMessage: {m_}\n\nIntent:", "kind": "nearmiss"})
for f in OTHER_FMT:
    for m_ in random.sample(OTHER_MSG, 6): rows.append({"text": f.format(m=m_), "kind": "nearmiss"})
for t in UNREL: rows.append({"text": t, "kind": "unrelated"})
for lab in sorted(byc):
    cands = [t for t in byc[lab] if t not in old]
    rows.append({"text": random.choice(cands), "kind": "banking_chat"})
for k, r in enumerate(rows): r["split"] = "fit" if k % 3 != 2 else "check"
with open('work/general2.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + "\n")
import collections; print(len(rows), collections.Counter((r['kind'], r['split']) for r in rows))
