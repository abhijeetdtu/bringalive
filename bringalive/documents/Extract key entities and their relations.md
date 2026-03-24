Extract key entities and their relationship tuples that completely summarize the text in sequence. Skip unimportant, filler sentences.
Use clear triples for mind maps: Subject → Predicate → Object.

Extract SPO as atomic knowledge triples. Use the subject as the actor, predicate as the main action verb, and object as the shortest direct noun phrase receiving that action. Exclude prepositional phrases, explanations, titles-with-framing text, infinitive clauses, and extra modifiers unless essential. When an object is complex, split it into multiple smaller triples. Prefer active voice.

Never create a new object entity by combining quantity, adjective, and noun into one node. If a phrase includes count, state, or description, split it into separate triples. Extract participants, counts, and attributes as different rows.

Extract only atomic entities and relations. Do not use descriptive wrappers such as “life,” “activity,” “history of his time,” “geography of his time,” “rise of,” “fall of,” or adjective+noun bundles as subject or object nodes. Normalize to the core entity and split attributes, time periods, and event changes into separate triples. Prefer “England” over “Victorian England,” “South Africa” over “Racialist South Africa,” and “Bolshevism” over “Rise of Bolshevism.”

Output should be in markdown table with columns (subject, predicate, object). No additional summary.