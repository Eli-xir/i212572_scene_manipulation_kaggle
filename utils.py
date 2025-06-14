import spacy

nlp = spacy.load("en_core_web_sm")

def parse_instruction(instruction: str) -> dict:
    doc = nlp(instruction.lower())

    action_details = {
        "object": None,
        "relocation": None,
        "relighting": None
    }

    # Step 1: Find object of motion verb
    for token in doc:
        if token.dep_ in ("dobj", "pobj") and token.head.lemma_ in ["move", "place", "put", "relocate"]:
            action_details["object"] = token.text
            break  # Stop after finding the first good object

    # Step 2: Fallback to first noun chunk
    if not action_details["object"]:
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ != "nsubj":
                action_details["object"] = chunk.root.text
                break

    # Step 3: Check for relocation cues
    for token in doc:
        if token.text in ["left", "right", "top", "bottom", "center"]:
            action_details["relocation"] = token.text
            break

    # Step 4: Check for relighting phrases
    for token in doc:
        if "light" in token.text or "shadow" in token.text or "sun" in token.text:
            subtree = list(token.head.subtree)
            lighting_phrase = [t.text for t in subtree if t.pos_ in ["ADJ", "NOUN"]]
            action_details["relighting"] = " ".join(lighting_phrase)
            break

    return action_details

