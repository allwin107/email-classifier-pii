import re
import spacy

# Load the small English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

def mask_pii(text):
    """
    Detects and masks PII (personally identifiable information) in the input text.
    Returns the masked text and a list of detected entities with their positions.
    """
    masked_text = text  # Holds the progressively masked version of the input text
    entities = []       # Stores metadata about each masked entity

    # Define regex patterns for various PII types
    patterns = {
        "email": r'[\w\.-]+@[\w\.-]+\.\w+',
        "phone_number": r'\+?\d[\d\s\-\(\)]{8,}\d',
        "aadhar_num": r'\b\d{4}\s\d{4}\s\d{4}\b',
        "credit_debit_no": r'\b(?:\d[ -]*?){13,16}\b',
        "cvv_no": r'\b\d{3}\b',
        "expiry_no": r'(0[1-9]|1[0-2])\/?([0-9]{2}|[0-9]{4})',
        "dob": r'\b(?:\d{1,2}[-/\s])?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s]?\d{2,4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
    }

    # Mask regex-based PII
    for entity_type, pattern in patterns.items():
        matches = list(re.finditer(pattern, masked_text))
        for match in reversed(matches):  # Loop in reverse to preserve correct indexing
            start, end = match.start(), match.end()
            original_value = match.group()
            placeholder = f"[{entity_type}]"
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
            entities.append({
                "position": [start, start + len(placeholder)],
                "classification": entity_type,
                "entity": original_value
            })

    # Mask full names using spaCy's NER
    doc = nlp(masked_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            original_value = ent.text
            placeholder = "[full_name]"
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
            entities.append({
                "position": [start, start + len(placeholder)],
                "classification": "full_name",
                "entity": original_value
            })

    return masked_text, entities


if __name__ == "__main__":
    email = "Hi, I'm John Doe. My email is john.doe@example.com and card is 1234 5678 9012 3456."
    masked, ents = mask_pii(email)
    print("masked email:\n", masked)
    print("\nEntities:\n", ents)
