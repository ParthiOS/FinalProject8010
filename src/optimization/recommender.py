def simplify_prompt(prompt: str) -> str:
    # Simplify by removing filler or redundant phrases (mock logic)
    replacements = {
        "please": "",
        "kindly": "",
        "I would like to": "I'd like to",
        "could you": "can you",
        "would you mind": "can you"
    }

    simplified = prompt.lower()
    for k, v in replacements.items():
        simplified = simplified.replace(k, v)
    
    return simplified.strip().capitalize()