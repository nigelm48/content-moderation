from textblob import TextBlob

def detect_and_fallback(texts, fallback_fn):
    """
    Detects likely obfuscated or low-confidence texts and applies a fallback mitigation.
    """
    mitigated = []
    for text in texts:
        # Ensure it's a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        blob = TextBlob(text)
        corrected = blob.correct().string  # spell-correct (optional)
        mitigated.append(corrected)
    
    # Run the fallback model (e.g., Detoxify) on the cleaned text
    return fallback_fn(mitigated)
