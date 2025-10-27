import re

def normalise_text(text):
    """
    Cleans up common obfuscations: replaces leetspeak, extra punctuation, etc.
    """
    text = text.lower()
    text = re.sub(r'0', 'o', text)
    text = re.sub(r'1', 'i', text)
    text = re.sub(r'3', 'e', text)
    text = re.sub(r'@', 'a', text)
    text = re.sub(r'\$', 's', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

if __name__ == "__main__":
    print(normalise_text("H@t3 sp33ch!!!"))
