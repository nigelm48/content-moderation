import re
import unicodedata

leet_dict = {'0':'o','1':'i','3':'e','4':'a','5':'s','7':'t','@':'a','!':'i','+':'t','$':'s','|':'i'}
translation = str.maketrans(leet_dict)

def normalise_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.translate(translation)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == "__main__":
    print(normalise_text("H@t3 sp33ch!!! sooo b4d!!!"))
