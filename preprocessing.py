# This is just a copy from clean_text function from my notebooks. Why?? because it makes me easier to stablize clean
# text process in my streamlit app
import re
import html

def clean_text(text):
    # 1. Decode standard HTML entities
    text = html.unescape(text)

    # 2. Fix remaining HTML entity fragments
    text = text.replace("#39;", "'")
    text = re.sub(r"(?<=\s)amp;", "&", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\s)quot;", '"', text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\s)(?:lt|gt);", " ", text, flags=re.IGNORECASE)

    # 3. Remove pseudo-HTML tags: <b>, <p>, <a href=...>, </strong>
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Remove URLs and email addresses
    text = re.sub(r"https?://\S*|www\.\S*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " ", text)

    # 5. Remove source attribution inside parentheses: (AP-CP), (Reuters)
    text = re.sub(r"\s*\((?:Reuters|AP|AFP)(?:-[A-Za-z]+)?\)", "", text, flags=re.IGNORECASE)

    # 6. Remove source attribution after title: AP - / Reuters -
    text = re.sub(r"\s+\b(?:Reuters|AP|AFP)\s*-\s*", " ", text, flags=re.IGNORECASE)

    # 7. Remove incomplete parenthesized attribution: (AFP -
    text = re.sub(r"\s*\((?:Reuters|AP|AFP)\s*-\s*", " ", text, flags=re.IGNORECASE)

    # 8. Remove news codes at the beginning: AP-Italy-Alitalia
    text = re.sub(r"^(?:AP|AFP|Reuters)-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\s+", "", text, flags=re.IGNORECASE)

    # 9. Remove source attribution after a slash: UC Davis/AP --
    text = re.sub(r"/(?:Reuters|AP|AFP)\s*--?\s*", " ", text, flags=re.IGNORECASE)

    # 10. Remove escaped dollar signs and remaining backslashes
    text = text.replace("\\$", "$")
    text = text.replace("\\", " ")

    # 11. Lowercase
    text = text.lower()

    # 12. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text