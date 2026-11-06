"""Clean, title, tag and chunk the parsed Markdown documents."""
import html
import re
import unicodedata

from .config import CHUNK_WORDS, ORIGINAL

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean(text: str, from_pdf: bool) -> str:
    # PDF extraction leaves "Eﬀ ective", "ﬁ ndings": drop the stray space after a ligature.
    for lig, rep in LIGATURES.items():
        text = re.sub(lig + r" ?", rep, text)
    text = unicodedata.normalize("NFKC", text)
    if from_pdf:
        text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)   # re-join hyphenated line breaks
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # unwrap hard-wrapped lines
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def title_of(doc_id: str, ext: str, text: str) -> str:
    if ext == "html":
        raw = (ORIGINAL / f"{doc_id}.html").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
        if m:
            t = html.unescape(m.group(1)).strip()
            t = re.split(r"\s+[|–-]\s+(?:MSF|Médecins|Doctors|أطباء)", t)[0].strip()
            if t:
                return t[:150]
    for line in text.splitlines():
        line = line.strip("#-* \t")
        if len(line) > 3:
            return line[:150]
    return doc_id


# name -> regex alternatives. Word boundaries keep "Niger" from matching "Nigeria".
COUNTRIES = {
    "DR Congo": r"Democratic Republic of (?:the )?Congo|\bDRC\b|DR Congo",
    "Bangladesh": r"Bangladesh|Rohingya|Cox's Bazar",
    "Sudan": r"(?<!South )\bSudan\b|Darfur",
    "South Sudan": r"South Sudan",
    "Uganda": r"Uganda", "Kenya": r"Kenya", "Ethiopia": r"Ethiopia", "Somalia": r"Somalia",
    "Nigeria": r"Nigeria", "Niger": r"\bNiger\b", "Chad": r"\bChad\b", "Mali": r"\bMali\b",
    "Burkina Faso": r"Burkina Faso", "Cameroon": r"Cameroon",
    "Central African Republic": r"Central African Republic|\bCAR\b",
    "Guinea": r"(?<!New )(?<!Equatorial )\bGuinea\b(?!-Bissau)", "Sierra Leone": r"Sierra Leone",
    "Liberia": r"Liberia", "Mozambique": r"Mozambique", "Malawi": r"Malawi",
    "Zimbabwe": r"Zimbabwe", "South Africa": r"South Africa", "Eswatini": r"Eswatini|Swaziland",
    "Yemen": r"Yemen", "Syria": r"Syria", "Iraq": r"\bIraq\b", "Lebanon": r"Lebanon",
    "Palestine": r"Palestin|Gaza|West Bank", "Libya": r"Libya", "Ukraine": r"Ukrain",
    "Afghanistan": r"Afghanistan", "Pakistan": r"Pakistan", "India": r"\bIndia\b",
    "Myanmar": r"Myanmar", "Cambodia": r"Cambodia", "Haiti": r"Haiti",
    "Honduras": r"Honduras", "Colombia": r"Colombia", "Mexico": r"Mexico",
    "Uzbekistan": r"Uzbekistan", "Russia (Chechnya)": r"Chechnya", "Bosnia": r"Bosnia|Srebrenica",
    "Rwanda": r"Rwanda",
}
DISEASES = {
    "Tuberculosis": r"tuberculosis|\bTB\b|\bMDR-TB\b", "HIV": r"\bHIV\b|\bAIDS\b",
    "Malaria": r"malaria", "Cholera": r"cholera", "Measles": r"measles", "Ebola": r"ebola",
    "Malnutrition": r"malnutrition|malnourish", "Mental health": r"mental health|psycholog",
    "Sexual violence": r"sexual violence|\brape\b", "Maternal health": r"maternal|obstetric|pregnan",
    "Neonatal": r"neonat|newborn", "Vaccination": r"vaccin|immunis|immuniz",
    "Hepatitis": r"hepatitis", "Noma": r"\bnoma\b", "Meningitis": r"meningitis",
    "COVID-19": r"covid|sars-cov-2", "Trauma & surgery": r"war-wounded|surg|trauma care",
    "Kala-azar": r"kala-azar|leishmaniasis", "Dengue": r"dengue", "Diphtheria": r"diphtheria",
}
COUNTRIES = {k: re.compile(v) for k, v in COUNTRIES.items()}
DISEASES = {k: re.compile(v, re.I) for k, v in DISEASES.items()}


def tags(text: str, patterns: dict, min_hits: int = 2) -> list[str]:
    hits = {k: len(p.findall(text)) for k, p in patterns.items()}
    return [k for k, n in sorted(hits.items(), key=lambda kv: -kv[1]) if n >= min_hits]


def chunks(text: str) -> list[tuple[str, str]]:
    """Split into (heading, text) chunks of about CHUNK_WORDS words, breaking at headings."""
    out, heading, buf = [], "", []

    def flush():
        if buf:
            out.append((heading, " ".join(buf)))
            buf.clear()

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#") and len(para) < 200:
            flush()
            heading = para.lstrip("# ").strip()
            continue
        words = para.split()
        if buf and len(buf) + len(words) > CHUNK_WORDS:
            flush()
        while len(words) > CHUNK_WORDS:
            out.append((heading, " ".join(words[:CHUNK_WORDS])))
            words = words[CHUNK_WORDS:]
        buf.extend(words)
    flush()
    return [(h, t) for h, t in out if len(t.split()) >= 15]
