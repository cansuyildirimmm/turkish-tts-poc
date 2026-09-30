"""Deterministic Turkish text normalization for TTS input.

Turns written ERP text into the words a speaker would say:

    "KDV %20'dir. Toplam borç ₺12.550,75'tir."
 -> "ka de ve yüzde yirmidir. Toplam borç on iki bin beş yüz elli Türk lirası yetmiş beş kuruştur."

The original document text is never modified; `normalize()` returns a new
string that is only sent to the TTS model. Model-agnostic: model-specific
character handling lives in tts_engine.

Conventions (Turkish orthography): '.' groups thousands, ',' is the decimal
mark, suffixes after numbers/abbreviations follow an apostrophe (5'e, KDV'yi).
"""

import re
import unicodedata

# --------------------------------------------------------------------------
# Lexicons: edit these to change how terms are read.
# --------------------------------------------------------------------------

# Case-sensitive, matched as whole words. Values are the spoken form.
ABBREVIATIONS = {
    "KDV": "ka de ve",
    "ÖTV": "ö te ve",
    "SGK": "se ge ka",
    "TC": "te ce",
    "T.C.": "te ce",
    "ERP": "e re pe",
    "CRM": "si ar em",
    "KPI": "ke pe i",
    "PDF": "pe de ef",
    "SMS": "se me se",
    "IBAN": "iban",
    "e-Fatura": "e fatura",
    "e-Arşiv": "e arşiv",
    "e-İrsaliye": "e irsaliye",
    "e-Defter": "e defter",
    "e-posta": "e posta",
    "vb.": "ve benzeri",
    "vs.": "vesaire",
}

# English terms common in ERP UIs, respelled with Turkish phonetics.
# Case-insensitive, whole words.
FOREIGN_WORDS = {
    "dashboard": "deşbord",
    "lead": "lid",
    "export": "eksport",
    "excel": "eksel",
    "login": "login",
    "online": "onlayn",
}

CURRENCY_WORD = "Türk lirası"  # possessive compound: TL'ye -> Türk lirasına
SUBUNIT_WORD = "kuruş"

MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
          "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

# Turkish letter names, used for IBAN country codes.
LETTER_NAMES = {
    "A": "a", "B": "be", "C": "ce", "Ç": "çe", "D": "de", "E": "e", "F": "fe",
    "G": "ge", "Ğ": "yumuşak ge", "H": "he", "I": "ı", "İ": "i", "J": "je",
    "K": "ke", "L": "le", "M": "me", "N": "ne", "O": "o", "Ö": "ö", "P": "pe",
    "R": "re", "S": "se", "Ş": "şe", "T": "te", "U": "u", "Ü": "ü", "V": "ve",
    "Y": "ye", "Z": "ze", "Q": "kü", "W": "dabılyu", "X": "iks",
}

# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(10 ** 12, "trilyon"), (10 ** 9, "milyar"), (10 ** 6, "milyon"), (10 ** 3, "bin")]
_DIGITS = ["sıfır"] + _ONES[1:]


def _below_thousand(n: int) -> list[str]:
    words = []
    if n >= 100:
        if n // 100 > 1:
            words.append(_ONES[n // 100])
        words.append("yüz")
        n %= 100
    if n >= 10:
        words.append(_TENS[n // 10])
        n %= 10
    if n:
        words.append(_ONES[n])
    return words


def number_to_words(n: int) -> str:
    """Spell a non-negative integer in Turkish: 1250000 -> 'bir milyon iki yüz elli bin'."""
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n == 0:
        return "sıfır"
    words = []
    for scale, name in _SCALES:
        if n >= scale:
            head, n = divmod(n, scale)
            if not (scale == 1000 and head == 1):  # 'bin', not 'bir bin'
                words.extend(_below_thousand(head))
            words.append(name)
    words.extend(_below_thousand(n))
    return " ".join(words)


def digits_to_words(digits: str) -> str:
    """Read digit by digit: '0006' -> 'sıfır sıfır sıfır altı'."""
    return " ".join(_DIGITS[int(d)] for d in digits)


def _integer_words(digits: str) -> str:
    # Leading zeros ("0006") are read digit by digit, everything else as a number.
    if len(digits) > 1 and digits.startswith("0"):
        return digits_to_words(digits)
    return number_to_words(int(digits))


def _fraction_words(digits: str) -> str:
    # "05" -> "sıfır beş", "75" -> "yetmiş beş"
    zeros = len(digits) - len(digits.lstrip("0"))
    rest = digits[zeros:]
    parts = ["sıfır"] * zeros + ([number_to_words(int(rest))] if rest else [])
    return " ".join(parts)


# --------------------------------------------------------------------------
# Suffix harmony
# --------------------------------------------------------------------------

_VOWELS = "aeıioöuü"
_BACK = "aıou"
_ROUNDED = "ouöü"
_VOICELESS = "fstkçşhp"
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def _tr_lower(s: str) -> str:
    return s.translate(_TR_LOWER).lower()


def _last_vowel(word: str) -> str:
    for ch in reversed(_tr_lower(word)):
        if ch in _VOWELS:
            return ch
    return "e"


def _harmonize_vowel(v: str, prev: str) -> str:
    back = prev in _BACK
    if v in "ae":  # two-way
        return "a" if back else "e"
    if v in "ıiuü":  # four-way
        rounded = prev in _ROUNDED
        return {(True, False): "ı", (False, False): "i",
                (True, True): "u", (False, True): "ü"}[(back, rounded)]
    return v


def attach_suffix(base: str, suffix: str, possessive: bool = False) -> str:
    """Attach an apostrophe suffix to a spoken form, re-applying Turkish harmony.

    The suffix was written for the original token (e.g. 'TL'ye', '75'tir'); when
    the spoken last word changes ('Türk lirası', 'kuruş') vowels, d/t and the
    buffer consonant (y/n) are adjusted. `possessive` marks compounds ending
    in the 3rd-person possessive (-sı/-si) which take an 'n' buffer.
    """
    if not suffix:
        return base
    s = _tr_lower(suffix)
    last = _tr_lower(base)[-1]
    ends_vowel = last in _VOWELS

    # Strip a written buffer consonant: y/n before a vowel, n before d/t.
    if len(s) > 1 and ((s[0] in "yn" and s[1] in _VOWELS) or (s[0] == "n" and s[1] in "dt")):
        s = s[1:]

    buffer = ""
    starts_vowel = s[0] in _VOWELS
    is_genitive = bool(re.fullmatch(r"[ıiuü]n", s))
    is_instrumental = bool(re.fullmatch(r"l[ae]", s))
    is_loc_abl = bool(re.fullmatch(r"[dt][ae]n?", s))
    if possessive:
        if starts_vowel or is_loc_abl:
            buffer = "n"
        elif is_instrumental:
            buffer = "y"
    elif ends_vowel:
        if is_genitive:
            buffer = "n"
        elif starts_vowel or is_instrumental:
            buffer = "y"

    out = []
    prev = _last_vowel(base)
    for i, ch in enumerate(s):
        if i == 0 and ch in "dt":
            ch = "d" if buffer or last not in _VOICELESS else "t"
        elif ch in _VOWELS:
            ch = _harmonize_vowel(ch, prev)
            prev = ch
        out.append(ch)
    return base + buffer + "".join(out)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

_SUFFIX = r"(?:['’](?P<suffix>[^\W\d_]+))?"
_AMOUNT = r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)(?:,(?P<frac>\d+))?"
_WB_L = r"(?<![\w])"
_WB_R = r"(?![\w])"

_IBAN_RE = re.compile(
    _WB_L + r"(?P<cc>[A-Z]{2})(?P<check>\d{2})(?P<body>(?:\s?[0-9A-Z]{4}){3,7}(?:\s?[0-9A-Z]{1,3})?)" + _WB_R + _SUFFIX
)
_DATE_RE = re.compile(_WB_L + r"(?P<d>\d{1,2})([./])(?P<m>\d{1,2})\2(?P<y>\d{4})" + _WB_R + _SUFFIX)
_TIME_RE = re.compile(_WB_L + r"(?P<h>[01]?\d|2[0-3]):(?P<min>[0-5]\d)" + _WB_R + _SUFFIX)
_CURRENCY_PREFIX_RE = re.compile(r"₺\s?" + _AMOUNT + _WB_R + _SUFFIX)
_CURRENCY_SUFFIX_RE = re.compile(_WB_L + _AMOUNT + r"\s?(?:TL|₺)" + _WB_R + _SUFFIX)
_CURRENCY_WORD_RE = re.compile(_WB_L + r"TL" + _WB_R + _SUFFIX)
_PERCENT_RE = re.compile(r"%\s?" + _AMOUNT + _WB_R + _SUFFIX)
_NUMBER_RE = re.compile(_WB_L + _AMOUNT + _WB_R + _SUFFIX)


def _with_suffix(spoken: str, m: re.Match, possessive: bool = False) -> str:
    return attach_suffix(spoken, m.group("suffix") or "", possessive)


def _amount_words(int_part: str, frac: str | None) -> str:
    words = _integer_words(int_part.replace(".", ""))
    if frac:
        words += " virgül " + _fraction_words(frac)
    return words


def _iban(m: re.Match) -> str:
    cc = " ".join(LETTER_NAMES.get(c, c) for c in m.group("cc"))
    groups = m.group("body").split() if " " in m.group("body") else re.findall(r".{1,4}", m.group("body"))
    parts = [cc, number_to_words(int(m.group("check")))]
    for g in groups:
        parts.append(" ".join(_DIGITS[int(c)] if c.isdigit() else LETTER_NAMES.get(c, c) for c in g))
    return _with_suffix(", ".join(parts), m)


def _date(m: re.Match) -> str:
    d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        # Not a calendar date: read the three numbers separately.
        return _with_suffix(f"{number_to_words(d)} {number_to_words(mo)} {number_to_words(y)}", m)
    return _with_suffix(f"{number_to_words(d)} {MONTHS[mo - 1]} {number_to_words(y)}", m)


def _time(m: re.Match) -> str:
    h, mi = int(m.group("h")), m.group("min")
    spoken = number_to_words(h)
    if mi != "00":
        spoken += " " + (f"sıfır {_ONES[int(mi)]}" if mi[0] == "0" else number_to_words(int(mi)))
    return _with_suffix(spoken, m)


def _currency(m: re.Match) -> str:
    lira = _integer_words(m.group("int").replace(".", ""))
    frac = m.group("frac")
    if frac and len(frac) <= 2 and int(frac.ljust(2, "0")):
        kurus = number_to_words(int(frac.ljust(2, "0")))
        return _with_suffix(f"{lira} {CURRENCY_WORD} {kurus} {SUBUNIT_WORD}", m)
    if frac and len(frac) > 2:
        lira = _amount_words(m.group("int"), frac)
    return _with_suffix(f"{lira} {CURRENCY_WORD}", m, possessive=True)


def _percent(m: re.Match) -> str:
    return _with_suffix("yüzde " + _amount_words(m.group("int"), m.group("frac")), m)


def _number(m: re.Match) -> str:
    int_part, frac = m.group("int"), m.group("frac")
    digits = int_part.replace(".", "")
    # Long unseparated digit strings (account/phone numbers): read digit by digit.
    if "." not in int_part and not frac and len(digits) >= 6:
        return _with_suffix(digits_to_words(digits), m)
    return _with_suffix(_amount_words(int_part, frac), m)


def _lexicon_pattern(words, ignore_case: bool) -> re.Pattern:
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    # Abbreviations like "vb." end in '.', so the right boundary is not \w-based.
    return re.compile(_WB_L + f"(?P<word>{alts})" + r"(?![\w-])" + _SUFFIX, re.IGNORECASE if ignore_case else 0)


_ABBR_RE = _lexicon_pattern(ABBREVIATIONS, ignore_case=False)
_FOREIGN_RE = _lexicon_pattern(FOREIGN_WORDS, ignore_case=True)


def _abbr(m: re.Match) -> str:
    return _with_suffix(ABBREVIATIONS[m.group("word")], m)


def _foreign(m: re.Match) -> str:
    word = m.group("word")
    spoken = FOREIGN_WORDS[_tr_lower(word)]
    if word[0].isupper():
        spoken = spoken[0].upper() + spoken[1:]
    return _with_suffix(spoken, m)


# Order matters: specific patterns (IBAN, dates, times, money, %) before bare numbers.
_RULES = [
    (_IBAN_RE, _iban),
    (_DATE_RE, _date),
    (_TIME_RE, _time),
    (_CURRENCY_PREFIX_RE, _currency),
    (_CURRENCY_SUFFIX_RE, _currency),
    (_PERCENT_RE, _percent),
    (_NUMBER_RE, _number),
    (_CURRENCY_WORD_RE, lambda m: _with_suffix(CURRENCY_WORD, m, possessive=True)),
    (_ABBR_RE, _abbr),
    (_FOREIGN_RE, _foreign),
]


_SENTENCE_START_RE = re.compile(r"(^|[.!?]\s+)([a-zçğıöşü])")
_TR_UPPER = {"i": "İ", "ı": "I"}


def _capitalize_sentences(t: str) -> str:
    # Expansions like 'ka de ve' can land at a sentence start; restore the capital.
    return _SENTENCE_START_RE.sub(lambda m: m.group(1) + _TR_UPPER.get(m.group(2), m.group(2).upper()), t)


def normalize(text: str) -> str:
    """Return the TTS-ready spoken form of `text`. Pure and deterministic."""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("’", "'").replace("‘", "'")
    for pattern, repl in _RULES:
        t = pattern.sub(repl, t)
    t = re.sub(r"\s+", " ", t).strip()
    return _capitalize_sentences(t)
