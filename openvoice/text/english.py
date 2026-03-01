import re
import inflect
from unidecode import unidecode

try:
    import eng_to_ipa as ipa
    _IPA_AVAILABLE = True
except ImportError:
    _IPA_AVAILABLE = False

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"\xa3([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_abbreviations = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
        ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
        ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ]
]

_lazy_ipa2 = [
    (re.compile(x[0]), x[1])
    for x in [
        ("r", "\u0279"), ("\xf0", "z"), ("\u03b8", "s"),
        ("\u0292", "\u0291"), ("\u02a4", "d\u0291"), ("\u02c8", "\u2193"),
    ]
]

_ipa_to_ipa2 = [
    (re.compile(x[0]), x[1])
    for x in [("r", "\u0279"), ("\u02a4", "d\u0292"), ("\u02a7", "t\u0283")]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text)


def normalize_numbers(text):
    text = re.sub(_comma_number_re, lambda m: m.group(1).replace(",", ""), text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, lambda m: m.group(1).replace(".", " point "), text)
    text = re.sub(_ordinal_re, lambda m: _inflect.number_to_words(m.group(0)), text)
    text = re.sub(_number_re, lambda m: _inflect.number_to_words(int(m.group(0)), andword=""), text)
    return text


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        return f"{dollars} {'dollar' if dollars == 1 else 'dollars'}, {cents} {'cent' if cents == 1 else 'cents'}"
    elif dollars:
        return f"{dollars} {'dollar' if dollars == 1 else 'dollars'}"
    elif cents:
        return f"{cents} {'cent' if cents == 1 else 'cents'}"
    return "zero dollars"


def mark_dark_l(text):
    return re.sub(r"l([^aeiou\xe6\u0251\u0254\u0259\u025b\u026a\u028a ]*(?: |$))", lambda x: "\u026b" + x.group(1), text)


def english_to_ipa(text):
    text = unidecode(text).lower()
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    if _IPA_AVAILABLE:
        phonemes = ipa.convert(text)
    else:
        phonemes = text
    return collapse_whitespace(phonemes)


def english_to_lazy_ipa(text):
    text = english_to_ipa(text)
    for regex, replacement in _lazy_ipa2:
        text = re.sub(regex, replacement, text)
    return text


def english_to_ipa2(text):
    text = english_to_ipa(text)
    text = mark_dark_l(text)
    for regex, replacement in _ipa_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text.replace("...", "\u2026")


def english_to_lazy_ipa2(text):
    return english_to_lazy_ipa(text)
