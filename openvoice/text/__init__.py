from .symbols import symbols
from . import cleaners

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, syms, cleaner_names):
    sequence = []
    symbol_to_id = {s: i for i, s in enumerate(syms)}
    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        if symbol in symbol_to_id:
            sequence.append(symbol_to_id[symbol])
    return sequence


def cleaned_text_to_sequence(cleaned_text, syms):
    symbol_to_id = {s: i for i, s in enumerate(syms)}
    return [symbol_to_id[symbol] for symbol in cleaned_text if symbol in symbol_to_id]


def sequence_to_text(sequence):
    return "".join(_id_to_symbol.get(sid, "") for sid in sequence)


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if cleaner is None:
            raise ValueError(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text
