import re
import json
import numpy as np


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def string_to_bits(string, pad_len=8):
    ascii_values = [ord(char) for char in string]
    
    binary_values = [bin(value)[2:].zfill(8) for value in ascii_values]
    
    bit_arrays = [[int(bit) for bit in binary] for binary in binary_values]
    
    numpy_array = np.array(bit_arrays)
    numpy_array_full = np.zeros((pad_len, 8), dtype=numpy_array.dtype)
    numpy_array_full[:, 2] = 1
    max_len = min(pad_len, len(numpy_array))
    numpy_array_full[:max_len] = numpy_array[:max_len]
    return numpy_array_full


def bits_to_string(bits_array):
    binary_values = [''.join(str(bit) for bit in row) for row in bits_array]
    
    ascii_values = [int(binary, 2) for binary in binary_values]
    
    output_string = ''.join(chr(value) for value in ascii_values)
    
    return output_string


def split_sentence(text, min_len=10, language_str='[EN]'):
    if language_str in ['EN']:
        sentences = split_sentences_latin(text, min_len=min_len)
    else:
        sentences = split_sentences_zh(text, min_len=min_len)
    return sentences

def split_sentences_latin(text, min_len=10):
    """Split Long sentences into list of short ones

    Args:
        str: Input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    text = re.sub('[πÇé∩╝ü∩╝ƒ∩╝¢]', '.', text)
    text = re.sub('[∩╝î]', ',', text)
    text = re.sub('[ΓÇ£ΓÇ¥]', '"', text)
    text = re.sub('[ΓÇÿΓÇÖ]', "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\┬½\┬╗]+", "", text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent.split(" "))
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_latin(new_sentences)


def merge_short_sentences_latin(sens):
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

def split_sentences_zh(text, min_len=10):
    text = re.sub('[πÇé∩╝ü∩╝ƒ∩╝¢]', '.', text)
    text = re.sub('[∩╝î]', ',', text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def merge_short_sentences_zh(sens):
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out
