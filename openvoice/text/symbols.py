_pad = "_"
_punctuation = ",.!?-~\u2026"
_letters = "NQabdefghijklmnopstuvwxyz\u0251\xe6\u0283\u0291\xe7\u026f\u026a\u0254\u025b\u0279\xf0\u0259\u026b\u0265\u00f8\u028a\u027e\u02b0\u03b2\u014b\u0266\u02c8\u02cc\u2192\u2193\u2191 "

symbols = [_pad] + list(_punctuation) + list(_letters)

SPACE_ID = symbols.index(" ")

num_zh_tones = 6
num_ja_tones = 1
num_en_tones = 4
num_kr_tones = 1

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
    "KR": num_zh_tones + num_ja_tones + num_en_tones,
}
