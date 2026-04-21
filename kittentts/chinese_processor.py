"""
chinese_processor.py
Chinese text processing module for KittenTTS.
Converts Chinese text to pinyin for speech synthesis.
"""

import re
from typing import List, Tuple, Optional

try:
    from pypinyin import pinyin, Style
    PINYIN_AVAILABLE = True
except ImportError:
    PINYIN_AVAILABLE = False

try:
    import cn2an
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False


CHINESE_PUNCTUATION = {
    '。': '.',
    '，': ',',
    '、': ',',
    '；': ';',
    '：': ':',
    '？': '?',
    '！': '!',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '「': '"',
    '」': '"',
    '『': '"',
    '』': '"',
    '《': '"',
    '》': '"',
    '…': '...',
    '—': '-',
    '～': '~',
}

PINYIN_TONE_MAP = {
    'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
    'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
    'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
    'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
    'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
    'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v',
    'ü': 'v',
}

CHINESE_DIGITS = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}

CHINESE_UNITS = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿']


def has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_chinese_char(char: str) -> bool:
    """Check if a character is a Chinese character."""
    return '\u4e00' <= char <= '\u9fff'


def normalize_chinese_punctuation(text: str) -> str:
    """Convert Chinese punctuation to English punctuation."""
    for cn_punct, en_punct in CHINESE_PUNCTUATION.items():
        text = text.replace(cn_punct, en_punct)
    return text


def number_to_chinese_words(n: int) -> str:
    """Convert an integer to Chinese word representation.
    
    Examples:
        123 → "一百二十三"
        10000 → "一万"
        0 → "零"
    """
    if n == 0:
        return "零"
    
    if n < 0:
        return "负" + number_to_chinese_words(-n)
    
    if CN2AN_AVAILABLE:
        try:
            return cn2an.an2cn(n)
        except:
            pass
    
    digits = list(str(n))
    length = len(digits)
    result = []
    
    for i, digit in enumerate(digits):
        pos = length - i - 1
        d = int(digit)
        
        if d != 0:
            result.append(CHINESE_DIGITS[str(d)])
            if pos > 0:
                result.append(CHINESE_UNITS[pos])
        else:
            if result and result[-1] != '零':
                if i < length - 1 and any(int(digits[j]) != 0 for j in range(i + 1, length)):
                    result.append('零')
    
    chinese_str = ''.join(result)
    
    if length == 2 and digits[0] == '1':
        chinese_str = chinese_str[1:] if chinese_str.startswith('一') else chinese_str
    
    return chinese_str


def float_to_chinese_words(value, decimal_sep: str = "点") -> str:
    """Convert a float to Chinese words.
    
    Examples:
        3.14 → "三点一四"
        -0.5 → "负零点五"
    """
    text = value if isinstance(value, str) else f"{value}"
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    
    if "." in text:
        int_part, dec_part = text.split(".", 1)
        int_words = number_to_chinese_words(int(int_part)) if int_part else "零"
        dec_words = "".join(CHINESE_DIGITS.get(d, d) for d in dec_part)
        result = f"{int_words}{decimal_sep}{dec_words}"
    else:
        result = number_to_chinese_words(int(text))
    
    return f"负{result}" if negative else result


def chinese_number_to_words(text: str) -> str:
    """Replace Arabic numbers in Chinese context with Chinese words.
    
    Examples:
        "我有123个苹果" → "我有一百二十三个苹果"
        "价格是3.14元" → "价格是三点一四元"
    """
    def replace_int(match):
        num = int(match.group(0))
        return number_to_chinese_words(num)
    
    def replace_float(match):
        raw = match.group(0)
        return float_to_chinese_words(raw)
    
    text = re.sub(r'(?<![a-zA-Z])-?\d+\.\d+', replace_float, text)
    text = re.sub(r'(?<![a-zA-Z])-?\d+', replace_int, text)
    
    return text


def chinese_to_pinyin(text: str, tone: bool = False) -> str:
    """Convert Chinese text to pinyin.
    
    Args:
        text: Chinese text
        tone: If True, keep tone marks (for display/subtitles)
    
    Returns:
        Pinyin string
    """
    if not PINYIN_AVAILABLE:
        return text
    
    result = []
    
    if tone:
        pinyin_list = pinyin(text, style=Style.TONE)
    else:
        pinyin_list = pinyin(text, style=Style.NORMAL)
    
    for i, py in enumerate(pinyin_list):
        if not py:
            continue
        p = py[0]
        
        if not tone:
            p = ''.join(PINYIN_TONE_MAP.get(c, c) for c in p)
        
        if i > 0 and result:
            last_char = result[-1]
            if last_char and last_char[-1].isalpha() and p[0].isalpha():
                result.append(' ')
        
        result.append(p)
    
    return ''.join(result)


def split_mixed_text(text: str) -> List[Tuple[str, str]]:
    """Split mixed Chinese-English text into segments.
    
    Returns:
        List of (segment, type) tuples, where type is 'chinese' or 'english'
    """
    segments = []
    current_segment = []
    current_type = None
    
    for char in text:
        if is_chinese_char(char):
            char_type = 'chinese'
        elif char.isalpha() or char.isdigit():
            char_type = 'english'
        else:
            char_type = 'punctuation'
        
        if char_type == 'punctuation':
            if current_segment:
                segments.append((''.join(current_segment), current_type))
                current_segment = []
                current_type = None
            segments.append((char, 'punctuation'))
        else:
            if current_type is not None and current_type != char_type:
                segments.append((''.join(current_segment), current_type))
                current_segment = []
            current_segment.append(char)
            current_type = char_type
    
    if current_segment:
        segments.append((''.join(current_segment), current_type))
    
    return segments


def process_mixed_text(text: str, for_synthesis: bool = True) -> Tuple[str, str]:
    """Process mixed Chinese-English text for TTS.
    
    Args:
        text: Input text (may contain Chinese, English, numbers)
        for_synthesis: If True, convert Chinese to pinyin without tones
                       If False, keep original for subtitles
    
    Returns:
        Tuple of (processed_text, original_text_for_subtitles)
    """
    original_text = text
    
    text = normalize_chinese_punctuation(text)
    
    if for_synthesis:
        segments = split_mixed_text(text)
        processed_parts = []
        
        for i, (segment, seg_type) in enumerate(segments):
            if seg_type == 'chinese':
                if PINYIN_AVAILABLE:
                    processed_parts.append((' ' + chinese_to_pinyin(segment, tone=False) + ' ', seg_type))
                else:
                    processed_parts.append((segment, seg_type))
            elif seg_type == 'english':
                processed_parts.append((' ' + segment + ' ', seg_type))
            else:
                processed_parts.append((segment, seg_type))
        
        processed_text = ''.join(p[0] for p in processed_parts)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    else:
        processed_text = text
    
    return processed_text, original_text


def extract_subtitle_info(text: str) -> dict:
    """Extract subtitle information from text.
    
    Returns:
        Dictionary containing:
        - original: Original text
        - pinyin: Pinyin transcription (if Chinese)
        - segments: List of segments with type and content
    """
    segments = split_mixed_text(text)
    
    pinyin_segments = []
    for segment, seg_type in segments:
        if seg_type == 'chinese' and PINYIN_AVAILABLE:
            pinyin_segments.append((chinese_to_pinyin(segment, tone=True), 'chinese'))
        else:
            pinyin_segments.append((segment, seg_type))
    
    full_pinyin = ''.join(s[0] for s in pinyin_segments)
    
    return {
        'original': text,
        'pinyin': full_pinyin,
        'segments': [{'content': s[0], 'type': s[1], 'pinyin': p[0] if s[1] == 'chinese' else s[0]} 
                     for s, p in zip(segments, pinyin_segments)]
    }


class ChineseTextProcessor:
    """Chinese text processor for KittenTTS.
    
    This class handles:
    - Chinese text detection
    - Chinese punctuation normalization
    - Chinese number conversion
    - Chinese to pinyin conversion
    - Mixed Chinese-English text processing
    """
    
    def __init__(self):
        self.pinyin_available = PINYIN_AVAILABLE
        self.cn2an_available = CN2AN_AVAILABLE
    
    def has_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return has_chinese(text)
    
    def normalize_punctuation(self, text: str) -> str:
        """Convert Chinese punctuation to English."""
        return normalize_chinese_punctuation(text)
    
    def convert_numbers(self, text: str) -> str:
        """Convert Arabic numbers to Chinese words in Chinese context."""
        return chinese_number_to_words(text)
    
    def to_pinyin(self, text: str, tone: bool = False) -> str:
        """Convert Chinese text to pinyin."""
        return chinese_to_pinyin(text, tone=tone)
    
    def process(self, text: str, convert_pinyin: bool = True) -> Tuple[str, dict]:
        """Process text for TTS synthesis.
        
        Args:
            text: Input text
            convert_pinyin: If True, convert Chinese to pinyin
        
        Returns:
            Tuple of (processed_text, subtitle_info)
        """
        subtitle_info = extract_subtitle_info(text)
        
        if convert_pinyin and self.has_chinese(text):
            processed_text, _ = process_mixed_text(text, for_synthesis=True)
        else:
            processed_text = text
        
        return processed_text, subtitle_info


def get_chinese_processor() -> ChineseTextProcessor:
    """Get a Chinese text processor instance."""
    return ChineseTextProcessor()


def check_dependencies() -> dict:
    """Check if required dependencies for Chinese support are installed.
    
    Returns:
        Dictionary with dependency status
    """
    return {
        'pypinyin': PINYIN_AVAILABLE,
        'cn2an': CN2AN_AVAILABLE,
        'full_support': PINYIN_AVAILABLE and CN2AN_AVAILABLE,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Chinese Text Processor Demo")
    print("=" * 60)
    
    deps = check_dependencies()
    print(f"\nDependencies:")
    print(f"  pypinyin: {'✓' if deps['pypinyin'] else '✗'}")
    print(f"  cn2an: {'✓' if deps['cn2an'] else '✗'}")
    
    processor = ChineseTextProcessor()
    
    test_cases = [
        "你好，世界！",
        "我有123个苹果。",
        "价格是3.14元。",
        "Hello，我是KittenTTS！",
        "今天天气真好，气温是25度。",
        "这是一个mixed test：中文和English混合。",
    ]
    
    print("\n" + "=" * 60)
    print("Test Cases")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\nInput: {text}")
        print(f"Has Chinese: {processor.has_chinese(text)}")
        
        processed, subtitle = processor.process(text)
        print(f"Processed: {processed}")
        print(f"Pinyin: {subtitle['pinyin']}")
