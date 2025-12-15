import zlib

# Encoding table as per PlantUML spec
PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

def encode6bit(b):
    if 0 <= b <= 63:
        return PLANTUML_ALPHABET[b]
    raise ValueError("Value out of range for 6-bit encoding")

def append3bytes(b1, b2, b3):
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return ''.join([
        encode6bit(c1),
        encode6bit(c2),
        encode6bit(c3),
        encode6bit(c4)
    ])

def encode_plantuml(source: str) -> str:
    """
    Encodes a PlantUML source string using Deflate compression and PlantUML base64-like encoding.
    """
    # Step 1: UTF-8 encode the input string
    utf8_bytes = source.encode('utf-8')

    # Step 2: Deflate compression (strip zlib headers)
    compressed = zlib.compress(utf8_bytes)[2:-4]

    # Step 3: PlantUML-specific base64-like encoding
    encoded = ''
    i = 0
    while i < len(compressed):
        b1 = compressed[i]
        b2 = compressed[i+1] if i+1 < len(compressed) else 0
        b3 = compressed[i+2] if i+2 < len(compressed) else 0
        encoded += append3bytes(b1, b2, b3)
        i += 3

    return encoded