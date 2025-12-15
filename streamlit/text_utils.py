def correct_similarities_to_parenthesis(text):
    chars_similar_to_parenthesis = ["O", "0", "Q", "®"]
    for char in chars_similar_to_parenthesis:
        if char in text and text.index(char) == len(text) - 1 and text[len(text) - 2].isalpha():
            return text[:-1] + "()"
        if char + ":" in text:
            return text.replace(char + ":", "():")

    return text

def is_method(text):
    if "(" in text and ")" in text and text.index("(") < text.index(")"):
        return True
