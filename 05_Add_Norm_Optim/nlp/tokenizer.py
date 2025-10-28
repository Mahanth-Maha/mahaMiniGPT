def enc(x, chars):
    return [chars.index(c) for c in x]

def dec(x, chars):
    return "".join(chars[i] for i in x)

