from functools import wraps

# memoize helper
def memoize(f):
    cache = {}
    @wraps(f)
    def decorated(*args):
        key = (f, str(args))
        result = cache.get(key, None)
        if result is None:
            result = f(*args)
            cache[key] = result
        return result
    return decorated

# takes in the script as a one liner string
def script_tokenizer(script):
    # keep all the punctuation
    tokens = script.split(' ')
    tokens[-1] = tokens[-1].replace('\n', '')
    return tokens
