from edit_distance import SequenceMatcher
# from functools import reduce
# error_codes = ['replace', 'delete', 'insert']

def get_match_count(sm):
    "Return the number of matches, given a sequence matcher object."
    matches = sm.matches()
    return matches

"""Return the number of errors (insertion, deletion, and substitutiions
    , given a sequence matcher object.
"""
# def get_error_count(sm):
#     opcodes = sm.get_opcodes()
#     errors = [x for x in opcodes if x[0] in error_codes]
#     error_lengths = [max(x[2] - x[1], x[4] - x[3]) for x in errors]
#     return reduce(lambda x, y: x + y, error_lengths, 0)

def wer(ref, hyp):
    ref = ref.strip().replace("'","").split()
    hyp = hyp.strip().replace("'","").split()
    ref = list(map(str.lower, ref))
    hyp = list(map(str.lower, hyp))

    ref_token_count = len(ref)

    sm = SequenceMatcher(a=ref, b=hyp)
    # error_count = get_error_count(sm)
    match_count = get_match_count(sm)
    # wrr = match_count / ref_token_count
    wer = 1 - match_count / ref_token_count
    return wer, match_count, ref_token_count
