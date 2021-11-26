from math import ceil, floor


def round_any(x, base, round_type='round'):
    """
    rounds the number x to the nearest multiple of base with the rounding done according to round_type.
    e.g. round_any(3, 5) = 5. round_any(3, 5, 'floor') = 0.
    """
    if round_type == 'round':
        return base * round(x / base)
    elif round_type == 'ceil':
        return base * ceil(x / base)
    elif round_type == 'floor':
        return base * floor(x / base)
