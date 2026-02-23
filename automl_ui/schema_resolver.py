def resolve_schema(df_columns, categorical, numeric, target):

    cols = set(df_columns)
    cat = set(categorical)
    num = set(numeric)
    tgt = {target} if target else set()

    unknown = (cat | num | tgt) - cols
    if unknown:
        raise ValueError(f"Columns not found: {unknown}")

    remaining = cols - cat - num - tgt

    # nothing provided
    if not cat and not num and not tgt:
        return [], list(cols), None

    if cat and tgt and not num:
        num = remaining

    elif num and tgt and not cat:
        cat = remaining

    elif cat and num and not tgt:
        if len(remaining) == 1:
            tgt = remaining
        else:
            tgt = None

    elif tgt and not cat and not num:
        num = remaining

    return list(cat), list(num), (list(tgt)[0] if tgt else None)