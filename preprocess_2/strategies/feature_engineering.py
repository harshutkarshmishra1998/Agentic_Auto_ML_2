# import numpy as np
# import pandas as pd


# # SAFETY LIMITS (CRITICAL)

# MAX_INTERACTIONS = 50
# MAX_RATIOS = 50
# MAX_LOG_FEATURES = 50
# MAX_TOTAL_NEW_FEATURES = 200


# def apply(df, rules=None):

#     log = {"strategy": "feature_engineering", "created_features": []}

#     numeric = df.select_dtypes(include=np.number).columns.tolist()
#     categorical = df.select_dtypes(include="object").columns.tolist()

#     new_features = {}
#     created = 0

#     # 1. LOG STABILIZATION (skewed numeric only)
#     log_count = 0
#     for col in numeric:

#         if log_count >= MAX_LOG_FEATURES:
#             break

#         s = df[col]

#         if s.notna().all() and (s > 0).all() and abs(s.skew()) > 1:
#             new_features[f"{col}_log"] = np.log1p(s)
#             log["created_features"].append(f"{col}_log")
#             log_count += 1
#             created += 1

#         if created >= MAX_TOTAL_NEW_FEATURES:
#             break

#     # 2. INTERACTIONS (variance-ranked features)
#     var_rank = df[numeric].var().sort_values(ascending=False).index.tolist()
#     interaction_pairs = []

#     for i in range(len(var_rank)):
#         for j in range(i + 1, len(var_rank)):
#             interaction_pairs.append((var_rank[i], var_rank[j]))

#     for c1, c2 in interaction_pairs[:MAX_INTERACTIONS]:

#         new_col = f"{c1}_x_{c2}"
#         new_features[new_col] = df[c1] * df[c2]
#         log["created_features"].append(new_col)
#         created += 1

#         if created >= MAX_TOTAL_NEW_FEATURES:
#             break

#     # 3. RATIO FEATURES (safe divide)
#     ratio_count = 0
#     for c1, c2 in interaction_pairs:

#         if ratio_count >= MAX_RATIOS:
#             break

#         denom = df[c2]

#         if (denom.abs() > 1e-12).all():
#             new_col = f"{c1}_div_{c2}"
#             new_features[new_col] = df[c1] / (denom + 1e-9)
#             log["created_features"].append(new_col)
#             ratio_count += 1
#             created += 1

#         if created >= MAX_TOTAL_NEW_FEATURES:
#             break

#     # 4. FREQUENCY ENCODING
#     for col in categorical:
#         freq = df[col].value_counts(normalize=True)
#         new_col = f"{col}_freq"
#         new_features[new_col] = df[col].map(freq)
#         log["created_features"].append(new_col)
#         created += 1

#         if created >= MAX_TOTAL_NEW_FEATURES:
#             break

#     # APPLY FEATURES (single concat)
#     if new_features:
#         feat_df = pd.DataFrame(new_features, index=df.index)
#         df = pd.concat([df, feat_df], axis=1)

#     df = df.copy()

#     log["total_created"] = len(log["created_features"])

#     return df, log

def apply(df, rules=None):

    if not rules:
        return df, None

    # future logic here

    return df, {
        "strategy": "feature_engineering",
        "rules": rules
    }