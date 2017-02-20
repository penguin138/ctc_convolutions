
def top_k(values, k=1):
    return sorted(values, reverse=True)[:k]


def top_k_indices(values, k=1):
    return sorted(range(len(values)), key=lambda x: values[x], reverse=True)[:k]
