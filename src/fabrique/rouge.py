from collections import Counter
from typing import List, Tuple


def get_ngrams(tokens: List[int], n: int) -> Counter:
    """Extract n-grams from a list of tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))
    return Counter(ngrams)


def rouge_n(reference: List[int], candidate: List[int], n: int = 1) -> dict:
    """
    Calculate ROUGE-N score between reference and candidate.

    Args:
        reference: Reference token list
        candidate: Candidate token list
        n: N-gram size (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if len(reference) < n or len(candidate) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_ngrams = get_ngrams(reference, n)
    cand_ngrams = get_ngrams(candidate, n)

    # Count overlapping n-grams
    overlap = sum((ref_ngrams & cand_ngrams).values())

    # Calculate precision and recall
    total_cand = sum(cand_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = overlap / total_cand if total_cand > 0 else 0.0
    recall = overlap / total_ref if total_ref > 0 else 0.0

    # Calculate F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(x: List[int], y: List[int]) -> Tuple[int, List[List[int]]]:
    """
    Calculate the longest common subsequence (LCS) length and DP table.

    Args:
        x: First token sequence
        y: Second token sequence

    Returns:
        Tuple of (LCS length, DP table)
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n], dp


def rouge_l(reference: List[int], candidate: List[int]) -> dict:
    """
    Calculate ROUGE-L score (based on longest common subsequence).

    Args:
        reference: Reference token list
        candidate: Candidate token list

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if len(reference) == 0 or len(candidate) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len, _ = lcs_length(reference, candidate)

    precision = lcs_len / len(candidate)
    recall = lcs_len / len(reference)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# # Example usage
# if __name__ == "__main__":
#     # Example with token IDs
#     reference = [1, 2, 3, 4, 5, 6, 7]
#     candidate = [1, 2, 3, 8, 9, 6, 7]

#     print("Reference:", reference)
#     print("Candidate:", candidate)
#     print()

#     # ROUGE-1 (unigram overlap)
#     r1 = rouge_n(reference, candidate, n=1)
#     print(f"ROUGE-1: Precision={r1['precision']:.3f}, Recall={r1['recall']:.3f}, F1={r1['f1']:.3f}")

#     # ROUGE-2 (bigram overlap)
#     r2 = rouge_n(reference, candidate, n=2)
#     print(f"ROUGE-2: Precision={r2['precision']:.3f}, Recall={r2['recall']:.3f}, F1={r2['f1']:.3f}")

#     # ROUGE-L (longest common subsequence)
#     rl = rouge_l(reference, candidate)
#     print(f"ROUGE-L: Precision={rl['precision']:.3f}, Recall={rl['recall']:.3f}, F1={rl['f1']:.3f}")
