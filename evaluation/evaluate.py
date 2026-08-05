import json
import math

# Load results
with open("results.json", "r") as f:
    results = json.load(f)

total_precision = 0
total_recall = 0
total_mrr = 0
total_hit_rate = 0
total_ndcg = 0
total_context_precision = 0

print("=" * 70)

for item in results:
    question = item["question"]
    expected_page = item["expected_page"]
    retrieved_pages = item["retrieved_pages"]

    k = len(retrieved_pages)

    # Precision@K
    relevant = sum(1 for page in retrieved_pages if page == expected_page)
    precision = relevant / k

    # Recall@K
    recall = relevant / 1

    # Hit Rate
    hit_rate = 1 if expected_page in retrieved_pages else 0

    # MRR
    mrr = 0
    for rank, page in enumerate(retrieved_pages, start=1):
        if page == expected_page:
            mrr = 1 / rank
            break

    # nDCG@K
    dcg = 0
    for rank, page in enumerate(retrieved_pages, start=1):
        if page == expected_page:
            dcg = 1 / math.log2(rank + 1)
            break

    idcg = 1  # Only one relevant page
    ndcg = dcg / idcg

    # Context Precision
    context_precision = 0
    relevant_found = 0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page == expected_page:
            relevant_found += 1
            context_precision += relevant_found / rank

    if relevant_found > 0:
        context_precision /= relevant_found

    # Totals
    total_precision += precision
    total_recall += recall
    total_hit_rate += hit_rate
    total_mrr += mrr
    total_ndcg += ndcg
    total_context_precision += context_precision

    print(f"\nQuestion : {question}")
    print(f"Expected Page : {expected_page}")
    print(f"Retrieved Pages : {retrieved_pages}")
    print(f"Precision@{k}       : {precision:.2f}")
    print(f"Recall@{k}          : {recall:.2f}")
    print(f"Hit Rate            : {hit_rate:.2f}")
    print(f"MRR                 : {mrr:.2f}")
    print(f"nDCG@{k}            : {ndcg:.2f}")
    print(f"Context Precision   : {context_precision:.2f}")

# Overall metrics
n = len(results)

print("\n" + "=" * 70)
print("OVERALL METRICS")
print("=" * 70)
print(f"Average Precision@K      : {total_precision / n:.2f}")
print(f"Average Recall@K         : {total_recall / n:.2f}")
print(f"Average Hit Rate         : {total_hit_rate / n:.2f}")
print(f"Average MRR              : {total_mrr / n:.2f}")
print(f"Average nDCG@K           : {total_ndcg / n:.2f}")
print(f"Average Context Precision: {total_context_precision / n:.2f}")