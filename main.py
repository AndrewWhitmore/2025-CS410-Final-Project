import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
import numpy as np
import subprocess
import csv

def preprocess_corpus(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r', encoding="utf-8", newline="") as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": f"{i}",  # Changed to match qrels format
                "contents": line.strip()
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as out:
                json.dump(doc, out)


def build_index(input_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    subprocess.run(cmd, check=True)


def load_queries(query_file):
    with open(query_file, 'r') as f:
        return [line.strip() for line in f]


def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                qid, docid, rel = parts
            else:
                raise Exception(f"incorrect line: {line.strip()}")

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    return qrels


def search(searcher, queries, top_k=10, query_id_start=0):
    results = {}
    for i, query in enumerate(tqdm(queries, desc="Searching")):
        hits = searcher.search(query, k=top_k)
        results[str(i + query_id_start)] = [(hit.docid, hit.score) for hit in hits]
    return results


def compute_ndcg(results, qrels, k=10):
    def dcg(relevances):
        # return sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        dcg_simple = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        return dcg_simple

    ndcg_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            # print(f"Query {qid} not found in qrels")
            continue
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results]
        idcg = dcg(sorted(qrels[qid].values(), reverse=True))
        if idcg == 0:
            print(f"IDCG is 0 for query {qid}")
            continue
        ndcg_scores.append(dcg(relevances_current) / idcg)

    if not ndcg_scores:
        print("No valid NDCG scores computed")
        return 0.0
    return np.mean(ndcg_scores)


# New def to computer precision added:
def compute_precision_at_k(results, qrels, k=10):
    precision_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        # Get the top-k retrieved docids for this query
        top_k_docids = [docid for docid, _ in query_results[:k]]

        # Count how many of the top-k documents are relevant
        count_relevant = sum(1 for docid in top_k_docids if qrels[qid].get(docid, 0) > 0)

        # Compute precision at k
        precision = count_relevant / k
        precision_scores.append(precision)

    if not precision_scores:
        print("No valid precision scores computed")
        return 0.0
    return np.mean(precision_scores)


def generate_queries_and_qrels(corpus_file, query_file, qrels_file, num_queries=10):
    """
    Generate query_file (one query per line) and qrels_file (qid docid rel)
    - num_queries: number of queries to generate from corpus
    """
    queries = []
    qrels_lines = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row["text"].strip()
            docid = str(i)

            # For simplicity, choose first num_queries rows as queries
            if i < num_queries:
                queries.append(text)
                # Assign relevance 1 to its own docid
                qrels_lines.append(f"{i} {docid} 1\n")
            else:
                # Optionally, assign other relevance relationships here
                pass

    # Write queries to file
    os.makedirs(os.path.dirname(query_file), exist_ok=True)
    with open(query_file, 'w', encoding='utf-8') as f:
        for q in queries:
            f.write(q + "\n")

    # Write qrels to file
    with open(qrels_file, 'w', encoding='utf-8') as f:
        f.writelines(qrels_lines)

    print(f"Generated {len(queries)} queries and {len(qrels_lines)} qrels")


def main():
    """main function for searching"""

    """=======TODO: Choose Dataset======="""
    cname = "freelaw"
    """============================"""

    base_dir = f"cleaned_data/{cname}"
    query_id_start = {
        "freelaw": 1,
    }[cname]



    # Paths to the raw corpus, queries, and relevance label files
    corpus_file = f"cleaned_data/{cname}.csv"
    query_file = f"cleaned_data/{cname}-queries.txt"
    qrels_file = f"cleaned_data/{cname}-qrels.txt"
    # processed_corpus_dir = os.path.join(base_dir, "corpus")

    # Generate query and qrels files if they don't exist
    if not os.path.exists(query_file) or not os.path.exists(qrels_file):
        generate_queries_and_qrels(corpus_file, query_file, qrels_file, num_queries=50)

    # Directories where the processed corpus and index will be stored for toolkit
    processed_corpus_dir = f"processed_corpus/{cname}"
    os.makedirs(processed_corpus_dir, exist_ok=True)
    index_dir = f"indexes/{cname}"

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # Build index
    build_index(processed_corpus_dir, index_dir)

    # Load queries and qrels
    queries = load_queries(query_file)
    qrels = load_qrels(qrels_file)

    # Debug info
    print(f"Number of queries: {len(queries)}")
    print(f"Number of qrels: {len(qrels)}")
    print(f"Sample qrel: {list(qrels.items())[0] if qrels else 'No qrels'}")

    # Search - original
    #searcher = LuceneSearcher(index_dir)

    # TF-IDF searcher
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.0, b=0)

    """=======TODO: Set Ranking Hyperparameters======="""
    # Part 1 - uncomment to run
    # searcher.set_bm25(k1=0.9, b=.5)

    # third algo uncomment to run
    #searcher.set_rm3(20, 10, 0.5) # optional query expansion
    """========================================="""

    results = search(searcher, queries, query_id_start=query_id_start)

    # Debug info
    print(f"Number of results: {len(results)}")
    print(f"Sample result: {list(results.items())[0] if results else 'No results'}")

    # Evaluate
    topk = 10
    ndcg = compute_ndcg(results, qrels, k=topk)
    cpak = compute_precision_at_k(results, qrels, k=topk)

    print(f"NDCG@{topk}: {ndcg:.4f}")
    print(f"Precision@{topk}: {cpak:.4f}")


    # Save results
    with open(f"results_{cname}.json", "w") as f:
        json.dump({"results": results, "ndcg": ndcg, "Precision": cpak}, f, indent=2)


if __name__ == "__main__":
    main()
