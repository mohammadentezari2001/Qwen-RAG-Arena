#Hybrid RAG Evaluator & Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to benchmark and compare different retrieval strategies. The system evaluates **Lexical (BM25)**, **Semantic (FAISS + Dense Embeddings)**, and **Hybrid (BM25 + FAISS + CrossEncoder Reranking)** approaches.

This project generates answers using the **Qwen2.5-1.5B-Instruct** LLM and evaluates the results automatically using **BERTScore** and an **LLM-as-a-Judge** methodology. It is highly optimized for execution in **Google Colab** (Free Tier compatible) with aggressive memory management.

---

## Features

- **Robust Preprocessing:** Text chunking via LangChain's `RecursiveCharacterTextSplitter`.
- **Multi-Strategy Retrieval:**
  - **Lexical Search:** `BM25Okapi`
  - **Semantic Search:** `sentence-transformers` + `FAISS` Vector Database
  - **Hybrid Search:** Merges top candidates from BM25 and FAISS, reranked using `BAAI/bge-reranker-v2-m3`.
- **LLM Generation:** Uses `Qwen/Qwen2.5-1.5B-Instruct` for context-grounded answer generation.
- **Automated Evaluation:**
  - **BERTScore:** Measures semantic similarity (Precision, Recall, F1) between generated answers and ground truth.
  - **LLM Judge:** A prompt-based LLM evaluator that checks for factual correctness based on the gold answer.
- **Memory Optimized:** Specifically coded to load/unload models and clear CUDA cache (`gc.collect()`, `torch.cuda.empty_cache()`) to prevent Colab crashes.

---

## Project Workflow

1. **Chunking:** Splits the corpus documents into smaller, overlapping chunks (Size: 512, Overlap: 50).
2. **Indexing:** Builds a BM25 index and a FAISS Vector index.
3. **Retrieval:** Retrieves top-K chunks using BM25, FAISS, and the Hybrid/Reranked approach.
4. **Generation:** Formats retrieved contexts and generates answers via the LLM.
5. **Evaluation:** Scores the generated answers against ground-truth answers.

---

## Installation & Setup

This project is built to run effortlessly on Google Colab or any machine with a CUDA-enabled GPU.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Hybrid-RAG-Evaluator.git
   cd Hybrid-RAG-Evaluator
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain langchain-community faiss-cpu rank_bm25 sentence-transformers bert_score torch numpy pandas tqdm accelerate transformers bitsandbytes
   ```

3. **Configure Paths:**
   Edit the `BASE_PATH` in the script to point to your data directory (or your mounted Google Drive if using Colab).

---

## Data Formats

The pipeline expects two `.jsonl` files in your `BASE_PATH`.

**1. `corpus.jsonl`** (The knowledge base)
```json
{"id": "doc_01", "text": "The Eiffel Tower is located in Paris, France."}
{"id": "doc_02", "text": "Python was created by Guido van Rossum."}
```

**2. `test.jsonl`** (The evaluation dataset)
```json
{"question": "Where is the Eiffel Tower?", "answers": ["Paris, France", "Paris"]}
{"question": "Who invented Python?", "answers":["Guido van Rossum"]}
```

---

## Usage

Execute the main script. If you are in Colab, run the cells sequentially.

###  Note on Memory Management
In step 5 (Evaluation), the code requires the LLM to act as a judge. Because the generation model is deleted from memory at the end of Step 4 to save RAM, ensure your `__main__` block re-instantiates it right before evaluation:

```python
# Add this right before evaluating in __main__
judge_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
judge_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
```

---

## Outputs & Metrics

The script generates several intermediate files to save progress and prevent data loss during memory crashes:
- `corpus_chunked.jsonl`
- `bm25_index.pkl` & `faiss_index.bin`
- Retrieval results: `bm25_retrieved.jsonl`, `faiss_retrieved.jsonl`, `hybrid_retrieved.jsonl`
- Generation results: `gen_bm25.jsonl`, `gen_faiss.jsonl`, `gen_hybrid.jsonl`

Finally, it outputs a **`final_report.txt`** containing the benchmark scores:

```text
    Evaluation Report
    =================
    Lexical (BM25):
        BERTScore Precision: 0.8542
        BERTScore Recall:    0.8611
        BERTScore F1:        0.8576
        LLM Judge Accuracy:  0.8200

    Semantic (FAISS):
        BERTScore Precision: 0.8812
        ...
```

---

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/Hybrid-RAG-Evaluator/issues).

## License
This project is open-source and available under the [MIT License](LICENSE).
