# EcoSort-Waste-Management-Assistance
# EcoSort Waste Management Assistant (Module 8 Summative Lab)

An integrated assistant for Metro City that takes either a **photo** or a **text description** of a waste item,
identifies its category, and returns recycling instructions grounded in the city's policy documents.

## Files

| File | Purpose |
|---|---|
| `waste_management_summative_completed.ipynb` | The completed notebook (all parts 1-5) |
| `waste_descriptions.csv` | 5,000 generated item descriptions with category, disposal instruction, common confusion, material composition |
| `waste_policy_documents.json` | 14 generated Metro City policy documents (9 single-category, 5 combined) |
| `RealWaste/` | Image dataset, **not included**: download from https://archive.ics.uci.edu/dataset/908/realwaste, unzip, and place the class folders in a folder named `RealWaste` next to the notebook |

## How to run

1. Use Google Colab with a GPU runtime (or a local machine with TensorFlow 2.x and PyTorch).
2. Install extras if missing: `pip install transformers sentence-transformers scikit-learn pillow`
3. Put the notebook, the CSV, the JSON and the `RealWaste` folder in the same directory.
4. Run all cells top to bottom. Training the CNN and fine-tuning Flan-T5 take several minutes on a GPU.

The notebook saves two artifacts as it goes: `best_cnn.keras` (image model) and `recycling_generator/` (fine-tuned generator).

## System overview

```
image  --> CNN (MobileNetV2 + custom head) --+
                                              +--> category + confidence
text   --> TF-IDF + logistic regression ------+
                                              v
              retriever (embeddings + category filter) --> policy + disposal tip + common-confusion note
                                              v
                    fine-tuned Flan-T5 --> instructions + source policy documents
```

1. **Image classifier**: MobileNetV2 (ImageNet weights), frozen first, then top 40 layers fine-tuned at a low learning rate. Dropout, L2, augmentation, class weights and early stopping are used. The model is chosen on validation accuracy.
2. **Text classifier**: Option A (traditional ML). Candidates are Naive Bayes, logistic regression and random forest on TF-IDF features, plus a MiniLM-embedding variant when available.
3. **RAG generator**: policies are split into section-level chunks, embedded, and retrieved per query. Flan-T5-base is fine-tuned on question + retrieved context pairs whose targets use only the retrieved text. Several decoding settings (greedy, beam, top-p, top-k) are compared on groundedness and overlap with reference.
4. **Assistant**: `waste_management_assistant(input_data, input_type="image" | "text")` returns a dict with the category, confidence, alternatives, a low-confidence flag, instructions, and source policy documents.

## Results that have been verified

Only the text and retrieval parts were run (on the real CSV and JSON, using a TF-IDF fallback for embeddings):

- Text classifier: 99.7% test accuracy, 5-fold CV about 99.9%. This is almost certainly inflated because the descriptions look template-generated.
- Retrieval on free-text questions: 9 of 9 correct at hit@1.

The CNN, generator and integrated evaluation are written and syntax-checked but need to be run on a GPU; record those numbers from your own run.

## Known issues and design notes

- **Split fix.** The provided validation/test split applies `take()`/`skip()` to a shuffled dataset, so the halves may overlap. The notebook adds a cell that rebuilds them deterministically and disjointly.
- **Policy conflicts.** The Residential, Municipal and Multi-Unit policies list items ("plastic bags and film", "window glass or mirrors") that the dedicated policies exclude. For plastic, the combined policies are the later ones. The system follows the dedicated policy, returns the others as references, and asks Metro City to confirm which prevails.
- **Policy file format.** The lab text mentions `waste_policy_documents.csv`; the supplied file is JSON, and the notebook loads the JSON.
- **Generator limits.** A small model can drop or reword items. The notebook reports groundedness and policy-violation checks; a production system should validate generated lists against the policy text.

## Citation

Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G
