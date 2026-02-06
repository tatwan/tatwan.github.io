### Foundation Model Evaluation Metrics Decision Tree (v1.0)

## 1. **What aspect of your Foundation Model are you evaluating?**
   - **Language Modeling Quality** (How well does the model predict/understand text?) → Go to 2
   - **Generation vs Reference** (Compare outputs against gold standard text) → Go to 3
   - **Task Performance** (Classification, QA, extraction, ranking accuracy) → Go to 4
   - **Human-like Quality** (Helpfulness, coherence, overall preference) → Go to 5
   - **Safety & Alignment** (Toxicity, bias, policy compliance) → Go to 6
   - **Factuality & Hallucination** (Truthfulness, grounding, RAG faithfulness) → Go to 7

---

## 2. **Intrinsic Language Modeling Metrics**

*Goal: Measure how well the model fits/understands text distributions*

These metrics evaluate the model's core language modeling capability without task-specific generation.

### **Primary Metrics:**

- **Perplexity (PPL)**
  - Measures how "surprised" the model is by real text. Lower perplexity = better language model.
  - **Range:** 1 to infinity (lower is better)
  - **Formula:** `PPL = exp(-1/N * sum(log P(token_i)))`
  - **When to use:** Comparing LMs, monitoring training, evaluating domain fit
  - **Pros:** Standard benchmark, easy to compute, interpretable
  - **Cons:** Doesn't capture generation quality, tokenizer-dependent, not task-specific
  - **Note:** A perplexity of 10 means the model is as uncertain as choosing from 10 equally likely options

- **Cross-Entropy Loss**
  - Average negative log-likelihood per token. Directly related to perplexity (PPL = exp(CE)).
  - **Range:** 0 to infinity (lower is better)
  - **Formula:** `CE = -1/N * sum(log P(token_i))`
  - **When to use:** Training loss monitoring, fine-tuning evaluation
  - **Note:** What you actually optimize during training

- **Bits-per-Byte (BPB)**
  - Cross-entropy normalized by bytes, not tokens. Allows comparison across different tokenizers.
  - **Range:** 0 to infinity (lower is better)
  - **Formula:** `BPB = CE * tokens / bytes * log2(e)`
  - **When to use:** Comparing models with different tokenizers
  - **Pros:** Fair comparison across tokenizers, used in GPT-4 report
  - **Note:** Increasingly used in frontier model papers

- **Bits-per-Character (BPC)**
  - Similar to BPB but normalized per character. Common for character-level models.
  - **When to use:** Character-level model evaluation
  - **Note:** Less common for modern tokenizer-based LLMs

### **When to use intrinsic metrics:**
- Pre-training evaluation and monitoring
- Comparing language models on same corpus
- Domain adaptation assessment
- **NOT sufficient alone for user-facing quality**

---

## 3. **Reference-Based Metrics (Generation vs Gold Standard)**

### 3a. **What type of generation task?**
   - **Summarization** → Go to 3b
   - **Translation** → Go to 3c
   - **General Text Generation** → Go to 3d
   - **Semantic Similarity** → Go to 3e

---

### 3b. **Summarization Metrics**

*Goal: Measure how well the summary captures reference content*

- **ROUGE-1**
  - Unigram (word) overlap between generated and reference summary. Recall-oriented.
  - **Range:** 0 to 1 (higher is better)
  - **Formula:** `ROUGE-1 = |matched unigrams| / |reference unigrams|`
  - **Variants:** ROUGE-1-R (recall), ROUGE-1-P (precision), ROUGE-1-F (F1)

- **ROUGE-2**
  - Bigram overlap. Captures some phrase-level similarity.
  - **Range:** 0 to 1 (higher is better)
  - **Note:** More sensitive to word order than ROUGE-1

- **ROUGE-L**
  - Longest Common Subsequence. Captures sentence-level structure.
  - **Range:** 0 to 1 (higher is better)
  - **Pros:** Order-sensitive, no fixed n-gram length
  - **Note:** Good for longer summaries

- **ROUGE-Lsum**
  - ROUGE-L computed at summary level (across sentences).
  - **When to use:** Multi-sentence summary evaluation
  - **Note:** Default for summarization benchmarks

- **BERTScore** ⭐ Recommended
  - Embedding-based similarity using BERT. Captures paraphrases better than n-gram methods.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Semantic similarity matters, paraphrasing acceptable
  - **Pros:** Handles paraphrases, semantic matching, correlates with human judgment
  - **Cons:** Slower to compute, model-dependent
  - **Tools:** bert-score Python package

- **SummaC**
  - NLI-based metric checking if summary is consistent with source document.
  - **When to use:** Detecting unfaithful summaries
  - **Paper:** SummaC: Re-Visiting NLI-based Models for Inconsistency Detection (2022)
  - **Note:** Complements ROUGE with faithfulness checking

### **Summarization Metric Selection:**
- ROUGE-1/2/L: Standard baseline, always report
- BERTScore: Better correlation with humans
- SummaC: Add for faithfulness checking
- Combine multiple metrics for complete picture

---

### 3c. **Machine Translation Metrics**

*Goal: Measure translation quality against reference translations*

- **BLEU**
  - Precision-oriented n-gram overlap with brevity penalty. Industry standard for decades.
  - **Range:** 0 to 100 (higher is better)
  - **Formula:** `BLEU = BP * exp(sum(w_n * log(p_n)))`
  - **When to use:** Standard MT evaluation, comparing to literature
  - **Pros:** Well-understood, fast, corpus-level reliable
  - **Cons:** Poor at sentence-level, ignores semantics, exact match only
  - **Variants:** BLEU-4 (up to 4-grams), SacreBLEU (standardized)
  - **Note:** Always use SacreBLEU for reproducibility

- **chrF / chrF++**
  - Character n-gram F-score. Better for morphologically rich languages.
  - **Range:** 0 to 100 (higher is better)
  - **When to use:** Agglutinative languages, word segmentation issues
  - **Pros:** No tokenization needed, works across writing systems
  - **Tools:** SacreBLEU

- **METEOR**
  - Considers synonyms, stems, and paraphrases. Better correlation with human judgment.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Want semantic matching beyond exact overlap
  - **Pros:** Handles synonyms, recall-focused
  - **Cons:** Language-specific resources needed

- **COMET** ⭐ Recommended
  - Learned neural metric trained on human judgments. State-of-the-art correlation.
  - **Range:** Varies by model (higher is better)
  - **When to use:** Best correlation with human quality assessment
  - **Pros:** Best human correlation, reference-free variants available
  - **Cons:** Requires model inference, slower
  - **Paper:** COMET: A Neural Framework for MT Evaluation (2020)
  - **Tools:** Unbabel/COMET

- **BLEURT**
  - BERT-based learned metric. Pre-trained then fine-tuned on human ratings.
  - **Range:** Typically 0-1 (higher is better)
  - **When to use:** Neural semantic similarity for translation
  - **Tools:** Google BLEURT

### **MT Metric Recommendations:**
- Always report BLEU (SacreBLEU) for comparability
- Add COMET for better human correlation
- chrF for morphologically rich languages
- Consider reference-free COMET for production

---

### 3d. **General Text Generation Metrics**

*Goal: Evaluate open-ended text generation quality*

- **MAUVE**
  - Compares distribution of generated text to human text. Captures diversity and coherence.
  - **Range:** 0 to 1 (higher is better, closer to human)
  - **When to use:** Open-ended generation, story writing, dialogue
  - **Pros:** Captures diversity, distribution-level comparison, no single reference needed
  - **Cons:** Needs corpus of generations, computationally heavier
  - **Paper:** MAUVE: Measuring the Gap Between Neural Text and Human Text (2021)
  - **Note:** Best for evaluating creative/diverse generation

- **Distinct-n**
  - Ratio of unique n-grams to total n-grams. Measures lexical diversity.
  - **Range:** 0 to 1 (higher = more diverse)
  - **When to use:** Checking for repetitive/boring generation
  - **Variants:** Distinct-1 (unigrams), Distinct-2 (bigrams)
  - **Note:** Simple but useful diversity check

- **Self-BLEU**
  - BLEU of each generation against others. Lower = more diverse.
  - **Range:** 0 to 100 (lower is more diverse)
  - **When to use:** Measuring generation diversity in a set
  - **Note:** High Self-BLEU = repetitive outputs

- **Repetition Rate**
  - Frequency of repeated n-grams within a single generation.
  - **When to use:** Detecting degenerate repetition
  - **Note:** Critical for long-form generation

- **PARENT**
  - For data-to-text generation. Balances fidelity to data and fluency.
  - **When to use:** Table-to-text, structured data generation
  - **Paper:** Handling Divergent Reference Texts (2019)

---

### 3e. **Semantic Similarity Metrics**

*Goal: Measure meaning preservation regardless of exact wording*

- **BERTScore**
  - Token-level cosine similarity using BERT embeddings. Handles paraphrases well.
  - **Range:** 0 to 1 (higher is better)
  - **Formula:** BERTScore = F1 of precision and recall over token similarities
  - **When to use:** Paraphrase detection, semantic equivalence
  - **Tools:** bert-score, HuggingFace evaluate

- **MoverScore**
  - Uses Word Mover's Distance with contextualized embeddings.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Need soft alignment between words
  - **Paper:** MoverScore: Text Generation Evaluating with Contextualized Embeddings (2019)

- **Sentence Transformers Similarity**
  - Cosine similarity of sentence embeddings from models like all-MiniLM.
  - **Range:** -1 to 1 (higher is better)
  - **When to use:** Quick semantic similarity check
  - **Pros:** Fast, good sentence-level semantics
  - **Tools:** sentence-transformers library

- **SimCSE**
  - Contrastively trained sentence embeddings. Strong semantic similarity.
  - **When to use:** High-quality sentence similarity needed
  - **Paper:** SimCSE: Simple Contrastive Learning of Sentence Embeddings (2021)

- **BLEURT**
  - BERT fine-tuned on human quality judgments. Captures more than pure similarity.
  - **When to use:** Need quality score, not just similarity
  - **Tools:** Google BLEURT

---

## 4. **Task Performance Metrics**

### 4a. **What type of task output?**
   - **Classification / Labels** → Go to 4b
   - **Question Answering** → Go to 4c
   - **Information Extraction** → Go to 4d
   - **Ranking / Retrieval** → Go to 4e
   - **Reasoning / Math** → Go to 4f
   - **Code Generation** → Go to 4g

---

### 4b. **Classification Metrics for LLMs**

*When using LLMs for classification, extract the predicted label and compute standard metrics.*

- **Accuracy**
  - Fraction of correct predictions. Misleading for imbalanced classes.
  - **Range:** 0 to 1 (higher is better)
  - **Formula:** `Accuracy = Correct / Total`
  - **When to use:** Balanced classes only
  - **Warning:** Avoid for imbalanced data

- **Macro F1**
  - Average F1 across classes. Treats all classes equally regardless of size.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Imbalanced classes, care about minority classes
  - **Note:** Standard for LLM classification benchmarks

- **Weighted F1**
  - F1 weighted by class frequency.
  - **When to use:** Want to weight by class prevalence

- **AUC-ROC / AUC-PR**
  - If LLM outputs probabilities/logits, use AUC for threshold-independent evaluation.
  - **When to use:** Have probability outputs, especially imbalanced data

- **Cohen's Kappa**
  - Agreement beyond chance. Good for multi-class with class imbalance.
  - **Range:** -1 to 1 (1 is perfect)
  - **When to use:** Want to account for chance agreement

### **LLM Classification Evaluation Tips:**
- Parse LLM output carefully for label extraction
- Consider constrained decoding for valid labels
- Report macro F1 for benchmark comparability
- Include confusion matrix for error analysis

---

### 4c. **Question Answering Metrics**

- **Exact Match (EM)**
  - Binary: is the predicted answer exactly equal to gold answer (after normalization)?
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Short, factoid answers (SQuAD-style)
  - **Normalization:** Lowercase, remove articles/punctuation
  - **Note:** Standard for extractive QA

- **F1 Score (Token-Level)**
  - Token overlap F1 between prediction and gold. Gives partial credit.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Want to reward partial matches
  - **Note:** More forgiving than EM

- **ROUGE-L**
  - For longer generative answers. LCS-based similarity.
  - **When to use:** Long-form answers, not just extractive

- **BERTScore**
  - Semantic similarity for answers that may be paraphrased.
  - **When to use:** Answers may use different wording

- **LLM-as-Judge**
  - Use GPT-4 or Claude to judge answer correctness. Flexible but requires validation.
  - **When to use:** Complex, open-ended questions
  - **Note:** Validate against human judgments first

### **QA Metric Selection:**
- EM + F1: Standard for extractive QA (SQuAD)
- Add BERTScore for semantic equivalence
- LLM-judge for complex open-domain QA
- Always include human evaluation for high-stakes

---

### 4d. **Information Extraction Metrics**

- **Span-Level F1**
  - F1 computed over entity spans. Requires exact boundary match.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Named Entity Recognition
  - **Variants:** Strict (exact match), Partial (overlap credit)

- **Token-Level F1**
  - F1 over individual token labels (B-I-O scheme).
  - **When to use:** Token classification approach

- **Relation F1**
  - F1 over (head, relation, tail) triplets.
  - **When to use:** Knowledge graph extraction

- **JSON/Schema Accuracy**
  - Does the LLM output valid JSON matching expected schema?
  - **When to use:** Function calling, structured extraction
  - **Metrics:** Valid JSON rate, Schema compliance rate, Field accuracy

- **Key-Value Accuracy**
  - Per-field accuracy for document/form extraction.
  - **When to use:** Document understanding, form parsing

---

### 4e. **Ranking & Retrieval Metrics**

*Same as traditional IR metrics. Focus on top-K performance.*

- **NDCG@K (Normalized Discounted Cumulative Gain)**
  - Handles graded relevance, rewards top positions.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Have relevance grades (e.g., 0-3), position matters
  - **Note:** Industry standard for search

- **MRR (Mean Reciprocal Rank)**
  - Average of 1/rank of first relevant result.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Care most about first correct answer
  - **Formula:** `MRR = mean(1/rank_first_relevant)`

- **Precision@K**
  - Fraction of top-K results that are relevant.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** How many top results are good

- **Recall@K**
  - Fraction of all relevant items found in top-K.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Need to find most relevant items

- **MAP (Mean Average Precision)**
  - Mean of AP across queries. AP = average precision at each relevant item's position.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Binary relevance, order matters

- **Hit Rate@K**
  - Did at least one relevant item appear in top-K?
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Just need one good result

---

### 4f. **Reasoning & Math Metrics**

- **Accuracy (Final Answer)**
  - Is the final numerical/logical answer correct?
  - **When to use:** GSM8K, MATH, logic puzzles
  - **Note:** Extract final answer, compare to gold

- **Pass@K**
  - Probability of getting correct answer in K samples. Accounts for sampling variance.
  - **Range:** 0 to 1 (higher is better)
  - **Formula:** `Pass@K = 1 - C(n-c, k)/C(n, k)`
  - **When to use:** Using temperature sampling
  - **Note:** More robust than single-sample accuracy

- **Chain-of-Thought Validity**
  - Are the intermediate reasoning steps correct?
  - **When to use:** Care about reasoning process, not just answer
  - **Methods:** Human annotation, LLM-as-judge on steps, Formal verification

- **Self-Consistency**
  - Sample multiple CoT paths, majority vote. Measures reasoning stability.
  - **When to use:** Testing reasoning robustness
  - **Paper:** Self-Consistency Improves Chain of Thought Reasoning (2022)

- **MATH Difficulty Levels**
  - Report accuracy per difficulty level (1-5) on MATH benchmark.
  - **When to use:** Understanding capability curve

### **Key Reasoning Benchmarks:**
- GSM8K: Grade school math word problems
- MATH: Competition math (5 difficulty levels)
- ARC: Science reasoning
- HellaSwag: Commonsense reasoning
- MMLU: Multitask knowledge/reasoning

---

### 4g. **Code Generation Metrics**

- **Pass@K** ⭐ Primary Metric
  - Probability that at least one of K samples passes all unit tests.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Standard code generation evaluation
  - **Note:** HumanEval standard: Pass@1, Pass@10, Pass@100

- **Pass@1**
  - Probability of first sample being correct. Most practical metric.
  - **When to use:** Real-world single-query scenario
  - **Note:** Often use greedy decoding for Pass@1

- **HumanEval / HumanEval+**
  - 164 Python functions with docstrings and test cases. HumanEval+ adds more tests.
  - **Note:** De facto standard for code LLM evaluation

- **MBPP**
  - Mostly Basic Python Problems. 974 crowd-sourced problems.
  - **Note:** Broader coverage than HumanEval

- **CodeBLEU**
  - BLEU variant considering code syntax (AST matching, data flow).
  - **When to use:** Want to measure code similarity beyond text
  - **Components:** N-gram match, Weighted n-gram, Syntax match, Dataflow match

- **Execution-Based Metrics**
  - Run generated code, check outputs match expected.
  - **When to use:** Have test cases or expected outputs
  - **Note:** Most reliable but requires execution environment

- **SWE-Bench**
  - Resolve actual GitHub issues from popular repos. Tests real software engineering.
  - **Paper:** SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (2024)
  - **Note:** Most realistic but challenging benchmark

### **Code Evaluation Best Practices:**
- Always use execution-based testing when possible
- Report Pass@1 for practical performance
- Report Pass@10/100 to show potential with sampling
- SWE-Bench for real-world agent capability

---

## 5. **Human-like Quality & Preference Metrics**

### 5a. **How do you want to measure quality?**
   - **Human Evaluation** → Go to 5b
   - **LLM-as-Judge** → Go to 5c
   - **Preference & Comparison** → Go to 5d
   - **Benchmark Suites** → Go to 5e

---

### 5b. **Human Evaluation Dimensions**

- **Helpfulness** (Core Dimension)
  - Does the response actually help the user accomplish their goal?
  - **Scale:** Typically 1-5 or 1-7 Likert
  - **When to use:** Assistant/chatbot evaluation

- **Fluency** (Language Quality)
  - Is the text grammatically correct and natural sounding?
  - **Scale:** 1-5 Likert or binary
  - **Note:** Modern LLMs rarely fail here

- **Coherence** (Logical Flow)
  - Does the response make logical sense? Is it well-organized?
  - **When to use:** Long-form generation

- **Relevance** (On-Topic)
  - Does the response address what was asked?
  - **When to use:** Checking for off-topic drift

- **Factual Accuracy** (Truthfulness)
  - Are the claims in the response correct?
  - **When to use:** Factual/knowledge-intensive tasks
  - **Note:** Often requires fact-checking

- **Safety** (Harm Prevention)
  - Is the response free from harmful, biased, or inappropriate content?
  - **When to use:** Always for user-facing applications

- **A/B Preference** (Comparative)
  - Which of two responses is better? Side-by-side comparison.
  - **When to use:** Comparing two models/versions
  - **Note:** More reliable than absolute ratings

### **Human Evaluation Best Practices:**
- Use clear rubrics with examples
- Calculate inter-annotator agreement (Kappa, Krippendorff's alpha)
- Minimum 3 annotators per item
- Prefer pairwise comparison over absolute scores

---

### 5c. **LLM-as-Judge Methods**

*Scalable alternative to human evaluation. Correlates well with humans when done correctly.*

- **G-Eval**
  - Chain-of-thought evaluation with defined criteria. LLM outputs scores with reasoning.
  - **When to use:** Need interpretable automatic scores
  - **Paper:** G-Eval: NLG Evaluation using GPT-4 (2023)
  - **Pros:** Interpretable, customizable criteria
  - **Setup:** Prompt with evaluation steps, extract score

- **MT-Bench**
  - 80 multi-turn questions across 8 categories. GPT-4 judges responses 1-10.
  - **When to use:** Chat/assistant model comparison
  - **Paper:** Judging LLM-as-a-Judge (2023)
  - **Note:** Standard for instruction-tuned LLMs

- **AlpacaEval**
  - Compare model outputs to reference (GPT-4/Claude). Report win rate.
  - **Range:** 0-100% win rate
  - **When to use:** Quick model comparison
  - **Variants:** AlpacaEval 1.0, AlpacaEval 2.0 (length-controlled)
  - **Note:** Length-controlled version reduces length bias

- **Arena-Hard**
  - 500 hard prompts from Chatbot Arena. More discriminative than MT-Bench.
  - **When to use:** Distinguishing frontier models
  - **Paper:** From Crowdsourced Data to High-Quality Benchmarks (2024)

- **Pairwise Preference**
  - LLM chooses which of two responses is better. Can swap order to reduce bias.
  - **When to use:** Model comparison
  - **Note:** Average scores over position swaps

- **Rubric-Based Scoring**
  - Define custom rubric, LLM scores each dimension.
  - **When to use:** Domain-specific evaluation criteria
  - **Setup:** Clear rubric + examples in prompt

### **LLM-as-Judge Best Practices:**
- Use GPT-4 or Claude as judges (stronger = better)
- Swap response positions to reduce order bias
- Validate against human judgments on subset
- Be aware of self-preference bias
- Length bias: control for or use length-controlled variants

---

### 5d. **Preference & Comparison Metrics**

- **Elo Rating**
  - Chess-style rating from pairwise comparisons. Used by Chatbot Arena.
  - **When to use:** Ranking multiple models
  - **Pros:** Transitive rankings, handles varying opponents
  - **Note:** Chatbot Arena has 1M+ human votes

- **Win Rate**
  - Percentage of head-to-head wins against baseline.
  - **Range:** 0-100%
  - **When to use:** Comparing to specific baseline (e.g., GPT-4)
  - **Note:** Report ties separately

- **Bradley-Terry Model**
  - Probabilistic model for pairwise comparisons. Estimates win probabilities.
  - **When to use:** Statistical analysis of preferences

- **TrueSkill**
  - Bayesian rating system. Handles uncertainty in ratings.
  - **When to use:** Want confidence intervals on rankings

- **Human Preference Rate**
  - Percentage of humans preferring model A over B.
  - **When to use:** Ground truth for model comparison
  - **Note:** Expensive but most reliable

---

### 5e. **Standard Benchmark Suites**

- **MMLU**
  - Massive Multitask Language Understanding. 57 subjects from STEM to humanities.
  - **When to use:** General knowledge evaluation
  - **Note:** 5-shot, report overall and per-category

- **MMLU-Pro**
  - More challenging version with 10 choices and harder questions.
  - **Paper:** MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark (2024)

- **HellaSwag**
  - Commonsense reasoning about situations.
  - **Note:** Adversarially filtered for difficulty

- **ARC (AI2 Reasoning Challenge)**
  - Science exam questions. ARC-Easy and ARC-Challenge splits.
  - **When to use:** Scientific reasoning evaluation

- **WinoGrande**
  - Winograd-style commonsense pronoun resolution.
  - **When to use:** Commonsense coreference

- **TruthfulQA**
  - Questions that elicit common misconceptions. Tests for truthfulness.
  - **When to use:** Evaluating tendency to generate falsehoods
  - **Paper:** TruthfulQA: Measuring How Models Mimic Human Falsehoods (2021)

- **BigBench / BigBench-Hard**
  - 200+ diverse tasks. BBH subset focuses on hard reasoning.
  - **When to use:** Broad capability assessment

- **Open LLM Leaderboard**
  - HuggingFace leaderboard with standardized evaluations.
  - **Tasks:** MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K
  - **Note:** De facto standard for open model comparison

### **Benchmark Reporting Best Practices:**
- Always report evaluation settings (n-shot, prompt format)
- Use standardized evaluation harness (lm-eval)
- Report confidence intervals when possible
- Be aware of data contamination concerns

---

## 6. **Safety & Alignment Metrics**

### 6a. **What safety aspect are you evaluating?**
   - **Toxicity & Hate Speech** → Go to 6b
   - **Bias & Fairness** → Go to 6c
   - **Privacy & PII** → Go to 6d
   - **Policy Compliance** → Go to 6e

---

### 6b. **Toxicity & Harmful Content Metrics**

- **Toxicity Score (Perspective API)**
  - Google's classifier for toxic, severe toxic, insult, threat, etc.
  - **Range:** 0 to 1 per category
  - **When to use:** Standard toxicity measurement
  - **Pros:** Well-tested, multiple categories
  - **Tools:** Perspective API

- **RealToxicityPrompts**
  - Benchmark for measuring toxicity in continuations of prompts.
  - **When to use:** Testing toxicity in free generation
  - **Metrics:** Expected maximum toxicity, toxicity probability
  - **Paper:** RealToxicityPrompts (2020)

- **ToxiGen**
  - Benchmark for implicit and machine-generated toxic language.
  - **When to use:** Detecting subtle/implicit hate speech
  - **Paper:** ToxiGen: A Large-Scale Machine-Generated Dataset (2022)

- **HateBERT / Detoxify**
  - Fine-tuned BERT models for hate speech and toxicity.
  - **Tools:** Detoxify library, HateBERT

- **Toxicity Rate**
  - Percentage of outputs exceeding toxicity threshold.
  - **Formula:** `Rate = count(toxic > threshold) / total`
  - **When to use:** Summary statistic for evaluation

---

### 6c. **Bias & Fairness Metrics**

- **BBQ (Bias Benchmark for QA)**
  - Multiple-choice QA testing social biases across 9 categories.
  - **When to use:** Measuring social bias in question answering
  - **Paper:** BBQ: A Hand-Built Bias Benchmark (2022)

- **WinoBias / WinoGender**
  - Coreference resolution tests for gender bias.
  - **When to use:** Measuring gender stereotypes

- **StereoSet**
  - Measures stereotypical associations in language models.
  - **Metrics:** Language Modeling Score, Stereotype Score, ICAT combined
  - **Paper:** StereoSet: Measuring Stereotypical Bias (2021)

- **CrowS-Pairs**
  - Paired sentences measuring bias across demographics.
  - **When to use:** Comparative bias measurement

- **Demographic Parity**
  - Are outcomes equal across demographic groups?
  - **When to use:** Classification tasks with demographic info

- **Embedding Bias (WEAT/SEAT)**
  - Measures bias in embedding space using word/sentence associations.
  - **When to use:** Analyzing embedding-level bias

### **Bias Evaluation Considerations:**
- Bias is multi-dimensional - test multiple aspects
- Benchmarks may not cover all relevant biases
- Consider intersectionality
- Combine with qualitative analysis

---

### 6d. **Privacy & PII Metrics**

- **PII Detection Rate**
  - Rate of outputs containing PII (names, emails, SSNs, etc.).
  - **Formula:** `Rate = outputs_with_PII / total_outputs`
  - **When to use:** Measuring memorization of private data
  - **Tools:** Presidio, spaCy NER, regex patterns

- **Membership Inference**
  - Can an adversary determine if data was in training set?
  - **When to use:** Testing memorization vulnerability
  - **Metrics:** Attack success rate, AUC of attack

- **Extraction Attack Success**
  - Can adversary extract specific private information from prompts?
  - **When to use:** Testing prompt injection / extraction attacks

- **Canary Extraction**
  - Insert canary strings in training, test if model memorizes them.
  - **Paper:** Extracting Training Data from Large Language Models (2021)

- **k-Anonymity Preservation**
  - Does generated data maintain k-anonymity properties?
  - **When to use:** Generating synthetic data

---

### 6e. **Policy Compliance & Red-Teaming Metrics**

- **Attack Success Rate (ASR)**
  - Percentage of adversarial prompts that bypass safety.
  - **Range:** 0 to 1 (lower is safer)
  - **When to use:** Red-teaming, jailbreak testing
  - **Note:** Test with diverse attack types

- **Refusal Rate**
  - Percentage of harmful requests properly refused.
  - **Range:** 0 to 1 (higher = safer, but watch for over-refusal)
  - **When to use:** Testing safety policy adherence

- **False Refusal Rate**
  - Percentage of benign requests incorrectly refused.
  - **Range:** 0 to 1 (lower is better)
  - **When to use:** Checking for over-cautious behavior
  - **Note:** Balance safety with helpfulness

- **HarmBench**
  - Benchmark for evaluating LLM safety against harmful requests.
  - **Paper:** HarmBench: A Standardized Evaluation Framework (2024)
  - **Categories:** Chemical/bio, Cybersecurity, Harassment, Misinformation

- **JailbreakBench**
  - Standardized jailbreak attack evaluation.
  - **When to use:** Testing adversarial prompt resistance

- **StrongREJECT**
  - Evaluates quality and appropriateness of refusal responses.
  - **Paper:** StrongREJECT (2024)

### **Safety Evaluation Principles:**
- Red-team with diverse attack types
- Balance refusal rate with false refusal rate
- Test against latest jailbreak techniques
- Include human red-teaming, not just automated

---

## 7. **Factuality & Hallucination Metrics**

### 7a. **What type of factuality evaluation?**
   - **RAG / Grounded Generation** → Go to 7b
   - **Open-Domain Factuality** → Go to 7c
   - **Hallucination Detection** → Go to 7d

---

### 7b. **RAG Evaluation Metrics (RAGAS Framework)**

*RAGAS provides comprehensive RAG evaluation.*

- **Faithfulness** ⭐ Core RAG Metric
  - Are all claims in the answer supported by the retrieved context?
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Checking for context-grounded answers
  - **Method:** LLM extracts claims, checks each against context
  - **Tools:** RAGAS, TruLens

- **Answer Relevance** ⭐ Core RAG Metric
  - Is the answer relevant to the question asked?
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Checking answer addresses the question
  - **Method:** Generate questions from answer, compare to original

- **Context Precision**
  - Are the retrieved contexts relevant to the question?
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Evaluating retriever performance
  - **Note:** Ranks relevant contexts higher

- **Context Recall**
  - Does the context contain information needed to answer?
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** Checking if retrieval found necessary info
  - **Note:** Requires ground truth answer

- **Answer Correctness**
  - Overall correctness combining factuality and semantic similarity.
  - **Range:** 0 to 1 (higher is better)
  - **When to use:** End-to-end RAG evaluation

- **Groundedness (TruLens)**
  - Similar to faithfulness. Measures answer support by context.
  - **Tools:** TruLens, LangChain evaluation

- **ARES**
  - Automated RAG Evaluation System. Trains custom evaluators.
  - **Paper:** ARES: An Automated Evaluation Framework (2023)
  - **Pros:** Domain-adaptive, lightweight after training

### **RAG Evaluation Best Practices:**
- Evaluate retrieval and generation separately
- Faithfulness is often more important than relevance
- Use multiple metrics for complete picture
- Validate LLM-judges against human labels

---

### 7c. **Open-Domain Factuality Metrics**

- **FActScore** ⭐ Primary Metric
  - Breaks text into atomic facts, verifies each against knowledge source.
  - **Range:** 0 to 1 (fraction of supported facts)
  - **Method:** Extract atomic claims -> retrieve evidence -> verify each
  - **Paper:** FActScore: Fine-grained Atomic Evaluation (2023)
  - **Pros:** Interpretable, fine-grained
  - **Cons:** Computationally expensive

- **TruthfulQA**
  - 817 questions designed to elicit false answers based on misconceptions.
  - **Range:** 0 to 1 accuracy
  - **When to use:** Testing tendency to generate common falsehoods
  - **Metrics:** Truthful %, Informative %, Truthful + Informative %
  - **Paper:** TruthfulQA: Measuring How Models Mimic Human Falsehoods (2021)

- **FACTOR**
  - Factuality benchmark for news generation.
  - **Paper:** Measuring and Improving Factuality in News (2023)

- **Knowledge F1**
  - Verify claims against knowledge base. F1 over supported/unsupported.
  - **When to use:** Have structured knowledge base

- **Source Attribution**
  - Does the model provide accurate citations/sources?
  - **Metrics:** Citation precision, citation recall, URL validity
  - **When to use:** Evaluating source-citing models

### **Factuality Challenges:**
- Knowledge cutoff affects evaluation
- Some facts are disputed or context-dependent
- Consider confidence calibration
- Long-tail facts are harder to verify

---

### 7d. **Hallucination Detection Metrics**

- **SelfCheckGPT** ⭐ Reference-Free
  - Sample multiple responses, check consistency. Hallucinations are inconsistent.
  - **Range:** Higher score = more hallucination
  - **Method:** Multiple samples -> measure agreement
  - **Paper:** SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection (2023)
  - **Pros:** No external knowledge needed, works with any LLM

- **Hallucination Rate**
  - Percentage of outputs containing hallucinated content.
  - **Formula:** `Rate = hallucinated_outputs / total_outputs`
  - **When to use:** Aggregate evaluation
  - **Note:** Requires ground truth or human annotation

- **HaluEval**
  - Large-scale hallucination evaluation benchmark with 35K samples.
  - **Categories:** QA hallucination, Dialogue hallucination, Summarization hallucination
  - **Paper:** HaluEval: A Large-Scale Hallucination Evaluation Benchmark (2023)

- **FAVA**
  - Fine-grained hallucination detection with error type classification.
  - **Types:** Entity error, Relation error, Contradictory, Invented, Subjective
  - **Paper:** FAVA: Fine-grained Hallucination Evaluation (2024)

- **NLI-Based Detection**
  - Use NLI model to check if claims are entailed by source.
  - **Method:** Classify (source, claim) as entailed/contradicted/neutral
  - **Tools:** TRUE benchmark, SummaC

- **Atomic Claim Verification**
  - Break into atomic claims, verify each independently.
  - **Method:** Decompose -> Retrieve evidence -> Classify each claim
  - **Pros:** Interpretable, localized
  - **Cons:** Decomposition errors propagate

### **Hallucination Types:**
- **Intrinsic:** Contradicts source/context
- **Extrinsic:** Adds unsupported information
- **Entity errors:** Wrong names, dates, numbers
- **Fabrication:** Completely invented content

---

## Quick Reference Summary

| **Evaluation Goal** | **Primary Metrics** | **Key Considerations** |
|---------------------|---------------------|------------------------|
| **Language Modeling** | Perplexity, Cross-Entropy | Lower is better, not sufficient alone |
| **Summarization** | ROUGE-1/2/L, BERTScore | Add SummaC for faithfulness |
| **Translation** | BLEU, COMET | COMET has best human correlation |
| **Semantic Similarity** | BERTScore, Sentence Transformers | Handles paraphrases |
| **Classification** | Macro F1, Accuracy (if balanced) | Use F1 for imbalanced |
| **Question Answering** | EM, F1 (token-level) | Add BERTScore for semantic |
| **Code Generation** | Pass@K, HumanEval | Execution-based is gold standard |
| **Reasoning/Math** | Accuracy, Pass@K | GSM8K, MATH benchmarks |
| **Assistant Quality** | MT-Bench, AlpacaEval, Arena Elo | LLM-as-judge or human |
| **Toxicity** | Perspective API, RealToxicityPrompts | Report toxicity rate |
| **Bias** | BBQ, WinoBias, StereoSet | Test multiple dimensions |
| **RAG Faithfulness** | RAGAS (Faithfulness, Relevance) | Separate retrieval & generation |
| **Factuality** | FActScore, TruthfulQA | Atomic fact verification |
| **Hallucination** | SelfCheckGPT, HaluEval | Consistency-based or NLI |

---

## Key Principles

1. **Use multiple metrics** for comprehensive evaluation
2. **Perplexity alone is NOT sufficient** for user-facing quality
3. **LLM-as-judge works well** but validate against humans
4. **For safety:** test with diverse red-team attacks
5. **For RAG:** evaluate retrieval and generation separately

---

## Popular Tools

| **Tool** | **Purpose** |
|----------|-------------|
| **lm-evaluation-harness** | Standard benchmark suite |
| **RAGAS** | RAG evaluation framework |
| **TruLens** | LLM app evaluation |
| **HuggingFace Evaluate** | Metrics library |
| **Perspective API** | Toxicity detection |
| **bert-score** | Semantic similarity |
| **SacreBLEU** | Standardized BLEU |

---

**Version 1.0 - Based on Latest Research (2024)**
*Comprehensive guide to evaluating Foundation Models and LLMs*
