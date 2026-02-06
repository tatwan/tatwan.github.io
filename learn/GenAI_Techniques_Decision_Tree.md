### GenAI Techniques Decision Tree (v1.0)

## 1. **What is your primary goal with your LLM?**
   - **Ground in External Knowledge** (Add up-to-date, private, or domain-specific knowledge without changing model weights) → Go to 2
   - **Change Model Behavior** (Modify how the model responds, follows instructions, or performs specific tasks) → Go to 3
   - **Optimize Size & Speed** (Make the model smaller, faster, or cheaper to run) → Go to 4
   - **Expand General Knowledge** (Teach the model new domains or update its world knowledge) → Go to 5

---

## 2. **Grounding with External Knowledge**

### 2a. **What type of knowledge do you need to add?**
   - **Dynamic/Changing Knowledge** (Documents, APIs, databases that update frequently) → Go to 2b (RAG)
   - **Static Domain Knowledge** (Stable, well-defined knowledge that rarely changes) → Go to 2c
   - **Need Both** (Combination of dynamic retrieval and learned behavior) → Go to 2d (RAG + Fine-tuning)

---

### 2b. **Retrieval-Augmented Generation (RAG)**

*Goal: Ground model responses in external, verifiable knowledge*

#### **Core RAG Techniques:**

- **Basic RAG** ⭐ Start Here
  - Retrieve relevant documents and include them in the prompt context. Model generates response based on retrieved content.
  - **When to use:** Knowledge changes frequently, need citations, data is private
  - **Components:** Vector Database → Embedding Model → Retriever → LLM
  - **Pros:** No retraining needed, easy to update knowledge, provides citations, keeps base model unchanged
  - **Cons:** Limited by context window, retrieval quality affects output, added latency
  - **Tools:** LangChain, LlamaIndex, Pinecone, Weaviate, ChromaDB

- **Hybrid Search RAG**
  - Combines dense vector search with sparse keyword search (BM25) for better recall.
  - **When to use:** Need both semantic and keyword matching
  - **Components:** Dense Embeddings + Sparse Index (BM25) + Reranker
  - **Pros:** Better retrieval accuracy, handles exact matches + semantic
  - **Cons:** More complex pipeline, requires tuning
  - **Tools:** Elasticsearch, Qdrant, Cohere Rerank

- **Agentic RAG**
  - LLM decides what to retrieve, when, and how to use multiple sources. Can query APIs, databases, and tools.
  - **When to use:** Complex queries requiring multi-step reasoning
  - **Components:** Agent Framework + Tool Calling + Memory
  - **Pros:** Handles complex queries, multi-source integration, adaptive retrieval
  - **Cons:** Higher complexity, more expensive, harder to debug
  - **Tools:** LangGraph, AutoGen, CrewAI

### **RAG Optimization Techniques:**

- **Chunk Optimization**
  - Optimize how documents are split. Consider semantic chunking, overlap, and hierarchy.
  - **When to use:** Poor retrieval relevance, missing context
  - **Tips:** Use 256-512 token chunks, add 10-20% overlap, consider document structure
  - **Tools:** LlamaIndex, Unstructured.io

- **Reranking** ⭐ High Impact
  - Use a cross-encoder to rerank retrieved documents by relevance before passing to LLM.
  - **When to use:** Initial retrieval returns relevant docs but wrong order
  - **Pros:** Significantly improves precision, works with any retriever
  - **Cons:** Adds latency, additional model cost
  - **Tools:** Cohere Rerank, BGE Reranker, ColBERT

- **Query Transformation**
  - Rewrite, expand, or decompose user queries for better retrieval.
  - **When to use:** User queries are ambiguous or complex
  - **Methods:** HyDE (Hypothetical Document), Query Expansion, Multi-Query
  - **Tools:** LangChain Query Transformers

- **Fine-tuned Embeddings**
  - Train custom embedding model on your domain data for better semantic matching.
  - **When to use:** Generic embeddings miss domain terminology
  - **Pros:** Better domain understanding, improved recall
  - **Cons:** Requires training data, maintenance overhead
  - **Tools:** Sentence Transformers, OpenAI fine-tuning

---

### 2c. **Options for Static Knowledge**

*For stable knowledge, you can either retrieve at runtime or embed during training.*

- **Still prefer retrieval** → Use RAG (Go to 2b)
  - Keep knowledge separate, easier updates

- **Embed in model weights** → Use Knowledge Fine-tuning (below)
  - Bake knowledge into the model via training

#### **Knowledge Fine-tuning:**

*Goal: Embed domain knowledge directly into model weights*

> **Warning:** Consider if RAG might be simpler. Fine-tuning for knowledge can be expensive and hard to update.

- **Domain-Specific SFT**
  - Fine-tune on Q&A pairs or documents from your domain. Model learns to generate domain-accurate responses.
  - **When to use:** Stable domain, need faster inference than RAG
  - **Data needed:** 1K-100K high-quality examples
  - **Pros:** No retrieval latency, consistent behavior
  - **Cons:** Hard to update, risk of hallucination, expensive
  - **Tools:** HuggingFace TRL, Axolotl, OpenAI Fine-tuning

- **Continued Pre-training + SFT**
  - First continue pre-training on domain text, then fine-tune for tasks.
  - **When to use:** Need deep domain understanding (legal, medical, scientific)
  - **Data needed:** Large domain corpus (GB of text)
  - **Pros:** Best domain adaptation, learns terminology and patterns
  - **Cons:** Most expensive, requires significant compute
  - **Note:** Consider this a two-phase approach: CPT then SFT

---

### 2d. **RAG + Fine-tuning Combination**

*Goal: Combine retrieval with behavioral fine-tuning for best results*

Use when RAG responses are correct but off-tone, or model ignores retrieved context.

- **RAG-Aware SFT** ⭐ Best Practice
  - Fine-tune on examples where model must read and correctly use retrieved context. Teaches model to ground responses.
  - **When to use:** Model ignores or misuses retrieved documents
  - **Data needed:** Examples with context + ideal responses
  - **Pros:** Better context utilization, maintains RAG benefits
  - **Cons:** Requires curated training data
  - **Setup:** Create synthetic data with retrieved context

- **RAFT (Retrieval Augmented Fine-Tuning)**
  - Train model to distinguish relevant from distractor documents and extract answers.
  - **When to use:** Need robust grounding behavior
  - **Paper:** RAFT: Adapting Language Model to Domain Specific RAG (2024)
  - **Pros:** Improved retrieval robustness, better answer extraction
  - **Note:** Mix of oracle and distractor documents during training

---

## 3. **Changing Model Behavior**

### 3a. **What aspect of behavior do you want to change?**
   - **Quick adjustments (no training)** → Go to 3b (Prompting)
   - **Task-specific behavior** → Go to 3c (Fine-tuning Methods)
   - **Alignment & Safety** → Go to 3d (Alignment Techniques)
   - **Tool/Function Calling** → Go to 3e

---

### 3b. **Prompting Techniques (No Training Required)**

*Goal: Modify behavior through prompt design - fastest and cheapest approach*

- **Zero-Shot Prompting**
  - Direct instruction without examples. Works well for capable models on common tasks.
  - **When to use:** Simple, well-defined tasks; strong base model
  - **Example:** `Classify this text as positive or negative: {text}`
  - **Pros:** No examples needed, fast iteration
  - **Cons:** Less reliable for complex tasks, model-dependent

- **Few-Shot Prompting** ⭐ Most Common
  - Include 2-8 examples demonstrating the desired input-output pattern.
  - **When to use:** Need consistent format, complex reasoning, or domain-specific behavior
  - **Example:** `Example 1: ... Example 2: ... Now do: {input}`
  - **Pros:** More reliable, format consistency, no training
  - **Cons:** Uses context window, example selection matters

- **Chain-of-Thought (CoT)**
  - Prompt model to show step-by-step reasoning before final answer.
  - **When to use:** Math, logic, multi-step reasoning tasks
  - **Trigger:** Add "Let's think step by step" or show reasoning examples
  - **Pros:** Better accuracy on reasoning, interpretable
  - **Cons:** More tokens, not always needed

- **System Prompts**
  - Set model personality, constraints, and behavior rules in system message.
  - **When to use:** Need consistent persona, safety guardrails, output format
  - **Tips:** Be specific about constraints, define edge cases, test adversarial inputs
  - **Note:** All chat APIs support system prompts

- **Structured Output**
  - Constrain model to output valid JSON, XML, or other formats.
  - **When to use:** Need machine-parseable output
  - **Methods:** JSON mode, Function calling, Grammar constraints
  - **Tools:** OpenAI JSON mode, Outlines, Guidance, LMQL

### **If prompting is not enough** → Consider Prompt Tuning or Fine-tuning

---

### 3b-ii. **Soft Prompting / Prompt Tuning**

*Goal: Learn optimal prompt embeddings without changing model weights*

Middle ground between prompting and fine-tuning. Learns continuous prompt vectors.

- **Prompt Tuning**
  - Learn soft prompt tokens prepended to input. Model weights stay frozen.
  - **When to use:** Want task adaptation without full fine-tuning
  - **Parameters:** ~20K parameters (just prompt embeddings)
  - **Pros:** Very parameter-efficient, multi-task via different prompts, no catastrophic forgetting
  - **Cons:** Less powerful than fine-tuning, requires training infrastructure
  - **Paper:** The Power of Scale for Parameter-Efficient Prompt Tuning (2021)

- **Prefix Tuning**
  - Learn continuous prefixes for each transformer layer, not just input.
  - **When to use:** Need more adaptation capacity than prompt tuning
  - **Parameters:** ~0.1% of model parameters
  - **Pros:** More expressive than prompt tuning, still very efficient
  - **Cons:** Slightly more complex
  - **Paper:** Prefix-Tuning: Optimizing Continuous Prompts (2021)

- **P-Tuning v2**
  - Adds learnable prompts to every layer. Matches fine-tuning on many tasks.
  - **When to use:** Need fine-tuning-level performance with efficiency
  - **Pros:** Competitive with fine-tuning, works across model sizes
  - **Tools:** PEFT library

---

### 3c. **Fine-tuning Methods**

### **What's your compute budget?**
   - **Limited / Large Model** → Use PEFT (LoRA, QLoRA, etc.) - most common choice
   - **Sufficient Compute** → Can afford Full Fine-tuning
   - **Using API-only Model** → Use Provider Fine-tuning APIs

---

#### **Parameter-Efficient Fine-Tuning (PEFT)**

*Goal: Adapt model behavior while training only a small fraction of parameters*

- **LoRA (Low-Rank Adaptation)** ⭐ Most Popular
  - Add small trainable low-rank matrices to attention layers. Merge for inference.
  - **When to use:** Most fine-tuning scenarios, especially 7B+ models
  - **Parameters:** 0.1-1% of model parameters
  - **Pros:** Very effective, easy to swap adapters, can merge into base model, well-supported
  - **Cons:** Rank selection matters, may need tuning
  - **Settings:** Typical: rank=8-64, alpha=16-128, target query/value projections
  - **Tools:** HuggingFace PEFT, Axolotl, LLaMA-Factory

- **QLoRA** ⭐ Memory-Efficient
  - Quantize base model to 4-bit, train LoRA adapters in full precision.
  - **When to use:** Limited GPU memory, fine-tuning 13B+ on consumer hardware
  - **Requirements:** Single 24GB GPU can fine-tune 33B model
  - **Pros:** Dramatic memory reduction, enables large model fine-tuning, minimal quality loss
  - **Cons:** Slower training, quantization artifacts possible
  - **Paper:** QLoRA: Efficient Finetuning of Quantized LLMs (2023)
  - **Tools:** bitsandbytes + PEFT, Axolotl

- **DoRA (Weight-Decomposed LRA)**
  - Decomposes weights into magnitude and direction, applies LoRA to direction only.
  - **When to use:** Want to improve on LoRA performance
  - **Pros:** Often outperforms LoRA, same efficiency
  - **Cons:** Newer, less tooling
  - **Paper:** DoRA: Weight-Decomposed Low-Rank Adaptation (2024)

- **IA3 (Infused Adapter)**
  - Learn rescaling vectors for keys, values, and FFN. Even fewer parameters than LoRA.
  - **When to use:** Extreme parameter efficiency needed
  - **Parameters:** ~0.01% of model parameters
  - **Pros:** Most parameter-efficient, fast
  - **Cons:** Less expressive than LoRA
  - **Paper:** Few-Shot Parameter-Efficient Fine-Tuning (2022)

- **Adapter Layers**
  - Insert small bottleneck layers between transformer blocks.
  - **When to use:** Multi-task scenarios, modular adaptation
  - **Pros:** Modular, easy to add/remove
  - **Cons:** Adds inference latency, less popular now
  - **Tools:** AdapterHub

### **Choosing PEFT Method:**
| Method | Parameters | Best For |
|--------|------------|----------|
| LoRA | 0.1-1% | Default choice, best balance |
| QLoRA | 0.1-1% | Memory-constrained setups |
| DoRA | 0.1-1% | Squeeze more quality |
| IA3 | ~0.01% | Extreme efficiency, simpler tasks |

---

#### **Full Fine-Tuning**

*Goal: Update all model parameters for maximum adaptation*

> **Warning:** Requires significant compute. Consider if PEFT would suffice first.

- **Supervised Fine-Tuning (SFT)**
  - Train on (instruction, response) pairs. Updates all weights.
  - **When to use:** Need maximum adaptation, have sufficient compute
  - **Requirements:** Multiple high-end GPUs, distributed training
  - **Pros:** Maximum flexibility, best potential performance
  - **Cons:** Expensive, risk of catastrophic forgetting, needs careful LR
  - **Tools:** HuggingFace Trainer, DeepSpeed, FSDP

- **Full Fine-Tuning + RLHF**
  - SFT followed by reinforcement learning from human feedback.
  - **When to use:** Training instruction-following or chat models from base
  - **Stages:** SFT on demonstrations → Reward model training → PPO optimization
  - **Pros:** Best alignment quality, how ChatGPT was trained
  - **Cons:** Complex pipeline, expensive, reward hacking risks
  - **Tools:** TRL, OpenRLHF

---

#### **API-Based Fine-Tuning**

*Goal: Fine-tune through provider APIs without managing infrastructure*

- **OpenAI Fine-Tuning**
  - Upload JSONL data, provider handles training. Get custom model endpoint.
  - **When to use:** Using OpenAI models, want simplicity
  - **Models:** GPT-4o, GPT-4o-mini, GPT-3.5-Turbo
  - **Pros:** No infrastructure, simple API, auto-scaling
  - **Cons:** Limited control, data goes to provider, ongoing costs
  - **Data format:** JSONL with messages array

- **Claude Fine-Tuning**
  - Available for enterprise customers. Contact Anthropic.
  - **When to use:** Using Claude at scale, need customization

- **Together.ai / Anyscale**
  - Fine-tune open-source models through managed APIs.
  - **When to use:** Want open model flexibility with managed infrastructure
  - **Pros:** More model choices, reasonable pricing
  - **Tools:** Together Fine-tuning, Anyscale Endpoints

---

### 3d. **Alignment & Safety Fine-Tuning**

*Goal: Improve helpfulness, harmlessness, and policy compliance*

- **RLHF (RL from Human Feedback)** ⭐ Gold Standard
  - Train reward model on human preferences, then optimize policy with PPO.
  - **When to use:** Need high-quality alignment, have resources for full pipeline
  - **Stages:** Collect comparison data → Train reward model → PPO training
  - **Pros:** Best alignment quality, proven at scale
  - **Cons:** Complex, expensive, reward hacking
  - **Tools:** TRL, OpenRLHF, DeepSpeed-Chat

- **DPO (Direct Preference Optimization)** ⭐ Recommended Alternative
  - Skip reward model. Directly optimize policy from preference pairs.
  - **When to use:** Want RLHF-like results with simpler pipeline
  - **Data needed:** (prompt, chosen, rejected) triplets
  - **Pros:** Simpler than RLHF, no reward model needed, stable training
  - **Cons:** May underperform RLHF on some tasks
  - **Paper:** Direct Preference Optimization (2023)
  - **Tools:** TRL DPOTrainer, Axolotl

- **ORPO (Odds Ratio Preference Optimization)**
  - Combines SFT and preference alignment in one stage.
  - **When to use:** Want single-stage alignment training
  - **Pros:** Simpler pipeline, efficient
  - **Paper:** ORPO: Monolithic Preference Optimization (2024)

- **Constitutional AI (RLAIF)**
  - Use AI feedback based on principles instead of human labels.
  - **When to use:** Need scalable alignment, can't collect enough human feedback
  - **Pros:** More scalable, consistent feedback
  - **Cons:** Depends on principle quality
  - **Paper:** Constitutional AI (Anthropic, 2022)

- **Safety Fine-Tuning**
  - Train on refusal examples and safe completions for harmful prompts.
  - **When to use:** Need to add safety guardrails to open models
  - **Data needed:** Harmful prompt + safe refusal pairs
  - **Note:** Combine with system prompts for layered safety

### **Alignment Method Selection:**
- Start with DPO - simpler and often sufficient
- Use RLHF for maximum alignment quality at scale
- ORPO for efficient single-stage training
- RLAIF when human feedback is limited

---

### 3e. **Tool Use & Function Calling**

*Goal: Enable model to use external tools, APIs, and functions*

- **Function Calling Fine-Tuning**
  - Fine-tune on examples of when and how to call functions with correct arguments.
  - **When to use:** Need reliable tool use, custom tools
  - **Data needed:** Conversations with function calls and results
  - **Pros:** Reliable function calls, custom tool support
  - **Tools:** Glaive function calling dataset, custom data

- **Tool-Augmented Models**
  - Use models already trained for tool use (GPT-4, Claude, Llama 3).
  - **When to use:** Standard tools, don't want to fine-tune
  - **Pros:** Works out of box, well-tested
  - **Cons:** Less customization

- **ReAct Pattern**
  - Prompt model to Reason and Act in alternating steps with observations.
  - **When to use:** Multi-step tool use, need reasoning traces
  - **Pattern:** Thought → Action → Observation → Thought → ...
  - **Tools:** LangChain Agents, LlamaIndex

---

### 3f. **Training Data Best Practices**

- **Data Quality > Quantity**
  - A few thousand high-quality examples often outperforms millions of low-quality ones.
  - **Tips:** Curate carefully, remove duplicates, verify correctness, diverse examples
  - **Recommendation:** Start with 1K-10K high-quality examples

- **Instruction Format**
  - Use consistent format matching your inference setup.
  - **Formats:** Alpaca (instruction/input/output), ChatML (messages array), Llama (special tokens)
  - **Tip:** Match your training format to inference format

- **Synthetic Data Generation**
  - Use strong models (GPT-4, Claude) to generate training examples.
  - **When to use:** Need more data, want to distill capabilities
  - **Methods:** Self-Instruct, Evol-Instruct, WizardLM approach
  - **Warning:** Verify quality, watch for model artifacts

- **Data Mixing**
  - Mix task-specific data with general instruction data to prevent forgetting.
  - **Ratio:** Typically 20-50% general data
  - **Tools:** Dolly, OpenAssistant datasets for mixing

---

### 3g. **Efficient Training Techniques**

- **Gradient Checkpointing**
  - Trade compute for memory by recomputing activations during backward pass.
  - **Savings:** ~60% memory reduction
  - **Cost:** ~20% slower training
  - **Tools:** Built into HuggingFace Trainer

- **Mixed Precision (FP16/BF16)**
  - Use half-precision for most operations, full precision for critical parts.
  - **Savings:** ~50% memory, faster training
  - **Recommendation:** BF16 preferred on newer GPUs
  - **Tools:** PyTorch AMP, Accelerate

- **DeepSpeed ZeRO**
  - Shard optimizer states, gradients, and parameters across GPUs.
  - **Stages:** ZeRO-1 (Optimizer states), ZeRO-2 (+ Gradients), ZeRO-3 (+ Parameters)
  - **When to use:** Training on multiple GPUs
  - **Tools:** DeepSpeed, HuggingFace integration

- **FSDP**
  - Fully Sharded Data Parallel - PyTorch's native model parallelism.
  - **When to use:** Multi-GPU, prefer native PyTorch
  - **Tools:** PyTorch FSDP, Accelerate

- **Flash Attention** ⭐ Always Use
  - Memory-efficient attention implementation. Faster and uses less memory.
  - **Savings:** 5-20x memory for long sequences
  - **Recommendation:** Enable by default
  - **Tools:** Flash Attention 2, xFormers

---

## 4. **Optimizing Size & Speed (Compression)**

### 4a. **What's your primary optimization goal?**
   - **Reduce Memory/Latency** → Go to 4b (Quantization) - Most common
   - **Reduce Parameters** → Go to 4c (Pruning)
   - **Smaller Model Architecture** → Go to 4d (Distillation)
   - **Maximum Compression** → Go to 4e (Combined Pipeline)

---

### 4b. **Quantization Techniques**

*Goal: Reduce precision of model weights and activations*

Quantization offers the best compression-to-quality ratio and is usually the first step.

- **GPTQ**
  - One-shot weight quantization using calibration data. Widely supported.
  - **Precision:** 4-bit, 8-bit
  - **Pros:** Good quality, fast inference, wide support
  - **Cons:** Requires calibration data, static quantization
  - **Tools:** AutoGPTQ, ExLlama, transformers

- **AWQ (Activation-Aware)** ⭐ High Quality
  - Protects salient weights based on activation magnitudes.
  - **Precision:** 4-bit
  - **Pros:** Often better quality than GPTQ, fast
  - **Cons:** Newer, less universal support
  - **Paper:** AWQ: Activation-aware Weight Quantization (2023)
  - **Tools:** AutoAWQ, vLLM

- **GGUF/GGML**
  - Format optimized for llama.cpp CPU inference. Various quantization levels.
  - **Precision:** 2-8 bit options (Q2_K to Q8_0)
  - **Pros:** CPU friendly, flexible precision, large ecosystem
  - **Cons:** Primarily for llama.cpp
  - **Tools:** llama.cpp, Ollama, LM Studio

- **bitsandbytes (bnb)**
  - 8-bit and 4-bit quantization with HuggingFace integration.
  - **Precision:** 4-bit (NF4), 8-bit (LLM.int8)
  - **Pros:** Easy HF integration, works with training (QLoRA)
  - **Cons:** Primarily NVIDIA
  - **Tools:** bitsandbytes, transformers

- **FP8**
  - 8-bit floating point on modern hardware (H100, RTX 4090).
  - **Precision:** 8-bit floating point
  - **Pros:** Good quality, hardware accelerated
  - **Cons:** Requires new hardware
  - **Tools:** TensorRT-LLM, vLLM

### **Quantization Selection Guide:**
| Precision | Memory Reduction | Quality Impact |
|-----------|------------------|----------------|
| 8-bit | ~50% | Minimal |
| 4-bit (GPTQ/AWQ) | ~75% | Small |
| 2-3 bit | ~85% | Noticeable |

**Recommendation:** Start with AWQ or GPTQ 4-bit for most use cases.

---

### 4c. **Pruning Techniques**

*Goal: Remove unnecessary weights or structures from the model*

- **Unstructured Pruning**
  - Zero out individual weights based on magnitude or importance.
  - **Sparsity:** Up to 50-90% sparsity possible
  - **Pros:** High compression potential
  - **Cons:** Needs sparse hardware/software for speedup, quality degrades at high sparsity
  - **Tools:** SparseML, PyTorch pruning

- **Structured Pruning**
  - Remove entire neurons, heads, or layers. Creates smaller dense model.
  - **Targets:** Attention heads, FFN neurons, layers
  - **Pros:** Actual speedup on standard hardware, easier deployment
  - **Cons:** Less compression than unstructured, may need retraining
  - **Tools:** LLM-Pruner, Wanda

- **Wanda (Pruning by Weights and Activations)** ⭐ SotA for LLMs
  - Prune based on weight magnitude × input activation. No retraining needed.
  - **Pros:** One-shot pruning, good quality retention
  - **Sparsity:** 50% sparsity with minimal degradation
  - **Paper:** A Simple and Effective Pruning Approach for LLMs (2023)

- **SparseGPT**
  - One-shot pruning using approximate second-order information.
  - **Pros:** No retraining, works on large models
  - **Sparsity:** 50-60% unstructured sparsity
  - **Paper:** SparseGPT (2023)

---

### 4d. **Knowledge Distillation**

*Goal: Train a smaller student model to mimic a larger teacher*

- **Response-Based Distillation** ⭐ Most Common
  - Train student on teacher's output text/logits. Student learns to match teacher responses.
  - **When to use:** Want smaller model with similar capabilities
  - **Data needed:** Large dataset of (input, teacher_output) pairs
  - **Pros:** Simple to implement, flexible student architecture
  - **Cons:** Requires generating teacher outputs, may miss nuanced behavior
  - **Examples:** Alpaca (from text-davinci-003), Vicuna, Orca

- **Feature-Based Distillation**
  - Match intermediate representations between teacher and student.
  - **When to use:** Need more faithful capability transfer
  - **Pros:** Better capability transfer
  - **Cons:** Requires compatible architectures, more complex
  - **Example:** DistilBERT approach

- **On-Policy Distillation**
  - Student generates, teacher provides feedback. Iterative improvement.
  - **When to use:** Maximum quality transfer needed
  - **Pros:** Addresses distribution mismatch, better generalization
  - **Cons:** Expensive, complex pipeline
  - **Paper:** On-Policy Distillation of Language Models (2023)

- **Synthetic Data Distillation**
  - Generate diverse training data using teacher, train student on it.
  - **When to use:** Need to scale data for distillation
  - **Methods:** Self-Instruct, Evol-Instruct, GLAN
  - **Examples:** WizardLM, OpenHermes

### **Distillation Strategies:**
- Start with response-based for simplicity
- Use diverse prompts for teacher generation
- Larger student = easier distillation
- Consider combining with PEFT for efficiency

---

### 4e. **Combined Compression Pipeline**

*Goal: Maximize compression by combining multiple techniques*

Research suggests optimal ordering: **Pruning → Distillation → Quantization**

- **Pruning → Quantization**
  - First prune to remove weights, then quantize remaining weights.
  - **Order:** 1. Prune → 2. Fine-tune to recover quality → 3. Quantize
  - **Pros:** Good compression, relatively simple
  - **Note:** Fine-tuning between steps helps recovery

- **Distillation → Quantization**
  - Distill to smaller architecture, then quantize the student.
  - **Order:** 1. Distill to smaller student → 2. Quantize student
  - **Pros:** Architectural reduction + precision reduction
  - **Example:** TinyLlama → 4-bit quantization

- **Full Pipeline (P-KD-Q)** ⭐ Maximum Compression
  - Pruning, Knowledge Distillation, Quantization in sequence.
  - **Order:** 1. Prune teacher → 2. Distill to student → 3. Quantize student
  - **Pros:** Highest compression ratios
  - **Cons:** Complex pipeline, quality monitoring needed
  - **Paper:** LLM-Pruner (2023), various compression studies

### **Pipeline Best Practices:**
- Evaluate quality at each stage
- Quantization typically last (easiest, most bang for buck)
- Consider if simpler single-technique is enough
- Test on your specific tasks, not just perplexity

---

## 5. **Expanding General Knowledge**

### 5a. **What's driving the knowledge update?**
   - **New Domain** (Legal, medical, scientific, or other specialized field) → Go to 5b
   - **Temporal Freshness** (Model knowledge is outdated) → Go to 5c
   - **New Language** (Extend to new languages) → Go to 5d

---

### 5b. **Domain Adaptation via Continued Pre-Training**

*Goal: Teach model deep understanding of a new domain*

- **Continued Pre-Training (CPT)** ⭐ Standard Approach
  - Continue training with next-token prediction on domain corpus.
  - **When to use:** Need broad domain knowledge at language model level
  - **Data needed:** Large unlabeled domain corpus (GB of text)
  - **Pros:** Deep domain understanding, learns terminology and patterns
  - **Cons:** Expensive, need substantial domain data, risk of forgetting
  - **Tools:** Standard pre-training frameworks

- **CPT on Base → Then Instruct Tune** ⭐ Recommended Path
  - Apply CPT to base model, then do instruction tuning.
  - **Order:** 1. CPT on base model with domain text → 2. SFT with domain instructions
  - **Pros:** Clear separation of knowledge and behavior, better control
  - **Cons:** Two training phases
  - **Paper:** Various domain adaptation studies

- **CPT on Instruct Model**
  - Continue pre-training directly on already instruction-tuned model.
  - **When to use:** Want simpler pipeline, accept some alignment drift
  - **Pros:** Single phase, preserves some instruction ability
  - **Cons:** Can distort alignment, quality varies
  - **Warning:** Monitor instruction following after CPT

- **DAPT (Domain-Adaptive Pre-Training)**
  - Targeted pre-training on domain, proven effective for many domains.
  - **Paper:** Don't Stop Pretraining (2020)
  - **Domains proven:** BioMed, CS, News, Reviews

### **When to use CPT vs RAG vs SFT:**
| Goal | Technique | Notes |
|------|-----------|-------|
| Deep domain understanding | CPT | Need large corpus |
| Dynamic/changing knowledge | RAG | Need citations |
| Specific tasks/behavior | SFT | Limited data OK |

---

### 5c. **Temporal Knowledge Updates**

*Goal: Update model's world knowledge to current time*

- **Continued Pre-Training on Recent Data**
  - Pre-train on recent news, web, papers to update world knowledge.
  - **Pros:** Comprehensive update, natural knowledge integration
  - **Cons:** Expensive, may need repeated updating
  - **Challenge:** Avoiding catastrophic forgetting of old knowledge

- **RAG with Current Sources** ⭐ Practical Default
  - Retrieve from up-to-date document stores, news APIs, web search.
  - **Pros:** Always current, no retraining, verifiable
  - **Cons:** Latency, retrieval dependent
  - **Tools:** Web search integration, news APIs

- **Knowledge Editing**
  - Surgically update specific facts without full retraining.
  - **When to use:** Need to correct specific facts
  - **Methods:** ROME, MEMIT, Knowledge neurons
  - **Pros:** Targeted, efficient
  - **Cons:** Limited scale, research stage
  - **Paper:** Locating and Editing Factual Associations (2022)

- **Temporal Tagging**
  - Include time context in training, teach model about its knowledge cutoff.
  - **Implementation:** Include dates in training data, system prompts about cutoff
  - **Pros:** Model aware of limitations
  - **Use:** Combine with RAG for best results

---

### 5d. **Multilingual Expansion**

*Goal: Extend model capabilities to new languages*

- **Multilingual CPT**
  - Continue pre-training with target language data.
  - **Data needed:** Large corpus in target language
  - **Pros:** Native language understanding
  - **Cons:** Expensive, may affect other languages
  - **Tip:** Mix target language with English to retain capabilities

- **Cross-Lingual Transfer (PEFT)**
  - Use LoRA/adapters to add language capabilities with minimal forgetting.
  - **Pros:** Efficient, less forgetting, modular
  - **Cons:** May be less fluent than full CPT
  - **Approach:** Train language-specific LoRA adapters

- **Translation Augmentation**
  - Translate English training data to target language, fine-tune on both.
  - **Pros:** Leverages existing data, quick
  - **Cons:** Translation artifacts, not native quality
  - **Tools:** NLLB, Google Translate API

---

## Quick Reference Summary

| **Goal** | **Recommended Technique** | **Notes** |
|----------|---------------------------|-----------|
| Add external knowledge | RAG | Dynamic, verifiable, no retraining |
| Change tone/format | Prompt Engineering | Start here, no training |
| Task specialization | LoRA/QLoRA (PEFT) | Default for most fine-tuning |
| Alignment/Safety | DPO or RLHF | DPO simpler, RLHF more powerful |
| Reduce memory/latency | Quantization (4-bit) | First compression step |
| Smaller architecture | Distillation | Train student from teacher |
| New domain expertise | Continued Pre-Training | Large domain corpus needed |
| API model customization | Provider Fine-Tuning | OpenAI, Together, etc. |
| Tool/function calling | Function Calling SFT | Or use capable base model |
| Maximum compression | Prune → Distill → Quantize | Combined pipeline |

---

## Decision Hierarchy

1. **Try prompting first** - it's free and fast
2. **If behavior issues persist** - consider RAG for knowledge or PEFT for behavior
3. **Use LoRA/QLoRA** as default fine-tuning method
4. **Full fine-tuning** only when PEFT is insufficient
5. **For deployment** - quantize first, then prune/distill if needed

---

## Common Patterns

| **Use Case** | **Recommended Stack** |
|--------------|----------------------|
| Chatbot | Base model + System prompt + RAG |
| Domain Expert | CPT + SFT (or RAG if data changes) |
| Production | LoRA fine-tune + 4-bit quantization |
| Edge Deployment | Distillation + Quantization |

---

## Popular Tools

| **Category** | **Tools** |
|--------------|-----------|
| RAG | LangChain, LlamaIndex, Pinecone, ChromaDB |
| Fine-tuning | HuggingFace TRL, Axolotl, LLaMA-Factory |
| PEFT | HuggingFace PEFT, bitsandbytes |
| Quantization | AutoGPTQ, AutoAWQ, llama.cpp |
| Alignment | TRL (DPO, PPO), OpenRLHF |
| Distributed | DeepSpeed, PyTorch FSDP |

---

**Version 1.0 - Based on Latest Research (2024)**
*Comprehensive guide to choosing the right LLM optimization technique*
