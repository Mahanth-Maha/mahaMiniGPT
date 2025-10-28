# Step-by-Step Towards Building a mini-GPT

This repository documents the incremental and empirical journey of building a GPT-like Transformer model from the ground up. This project is not just about arriving at a final model; it is about *understanding* the function and impact of every single component. We begin with the simplest possible language model a statistical Bigram and, step-by-step, add layers of complexity, measuring the impact of each addition.





<div align="center">
<br>
<br>
<br>
<h2><strong>The SOTA Transformer:</strong> <em>From</em> What if GPT-2 was built on current SOTA architectural design choices? ⇒ Building my own LLM </h2>

<p align="center">
|  
<a href="Research/DesignDoc.pdf"><b>📄 White Paper</b></a> | 
<a href="#"><b>🌐 Web Page</b></a> | 
<a href="logs/results_All.csv"><b>🥇 Results</b></a> |
<a href="#"><b>🐍 Sample Codes</b></a> |
<a href="#"><b>📃 Documentation</b></a> |
</p>
<br>
</div>





The project moves from:

1.  Simple statistical models (n-grams)
2.  Basic neural networks (MLPs)
3.  The first "attention" concept
4.  The full "Transformer block"
5.  Stabilization techniques (Residuals, LayerNorm)
6.  Modern architectural optimizations (SwiGLU, RMSNorm, GQA)
7.  State-of-the-art implementation details (Flash Attention)
8.  Stable pre-training methodologies (Initialization, LR Schedules)
9.  init methods, added features, Bug Fixes, loss and lr ablation studies
10. TRANSFORMER \& BEYOND -> I have pre-trained this model by scaling up (39 K $\rightarrow$ 1 B) the project is moved to the myt-llm project : [myT-LLM](github.com/Mahanth-Maha/myT-LLM) 


Each numbered directory represents a distinct experimental stage, isolating one new idea.

## Guiding Philosophy

The goal is to demystify the Transformer, which is often presented as a monolithic, "**black-box**" architecture. By building it piece by piece, we can answer critical questions:

* *Why* do we need self-attention? What problem does it *actually* solve over a simpler model?
  
* *Why* are residual connections and layer normalization non-negotiable? What happens if you don't have them? (See Stage 04 vs 05)

* *How* much does a better tokenizer *really* matter? (See Stage 07)

* *Are* modern tricks like SwiGLU and RMSNorm just marginal gains, or do they provide a significant, measurable advantage? (See Stage 08)

This repository serves as a "living" research notebook, where every experiment provides an empirical, data-driven answer to these questions.

---

## Project Structure & Experimental Log

Here is the high-level summary of each experimental stage, detailing the goals, design choices, and outcomes.

### 01: Probabilistic Models & Application

* **Goal:** Establish a quantitative baseline using the simplest, most fundamental generative models. This is the "control group" for our experiment, allowing us to scientifically measure the *actual* improvement provided by complex neural architectures.

* **Modules:**
    * `01_01_ProbablisticModels`: Implements a character-based **Bigram model**. This is essentially a **Markov chain** where the probability of the next character depends *only* on the single character that precedes it. It's a table of 
    $$P(char_n | char_n-1)$$
    * `01_02_Application_IndianNameCreator`: Applies this Bigram model and compares it against simple neural networks (MLPs) and a WaveNet-style convolutional model on a practical task: generating Indian names.

* **Key Insight (from original user notes):** This initial stage proved the fundamental weakness of statistical models. While the Bigram model can learn common letter pairings, it has no concept of long-term structure. The neural network models, even a simple MLP, performed significantly better. The best model from this exploratory phase (a 3-layer MLP with a 10D embedding and an 8-character context) achieved a validation loss of **1.40935**. This demonstrated that a model that learns *distributed representations* (embeddings) and can see a *wider context window* (8 chars vs. 1 char) is vastly superior.

*(Note: The comparative, quantitative analysis in this README begins at Stage 02, which is where the [results_All.csv](logs\results_All.csv) log starts.)*

---

### 02: Using Context Better (Introduction to Attention)

* **Goal:** To overcome the core limitation of n-gram and simple MLP models: the **fixed, local context window**. A model with an 8-char context cannot see the 9th character, failing to capture long-range dependencies (e.g., an opening parenthesis `(` must be paired with a closing one `)` pages later). 
 
    This stage experiments with methods for creating a *dynamic, learned context* that can, in theory, see the entire history.

* **Design Choices:**
    1.  `02_02_0_PureBigram`: The baseline control, using only the previous token (context=1).
    2.  `02_02_1_Average_ctx`: A "bag-of-words" approach. It *averages* the embeddings of *all* prior tokens. This model can "see" the entire history, but it has no idea *where* the tokens were. It's permutation-invariant, which is bad for language.
    3.  `02_02_2_Self_Attention`: A **single-head attention mechanism**. This is the breakthrough. Instead of a blind average, the model *learns* (via Query, Key, and Value projections) to decide which previous tokens are most *relevant* for predicting the next one. The "Query" asks, "I'm a verb, what's my subject?" The "Keys" from all previous tokens respond, "I'm a noun," "I'm an adjective," etc. The model then takes a weighted average of the "Values," paying the most "attention" to the most relevant token (the noun).
    4.  `02_02_3_Multi_Self_Attention`: A full (4-head) MHA block. This allows the model to ask *different questions* in parallel (e.g., one head for syntax, one head for semantics).

* **Results (Test Set):**

| Experiment | Design Choice | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| `02_02_0_PureBigram` | Baseline (Context=1) | 2.4640 | 0.2850 | 3.5548 |
| `02_02_1_Average_ctx` | Bag-of-Words Context | 2.4619 | 0.2853 | 3.5518 |
| `02_02_2_Self_Attention`| **Single-Head Attention** | **2.4578** | **0.2859** | **3.5458** |
| `02_02_3_Multi_Self_Attention`| Multi-Head (4) Attention | 2.4623 | 0.2852 | 3.5523 |

* **Insight:** Even at this tiny scale, the data is clear: the **Single-Head Attention** model (`02_02_2`) is measurably the best. It achieves the lowest test loss, highest accuracy, and lowest Bits Per Character (BPC). This empirically proves that a *learned, dynamic context* is superior to both a fixed context (Bigram) and a naive global context (Averaging). The MHA model performing worse suggests that, at this small scale, the extra parameters were harder to train and provided no benefit.

---

### 03: Adding Memory (Feed-Forward Networks)

* **Goal:** To add the second key component of a Transformer block: the **position-wise Feed-Forward Network (FFN)**. The attention layer's job is to *gather* and *mix* information across the sequence. The FFN's job is to *process* or "think" about that gathered information. It is an MLP applied *independently to each token* in the sequence, providing the block with deeper computational capacity and non-linearity.

* **Design Choices:**
    1.  `03_01_1_FFN_ReLU`: `MHA + FFN` using the standard, fast `ReLU` activation.
    2.  `03_01_2_FFN_GeLU`: `MHA + FFN` using `GeLU` (Gaussian Error Linear Unit). `GeLU` is a smoother, non-linear activation common in models like GPT and BERT, which is theorized to allow for more stable training.

* **Results (Test Set):**

| Experiment | Design Choice | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| `03_01_1_FFN_ReLU` | MHA + FFN (ReLU) | **2.4618** | **0.2853** | **3.5515** |
| `03_01_2_FFN_GeLU` | MHA + FFN (GeLU) | 2.4621 | 0.2853 | 3.5519 |

* **Insight:** The results are virtually identical. At this small scale, the choice of simple activation function (`ReLU` vs. `GeLU`) makes no difference. The critical outcome of this stage is that we have now successfully built a *single, complete Transformer block* (`MHA -> FFN`). However, its performance is no better than the attention-only model, suggesting a single block is not enough.

---

### 04: Stacking Blocks

* **Goal:** To create a "deep" network by stacking the complete Transformer blocks from Stage 03. The core hypothesis of deep learning is that depth enables the learning of **hierarchical representations**. Layer 1 might learn to group characters into word-parts, Layer 2 might learn syntax, and Layer 4 might learn semantic relationships.

* **Design Choices:** Stacked the (MHA + FFN) block `N=4` times to create the first "deep" model of the project.

* **Results (Test Set):**

| Experiment | Design Choice | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| `04_01_N_x_Blocks` | 4-Layer Stacked Blocks | 2.4622 | 0.2852 | 3.5521 |

* **Insight: A CRITICAL FAILURE!** This is the most important negative result. The 4-layer model (Test Loss: 2.4622) performs *no better* than the 1-layer model (Test Loss: 2.4618). The network is **not learning**. Simply stacking blocks does not work. This is due to the infamous **vanishing/exploding gradient problem**. The signal (gradient) is lost or corrupted as it tries to backpropagate through 4 deep blocks. This experiment *perfectly motivates* the next, and most crucial, architectural additions.

---

### 05: Adding Normalization & Optimizations

* **Goal:** To stabilize the training of the deep network from Stage 04 by adding the two components that truly make deep Transformers possible: **Residual Connections** ("Add") and **Layer Normalization** ("Norm").

* **Design Choices:**
    1.  `05_01_Residuals`: Added "shortcut" connections. The output of a block is `x_new = x + Block(x)`. This creates a direct "highway" for information and gradients to flow, bypassing the block. This solves the vanishing gradient problem and allows the model to easily learn an "identity" function (i.e., just pass `x` through) if a block isn't needed.
    2.  `05_03_1_post_LayerNorm`: The *original* Transformer design. `x_new = LayerNorm(x + Block(x))`. Normalization happens *after* the addition.
    3.  `05_03_2_pre_LayerNorm`: The GPT-2/LLaMA design. `x_new = x + Block(LayerNorm(x))`. Normalization happens *inside* the block, *before* the operation. This leaves the residual highway "clean" and is known to be more stable for very deep networks.
    4.  `05_04_Dropouts`: Added dropout, a regularization technique to prevent overfitting.

* **Results (Test Set):**

| Experiment | Design Choice | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| `05_01_Residuals` | + Residual Connections | **2.4611** | **0.2854** | **3.5505** |
| `05_02_Projections` | + Projection Layers | 2.4627 | 0.2851 | 3.5528 |
| `05_03_1_post_LayerNorm` | Post-LN (Original) | 2.4629 | 0.2851 | 3.5531 |
| `05_03_2_pre_LayerNorm` | Pre-LN (GPT-2 Style) | 2.4618 | 0.2853 | 3.5515 |
| `05_04_Dropouts` | + Dropout | 2.4623 | 0.2852 | 3.5523 |

* **Insight:** At the tiny scale of these char-level models, the performance differences are still negligible. The `Residuals` model is *technically* the best, but all models are stuck at the same ~2.46 loss. This is a profound insight: **these components are *enablers* for scaling, not *improvers* in themselves at a small scale.** Their true value is not that they lower the loss of a 4-block model, but that they *make it possible to train a 40-block or 400-block model at all.* We now have an architecturally-sound, stable, and *scalable* Transformer.

---

### 06: The First Transformer

* **Goal:** With a stable and complete architecture from Stage 05, the next logical step was to test the "scaling laws" hypothesis: does the model's performance predictably improve as we increase its parameter count?

* **Modules:** This stage (`06_Transformer.py`) involved training models of increasing size (`small`, `medium`, `large`, `xtraLarge`).

* **Insight:** *(Note: While these models were trained, their results are not logged in the final `results_All.csv`. The project's data log jumps from the micro-scale, char-level experiments of Stage 05 directly to a new, fundamentally different architecture in Stage 07. This implies Stage 07 was a hard pivot, deprecating the char-level models entirely in favor of a much more powerful approach.)*

---

### 07: Making Transformers Better (The Great Pivot)

* **Goal:** This stage represents a fundamental restart. The model is rebuilt to incorporate modern components, most importantly a **Byte-Pair Encoding (BPE) tokenizer** instead of a simple character vocabulary.

* **Design Choices:**
    1.  **Tokenization:** Replaced the character-level (vocab size ~128) model with a `cl100k_base` BPE tokenizer (vocab size **~100,000**). This is a *massive* change.
        * **Why?** A char model spends most of its capacity learning to group "t-h-e" into "the". A BPE model sees "the" as a single token.
        * **Effect 1 (Shorter Sequences):** A 1000-character sequence becomes ~250 tokens. Since attention is `O(n^2)`, this is a *massive* quadratic speed-up.
        * **Effect 2 (Easier Task):** The model now predicts "world" after "Hello", not "e" after "h". The task is more semantically meaningful from the start.
    2.  **Attention Efficiency:** Tested modern attention variants.
        * `MHA` (Multi-Head): Standard.
        * `MQA` (Multi-Query): All heads share one Key and Value. Drastically reduces memory bandwidth during *inference*.
        * `GQA` (Grouped-Query): A compromise. Groups of heads share K/V.

* **Results (Test Set):**

| Experiment | Design Choice | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| `07_00_0_Transformer` | Baseline (Char-level) | 11.7760 | 0.0480 | 11.5165 |
| `07_00_2_Tokenizer` | **+ BPE Tokenizer** | **7.9622** | **0.0483** | **11.4883** |
| `07_02_0_MultiHeadAttn`| Standard MHA | 7.9622 | 0.0483 | 11.4883 |
| `07_02_2_SingleQ` | Multi-Query Attn (MQA) | 7.9606 | 0.0484 | 11.4860 |
| `07_02_3_GroupedQ` | Grouped-Query Attn (GQA)| 7.9606 | 0.0484 | 11.4860 |

* **Insight:** This is the **single most important result in the project.**
    1.  Changing *nothing* but the tokenizer (`07_00_0` vs `07_00_2`) caused the test loss to **plummet from 11.77 to 7.96**. This proves that a good tokenizer is *more important* than many small architectural tweaks.
    2.  Test Accuracy is low (~4.8%) because predicting the *exact* token out of 100,000 is extremely hard. This is why loss and BPC are better metrics.
    3.  `MQA` and `GQA` performed *identically* to `MHA` (and slightly better). This is a huge win: we can get their massive inference speed-ups and memory savings for *free*, with zero loss in quality.

---

### 08: GPT Architectural Optimizations

* **Goal:** With a modern, tokenized, and efficient GQA model, this stage is a deep dive into fine-tuning the "micro-architecture" of the Transformer block, testing design choices from recent state-of-the-art models (like LLaMA).

* **Design Choices:**
    1.  **FFN Dimension:** Tested various "inverted bottleneck" sizes. The FFN expands the embedding dimension (`d_model -> d_ffn`) and then contracts. Standard is 4x. Is that optimal?
    2.  **Normalization:** Compared `LayerNorm (LN)` against the simpler, faster **`RMSNorm (RN)`**. RMSNorm *only* normalizes by the "scale" (the root-mean-square), skipping the mean-centering step. It's computationally cheaper.
    3.  **Activation:** Compared `GeLU` against gated activations like **`SwiGLU`**. `SwiGLU` uses a "gate" to control the flow of information, making it more dynamic and powerful.
    4.  **Attention Implementation:** Tested **`Flash Attention`**, an I/O-aware *implementation* (not architecture) that computes attention without ever writing the massive `N x N` attention matrix to GPU RAM, providing huge speedups.

* **Results (Test Set):**

| Experiment Group | Best Model | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| **FFN Dimension** | `08_01_03_Hid_dim_2x` | 7.9592 | 0.0483 | 11.4842 |
| **Normalization** | `08_02_3_Norm_RN_LN` | 7.9585 | 0.0483 | 11.4831 |
| **Activation** | `08_03_08_swiglufast` | **7.9543** | **0.0484** | **11.4771** |
| **Attention Impl.** | `08_04_02_flash` | 7.9606 | 0.0484 | 11.4860 |

* **Insight:** A series of massive wins:
    1.  **FFN:** A *smaller* FFN (2x `d_model`) performed *better* than the standard 4x. This is a crucial finding, saving parameters and computation.
    2.  **Normalization:** `RMSNorm` (`RN_LN`) performed best, confirming it's a faster and slightly better replacement for LayerNorm.
    3.  **Activation:** **`SwiGLU` (`swiglufast`) was the clear winner**, achieving the lowest loss so far (7.9543).
    4.  **Attention:** `Flash Attention` performed *identically* to the standard implementation. This is a perfect result: it proves Flash Attention is a 100% "free" optimization, providing massive speed and memory benefits with zero quality loss.
    * **Conclusion:** The optimal architecture is a (Pre-)RMSNorm, GQA, SwiGLU-FFN model. This is the LLaMA-style architecture.

---

### 09: Fast Pre-training Optimizations

* **Goal:** To finalize the "mytNano" model by combining all the winning components from Stages 07 and 08 and focusing on the dynamics of a stable pre-training run.

* **Design Choices (The "All-In" Model):**
    * **Architecture:** `Pre-RMSNorm`, `GQA`, `SwiGLU (2x FFN)`, `Flash Attention`.
    * **Initialization:** Implemented proper weight initialization (e.g., `std=0.02` as in GPT-2) to prevent the loss from exploding to `NaN` on the first step.
    * **LR Schedule:** Implemented a **Cosine Decay Learning Rate Schedule** with a linear **Warmup**. This starts the LR low, gently "warms up" to a max value, and then slowly decays it. This is far more stable and effective than a fixed LR.

* **Results (Test Set):**

| Experiment | Design Choice | Test Loss | Test Acc. | Test BPC |
| :--- | :--- | :--- | :--- | :--- |
| `09_01_myt-tiny` | Final Model + Tuning | **7.4214** | 0.0482 | 10.7068 |
| `09_02_myt-tiny` | Final Model + Tuning | **7.4214** | 0.0482 | 10.7068 |

* **Insight:** This is the final payoff. By combining all architectural optimizations (GQA, SwiGLU, RMSNorm) *and* proper training dynamics (init, LR schedule), the test loss **dropped again from ~7.95 to ~7.42**. This ~0.5 loss improvement is the "alpha" gained from meticulous pre-training tuning. This final model, "mytNano," is the culmination of all 9 stages of research.

---

## Future Work

* To Scale this and pretrain this model

TADA!!! Its Already Done : [myt-LLM](https://github.com/Mahanth-Maha/myt-llm)

⚠️ NOTE: If you can't able to access the myt-llm link → the project repo might be still in private, since documentation is hell a lot of work and I have to deal with lot of academic work, so, it takes time :) , or I might only release code for development (soon). 


---

<p align="left">
    <span style="
    font-size:22px;
    font-family:'Georgia', 'Garamond', 'Times New Roman', serif;
    font-weight:700;
    color:#94dfff;">
    <a href="https://mahanthyalla.in" style="color:#94dfff; text-decoration:none;">
      Mahanth Yalla
    </a>
  </span><br>
  <span style="font-size: 16px; font-family:'Times New Roman', Times, serif;">
      MTech - Artificial Intelligence,<br>
      <strong>Indian Institute of Science,</strong><br>
      Bangalore - 560012.<br>
  </span>
</p>
