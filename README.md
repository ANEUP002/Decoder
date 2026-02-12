
#  GPT From Scratch — PyTorch

### Building a GPT-style language model from first principles  
**Architecture • Training • Inference • Generation**

Inspired by **Andrej Karpathy’s “GPT from scratch / nanoGPT” videos**

---

## Overview

This repository contains my **end-to-end implementation of a GPT-style language model** using **PyTorch**, built entirely **from scratch**.

The purpose of this project was not to fine-tune a pretrained model, but to **rebuild the GPT architecture myself** in order to deeply understand how modern large language models work — from **causal self-attention**, to **training dynamics**, to **autoregressive text generation**.

> **Mental model:**  
> **tokens in → logits out**

Everything in this repository — the model, training loop, optimizer setup, and inference code — is implemented manually.

---

##  What This Project Includes

- GPT architecture implemented from first principles  
- Causal self-attention with autoregressive masking  
- Flash attention via `torch.scaled_dot_product_attention` (when available)  
- Pre-norm Transformer blocks (GPT-2 style)  
- MLP with 4× expansion and GELU activation  
- Weight tying between token embeddings and language modeling head  
- HuggingFace GPT-2 weight loading into a scratch model  
- Custom training loop with gradient accumulation  
- Linear warmup + cosine decay learning-rate schedule  
- Mixed-precision training (bfloat16)  
- `torch.compile` for performance optimization  
- Autoregressive inference with:
  - temperature
  - top-k sampling
  - top-p (nucleus sampling)
  - repetition penalty  

---

##  Model Architecture

This implementation closely follows **GPT-2 design decisions**, with an emphasis on clarity and correctness.

### Transformer Block (Pre-Norm)

LayerNorm → Causal Self-Attention → Residual
LayerNorm → MLP (4× expansion + GELU) → Residual



### Core Design Choices
- Single QKV projection (GPT-2 style)
- Causal masking to enforce autoregressive behavior
- Pre-norm architecture for training stability
- Weight tying for parameter efficiency

---

##  Training

The model was trained on **Shakespeare text**, tokenized using **GPT-2 BPE via `tiktoken`**.

### Training Setup
- Context length: 1024  
- Effective batch size: achieved via gradient accumulation  
- Optimizer: AdamW (with correct weight-decay handling)  
- Learning rate: linear warmup followed by cosine decay  
- Precision: bfloat16 autocast  
- Gradient clipping: 1.0  
- Compilation: `torch.compile` enabled  

The training loop mirrors real-world LLM pipelines, including:
- manual data loading
- micro-batch accumulation
- learning-rate scheduling
- throughput logging
- checkpoint saving

---

##  Inference & Text Generation

Text generation is implemented using a fully manual **autoregressive decoding loop**.

### Supported Sampling Controls
- Temperature scaling  
- Top-k sampling  
- Top-p (nucleus sampling)  
- Repetition penalty  

### Example Prompt
ROMEO:

KING HENRY VI:
But, having me when's voices, unnight.
Most children's death? my stroke that country?

Romeo.
LADY:
My child, and hast appointed


---

##  What I Learned

- How causal self-attention enforces autoregressive generation  
- Why pre-norm Transformers train more stably  
- Why weight decay should not apply to LayerNorm and bias terms  
- How GPT-2 uses a single QKV projection  
- How pretrained GPT-2 weights map onto a scratch implementation  
- How sampling strategies affect generation quality  
- How large batch sizes are simulated using gradient accumulation  
- What a real LLM training loop looks like beyond tutorials  

---

## Credits & Inspiration

This project is **heavily inspired by Andrej Karpathy**, particularly his:

- *“Let’s build GPT from scratch”*
- *nanoGPT walkthroughs*

I followed his philosophy of **minimal, readable, and correct code**, while implementing everything myself to deeply understand how GPT models work internally.

---

##  Final Notes

This project was built as a **serious learning exercise**, structured to reflect **real ML engineering practices**.

The focus was not on training the largest model, but on **understanding the architecture end-to-end** — from math → code → training → generation.

