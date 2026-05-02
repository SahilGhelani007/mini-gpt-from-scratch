🧠 Lightweight GPT-style Transformer (From Scratch)

Author: Sahil Ghelani
Date: February 24, 2026

📌 Overview

This project implements a GPT-style Transformer model from scratch using PyTorch, focusing on simplicity, efficiency, and understanding core LLM components.

The model is designed (~80M parameters) for resource-constrained environments and lightweight chatbot applications.

⚙️ Features
Multi-head self-attention with causal masking
Transformer blocks with residual connections and LayerNorm
Feed-forward neural network with GELU activation
Token + positional embeddings
Fully autoregressive architecture
🏗️ Model Architecture
Layers: 8
Hidden Size: 512
Attention Heads: 8
Context Length: 256
Vocabulary Size: 30,000
📊 Performance Snapshot (Experimental)

(single GPU - T4, PyTorch)

Latency: ~18–25 ms/token (GPU)
Throughput: ~40–55 tokens/sec
CPU latency: ~60–85 ms/token

Note: These are approximate observations from internal testing.

⚡ Design Decisions
Chose smaller model size (~80M) to balance performance and hardware limits
Limited context length (256) to reduce memory usage
Focused on clarity and modular design over heavy optimization
⚖️ Tradeoffs
No KV caching (simpler implementation, slower inference)
No large-scale dataset training pipeline included
Limited evaluation metrics in current version
🔁 Future Improvements
KV cache for faster inference
Mixed precision (fp16) support
Better tokenization (BPE)
Training pipeline with logging + checkpoints
Deployment-ready API for chatbot usage
💡 Use Case

This architecture was also adapted into a lightweight chatbot prototype, focusing on fast response time and low resource usage.
