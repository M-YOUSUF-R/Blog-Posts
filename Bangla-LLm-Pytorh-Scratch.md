# Building a Bangla LLM from Scratch: A Complete Guide

## Introduction

Have you ever wondered how language models like GPT actually work under the hood? In this comprehensive guide, we'll build a Bangla language model from scratch - complete with data processing, tokenization, model architecture, training, and text generation. This is not just another "copy-paste" tutorial; we'll deeply understand **why** each component exists and **how** they work together.

By the end of this blog post, you'll have:
- A working Bangla LLM trained on 100k samples from CulturaX
- Deep understanding of transformer architecture
- Hands-on experience with tokenization, training, and inference
- A modular codebase you can extend to other languages

Let's dive in!

---

## 1. Why Bangla? Why From Scratch?

**Bangla (Bengali)** is the 7th most spoken language in the world with over 230 million speakers. Yet, it's underrepresented in the LLM space. Building a Bangla model from scratch allows us to:

1. **Understand the fundamentals** without abstractions
2. **Customize everything** - tokenizer, architecture, training data
3. **Handle a non-Latin script** with its unique challenges
4. **Create something useful** for a large community

The "from scratch" approach means we won't use pre-built libraries like HuggingFace Transformers for the model - we'll implement the transformer ourselves using PyTorch.

---

## 2. Setting Up the Environment

### Hardware Detection

Our code automatically detects and uses the best available hardware:

```python
if torch.xpu.is_available():
    device = torch.device("xpu")      # Intel GPU via oneAPI
elif torch.cuda.is_available():
    device = torch.device("cuda")     # NVIDIA GPU
else:
    device = torch.device("cpu")      # Fallback
```

This makes the notebook **hardware-agnostic** - it runs on Intel GPUs, NVIDIA GPUs, or plain CPUs.

### Key Libraries

```python
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers
```

---

## 3. The Data Pipeline: From Raw Text to Training Ready

### 3.1 Downloading the Dataset

We use **CulturaX**, a multilingual dataset with 167 languages. For Bangla, we download 100,000 samples:

```python
dataset = load_dataset("uonlp/CulturaX", 'bn', split="train", streaming=True)
```

**Why streaming?** Instead of downloading the entire dataset (potentially 100GB+), we stream it sample by sample and cache only what we need.

**Caching strategy**: After the first download, we save the raw text as `raw_text.pkl` to avoid re-downloading:

```python
if RAW_CACHE_FILE.exists():
    raw_text = pickle.load(open(RAW_CACHE_FILE, 'rb'))  # Load from cache
else:
    # Download and save to cache
    raw_text = [sample['text'] for sample in dataset]
    pickle.dump(raw_text, open(RAW_CACHE_FILE, 'wb'))
```

### 3.2 Text Cleaning: The Art of Data Preparation

Bangla text comes with noise - URLs, HTML tags, emojis, English script, and various special characters. Our cleaning pipeline:

```python
def clean_text(text):
    # 1. Remove URLs and HTML
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    
    # 3. Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)
    
    # 4. Keep only Bangla + digits + basic punctuation
    text = keep_pure_bangla(text)
    
    # 5. Normalize whitespace and punctuation
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    return text.strip()
```

**Why NFKC normalization?** Bangla has multiple ways to represent the same character (e.g., using combining diacritics). NFKC normalizes these to a standard form, ensuring consistency.

**The character filter** `keep_pure_bangla` uses regex to keep:
- Bangla characters: `\u0980-\u09FF`
- Bangla digits: `\u09E6-\u09EF`
- Zero-width joiners: `\u200C\u200D` (important for correct rendering)
- Basic punctuation: `. , ! ? ; : ' " ( ) [ ] { } — – … / % @ & + = < > ৳ -`

### 3.3 Building the Corpus File

We write all cleaned texts to a single file, one sample per line:

```python
with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
    for text in tqdm(raw_text, desc="Cleaning & writing"):
        cleaned = clean_text(text)
        if cleaned:
            f.write(cleaned + '\n')
```

**Output**: ~600 MB corpus file with ~47M tokens after tokenization.

---

## 4. Tokenization: Building a BPE Tokenizer

### 4.1 Why BPE (Byte-Pair Encoding)?

BPE is a subword tokenization algorithm that strikes a balance between:
- **Word-level**: Too large vocabulary, poor handling of unseen words
- **Character-level**: Too long sequences, loses semantic meaning

BPE learns common subword units (like "মোকাবি" + "লায়") that appear frequently in the corpus.

### 4.2 Setting Up the Tokenizer

```python
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer.decoder = decoders.Metaspace()
```

**Metaspace pre-tokenizer**: Replaces spaces with `▁` (a special underscore). This ensures:
- Spaces are preserved during tokenization
- We can distinguish between word boundaries and internal characters

### 4.3 Special Tokens

```python
SPECIAL_TOKENS = [
    '<pad>',      # Padding for batch processing
    '<unk>',      # Unknown tokens
    '<bos>',      # Beginning of sequence
    '<eos>',      # End of sequence
    '<sep>',      # Separator for pairs
    '<|user|>',   # For future chat functionality
    '<|assistant|>',
    '<|system|>'
]
```

### 4.4 Training the BPE Tokenizer

```python
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True
)
tokenizer.train([CORPUS_FILE], trainer)
```

**Vocabulary size: 32,000** - This is small enough for our model size but large enough to cover Bangla effectively.

### 4.5 Testing the Tokenizer

```python
test_text = "করোনা মোকাবেলায় নিজ নিজ কর্মস্থলে"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded.ids)

print(f"Tokens: {encoded.tokens[:10]}...")
# Output: ['<bos>', '▁করোনা', '▁মোকাবেলায়', '▁নিজ', '▁নিজ', '▁কর্মস্থলে']
```

The `▁` token represents spaces. The roundtrip (encode -> decode) should preserve the original text.

---

## 5. Model Architecture: Decoder-Only Transformer

This is the heart of our LLM. We'll build a **GPT-style decoder-only transformer**. Let's understand each component in detail.

### 5.1 Configuration

```python
@dataclass
class BanglaLLMConfig:
    vocab_size: int = 32000
    d_model: int = 384          # Embedding dimension
    n_layers: int = 6           # Number of transformer blocks
    n_heads: int = 6            # Number of attention heads
    d_ff: int = 1536            # Feed-forward hidden size (4x d_model)
    dropout: float = 0.1
    max_seq_len: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
```

**Why these numbers?**
- `d_model=384`: A reasonable embedding size for ~23M parameters
- `n_layers=6`: Deep enough to learn complex patterns
- `n_heads=6`: Each head focuses on different aspects (syntax, semantics, etc.)
- `max_seq_len=256`: 256 tokens context window

### 5.2 Embeddings: Token + Position

```python
class BanglaGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
```

**Token Embedding**: Maps each token ID (0-31999) to a 384-dimensional vector. This is a lookup table with `32000 * 384 = 12.28M` parameters.

**Positional Embedding**: Since transformers have no built-in sense of order, we add a position-dependent vector to each token's embedding. Position 0 gets a unique vector, position 1 gets another, etc.

**Why not sinusoidal positional encoding?** Learned positional embeddings are simpler and work just as well for fixed sequence lengths.

### 5.3 Multi-Headed Self-Attention

This is where the magic happens! Self-attention allows each token to "look at" all previous tokens.

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
```

**The Attention Mechanism (Step by Step):**

1. **QKV Projection**: Each token's embedding is projected to query (Q), key (K), and value (V) vectors:
   ```
   Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
   ```

2. **Head Splitting**: The 384-dimensional vectors are split into 6 heads of 64 dimensions each.

3. **Attention Scores**: For each pair of tokens (i, j), compute:
   ```
   score(i,j) = Q_i · K_j / √(head_dim)
   ```

4. **Causal Masking**: For autoregressive generation, token i can only attend to tokens j ≤ i. We mask future positions:
   ```python
   mask = torch.tril(torch.ones(T, T))
   attn = attn.masked_fill(mask == 0, float('-inf'))
   ```

5. **Softmax**: Convert scores to probabilities:
   ```
   weights = softmax(attn_scores)
   ```

6. **Weighted Sum**: Compute context vectors:
   ```
   output_i = Σ weights(i,j) * V_j
   ```

**Why multiple heads?** Each head learns to focus on different types of relationships:
- Head 1 might track subject-verb agreement
- Head 2 might focus on nearby words
- Head 3 might look for specific syntactic patterns

### 5.4 Feed-Forward Network

After attention, we apply a 2-layer MLP:

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)      # 384 -> 1536
        self.fc2 = nn.Linear(config.d_ff, config.d_model)      # 1536 -> 384
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))      # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Why GELU?** Gaussian Error Linear Unit is smoother than ReLU and often performs better in transformers.

**Parameters**: `384 * 1536 + 1536 * 384 ≈ 1.18M` per block.

### 5.5 Transformer Block with Residual Connections

Each block follows the "Pre-LN" architecture:

```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        # Attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        
        # Feed-forward with residual
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
```

**Why residual connections?** They allow gradients to flow directly through the network, preventing vanishing gradients in deep models.

**Why LayerNorm before attention (Pre-LN)?** Pre-LN is more stable than Post-LN for deep transformers, allowing easier training.

### 5.6 Weight Tying

The final projection from `d_model` to `vocab_size` shares weights with the input embedding:

```python
self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
self.head.weight = self.token_emb.weight  # Weight tying!
```

**Why weight tying?**
- Reduces parameters by ~12M (from ~35M to ~23M)
- Forces the model to learn consistent representations
- Is a common technique in language models (used in GPT-1, BERT, etc.)

### 5.7 Parameter Count Breakdown

Let's count the parameters:

| Component | Parameters |
|-----------|------------|
| Token Embeddings + Head (tied) | 32,000 × 384 = 12.29M |
| Positional Embeddings | 256 × 384 = 0.10M |
| Each Transformer Block: | |
| - QKV Projection | 384 × (3 × 384) = 0.44M |
| - Output Projection | 384 × 384 = 0.15M |
| - Feed-Forward fc1 | 384 × 1536 = 0.59M |
| - Feed-Forward fc2 | 1536 × 384 = 0.59M |
| - LayerNorms (2) | 2 × 384 = 0.001M |
| **Total per block** | **~1.77M** |
| **6 blocks** | **~10.6M** |
| **Final LayerNorm** | **384** |
| **GRAND TOTAL** | **~23.03M** |

**23 million parameters** - This is small compared to GPT-3 (175B) but large enough for learning meaningful Bangla patterns.

---

## 6. Training the Model

### 6.1 Dataset Preparation

We chunk the tokenized text into sequences of length `max_seq_len + 1`:

```python
class BanglaTextDataset(Dataset):
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.token_ids[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target (shifted by 1)
```

For each position `t`, the model predicts token `t+1`. This is the **autoregressive** training objective.

**Example**: For sequence [A, B, C, D, E]:
- Input: [A, B, C, D]
- Target: [B, C, D, E]

### 6.2 Learning Rate Schedule

We use **warm-up + cosine decay**:

```python
def get_lr(step):
    if step < config.warmup_steps:
        # Linear warm-up from 0 to learning_rate
        return config.learning_rate * step / config.warmup_steps
    # Cosine decay to 0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
```

**Why warm-up?** At the start of training, gradients are unstable. Gradually increasing the learning rate prevents divergence.

**Why cosine decay?** Smoothly reducing the learning rate helps the model converge to a minimum.

### 6.3 Training Loop

```python
for epoch in range(max_epochs):
    for input_ids, targets in train_loader:
        # Forward pass
        outputs = model(input_ids, targets)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
```

**Gradient clipping**: Prevents exploding gradients by capping gradient norms to 1.0.

### 6.4 Validation and Checkpointing

After each epoch, we evaluate on the validation set and save the best model:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "model/best_model.pt")
```

---

## 7. Text Generation: Sampling Strategies

Our model outputs **logits** (raw scores). To generate text, we convert logits to probabilities and sample:

### 7.1 Temperature Scaling

```python
logits = logits / temperature
```

- `temperature = 1.0`: Standard softmax
- `temperature < 1.0`: Sharper distribution (more deterministic)
- `temperature > 1.0`: Softer distribution (more creative/random)

### 7.2 Top-K Filtering

Keep only the top K most likely tokens:

```python
v, _ = torch.topk(logits, top_k)
logits[logits < v[:, [-1]]] = float('-inf')
```

If `top_k=50`, we only consider the 50 most probable tokens.

### 7.3 Top-P (Nucleus) Filtering

Keep the smallest set of tokens whose cumulative probability exceeds `top_p`:

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
```

This is adaptive - for highly confident predictions, we keep fewer tokens; for uncertain predictions, we keep more.

### 7.4 Sampling

Finally, we sample from the filtered distribution:

```python
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

This introduces randomness, making generation diverse.

---

## 8. Results and Discussion

### Sample Output

```
Prompt: "বরগুনার আলোচিত রিফাত শরীফ হত্যা মামলার প্রধান সাক্ষী"

Output: "বরগুনার আলোচিত রিফাত শরীফ হত্যা মামলার প্রধান সাক্ষী শারীরিক ২০২১, 
হাস প্রতিবাদে প্রভাষ করেইফর্মীতি নামের 2 লেখক প্রদর্শছে যান নির্বাচিতঙ্গ ব্য"
```

### Why Isn't the Output Meaningful?

**The model has 23M parameters** - tiny compared to GPT-2 (124M) or GPT-3 (175B). With only 23M parameters and ~47M tokens of training data, the model:
- Has limited capacity to learn complex grammar
- Can't capture long-range dependencies
- Tends to memorize patterns rather than understand meaning

**Is this a failure? Not at all!** This is exactly what we expect from a small model. The goal of this tutorial is **education**, not SOTA performance.

### What Did We Learn?

1. **Fundamental building blocks** of LLMs
2. **Data preprocessing** challenges for non-Latin scripts
3. **Training dynamics** - learning rates, gradient clipping
4. **Generation strategies** - temperature, top-k, top-p
5. **Realistic expectations** - bigger models need more data

---

## 9. Scaling Up: How to Improve

If you want to build a better Bangla LLM:

### More Parameters
- Increase `d_model` to 768, 1024, or 1536
- Increase `n_layers` to 12, 24, or more
- Larger `vocab_size` (e.g., 50,000)

### More Data
- Use the full CulturaX dataset (or combine with other sources)
- Train on 1M+ samples

### Better Tokenization
- Experiment with Unigram tokenization (used in SentencePiece)
- Add Bangla-specific pre-processing

### Advanced Techniques
- **Flash Attention**: Faster attention for longer sequences
- **RoPE (Rotary Positional Embeddings)**: Better positional encoding
- **Grouped-Query Attention**: Reduces memory for large models

---

## 10. Code Architecture Overview

```
project/
├── data/
│   ├── raw/
│   │   └── raw_text.pkl        # Cached raw texts
│   └── cleaned/
│       └── bangla_corpus.txt   # Cleaned corpus file
├── tokenizer/
│   └── bangla_tokenizer/
│       └── bangla_bpe_tokenizer.json
├── model/
│   └── best_model.pt           # Trained weights
├── llm-bangla-note.ipynb       # Main notebook
└── README.md
```

---

## 11. Key Takeaways

1. **LLMs are complex but understandable**: Every component has a clear purpose
2. **Data is as important as the model**: Poor data quality = poor model
3. **Tokenization matters**: BPE handles subword units beautifully
4. **Small models are great for learning**: Quick iteration, clear debugging
5. **Hardware flexibility**: Our code runs on Intel, NVIDIA, or CPU
6. **Exportable knowledge**: The same architecture powers GPT-3, ChatGPT, and more

---

## 12. Next Steps

### For Learners
1. Implement Flash Attention for faster training
2. Add instruction fine-tuning with chat templates
3. Experiment with different model sizes
4. Build a web interface for your model

### For Practitioners
1. Scale to 100M+ parameters
2. Train on 1M+ samples
3. Deploy with ONNX or TorchScript
4. Add quantization for faster inference

---

## Appendix: The Attention Formula

The core attention equation in one line:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$: Query matrix (what the token is looking for)
- $K$: Key matrix (what each token offers)
- $V$: Value matrix (the actual information)
- $d_k$: Dimension of keys (head_dim)
- The division by $\sqrt{d_k}$ prevents gradients from vanishing

---

## Conclusion

Building an LLM from scratch demystifies the "black box" of AI. You now understand:
- How transformers process text
- Why tokenization is crucial
- What happens during training
- How generation works

**The code is yours to use, modify, and extend**. Run it on your machine, experiment with the parameters, and build your own language models!

---

*Happy building! Feel free to reach out with questions or share your experiments.*