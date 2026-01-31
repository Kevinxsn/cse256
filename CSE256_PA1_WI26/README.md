
# Assignment Execution Instructions

## Part 1: Deep Averaging Network (DAN)

To run the DAN model, use the following command:

```bash
python main.py --model DAN

```

### Configuration

To change the model configuration, please edit the `main.py` file directly under the `if args.model == 'DAN'` block.

**1. Changing Embedding Dimensions**
To switch between 50d and 300d embeddings, comment/uncomment the corresponding `emb_path` lines:

```python
# emb_path = "data/glove.6B.50d-relativized.txt"
emb_path = "data/glove.6B.300d-relativized.txt" 

```

**2. Changing Embedding Initialization (GloVe vs. Random)**
Modify the `DAN` model initialization parameters:

```python
model = DAN(
    word_embeddings=embs,
    hidden_size=200,
    num_layers=2,
    dropout=0.3,
    use_glove=False,          # Set to True for GloVe, False for random initialization
    frozen_embeddings=True,   # Set to True to freeze embeddings, False to train them
)

```

---

## Part 2: Subword DAN

To run the Subword DAN model, use the following command:

```bash
python main.py --model SUBWORDDAN

```

### Configuration

To modify BPE training parameters or model architecture, please edit the `DANSubword` initialization in `main.py`:

```python
model = DANSubword(
    vocab_size=len(subword_indexer),
    emb_dim=args.bpe_emb_dim,
    hidden_size=200,
    num_layers=2,
    dropout=0.3,
    pad_idx=0
)

```


