# LLM from scratch in pure NumPy

A small GPT-style language model built and trained from scratch using only NumPy: no PyTorch, no autograd. Every forward and backward pass is written by hand. It comes with a ChatGPT-style web UI that streams replies from the model.

Inspired by Green Code's videos [I Built an LLM from Scratch](https://www.youtube.com/watch?v=s9w3gtgvNSU) and [I Trained an LLM from Scratch](https://www.youtube.com/watch?v=vKdKtxpfvIc).

## What's inside

| File | What it does |
|---|---|
| `tokenizer.py` | Byte-level BPE tokenizer (GPT-2 style) trained on the dataset |
| `model.py` | Transformer (embeddings, multi-head causal self-attention, LayerNorm, GELU MLP, dropout) with hand-written backprop, plus AdamW |
| `test_gradcheck.py` | Checks every gradient against finite differences |
| `train.py` | Trains on [Tiny Shakespeare](https://github.com/karpathy/char-rnn) (downloaded automatically) |
| `generate.py` | Samples text from a checkpoint |
| `server.py` + `static/index.html` | ChatGPT-style web UI, using only the Python standard library |

## Quick start

Requires Python 3.9+ and NumPy.

```bash
pip install numpy

# chat with the included trained model
python3 server.py --out_dir out_dropout
# open http://localhost:8765

# or sample from the command line
python3 generate.py --out_dir out_dropout --prompt "JULIET:"
```

## Train your own

```bash
python3 test_gradcheck.py                 # verify backprop
python3 train.py                          # 4 layers, ~0.9M params, ~8 min on a laptop CPU
python3 train.py --n_layer 6 --n_embd 192 --dropout 0.2 --max_iters 5000 --out_dir out_dropout   # ~40 min
```

Run `python3 train.py --help` for all options. The web server reloads the checkpoint automatically whenever training saves a better one.

## Results

Tiny Shakespeare: 1.1M characters, 575K tokens with a 512-token vocabulary.

| Model | Params | Best held-out loss |
|---|---|---|
| 4 layers, 128 dim, no dropout | 0.94M | 2.937 |
| 6 layers, 192 dim, no dropout | 2.89M | 2.924 (overfits after step 1250) |
| 6 layers, 192 dim, dropout 0.2 (included) | 2.89M | **2.765** |

Sample:

```
JULIET:
I pray thee, thou shalt not speak.

CLARENCE:
Go, sit up thy brother, whose humble mother?

LADY CAPULET:
So you so?
```

## Using the chat

The model continues text; it doesn't follow instructions. The chat treats the conversation as one running play. End your message with a character name and a colon (e.g. `ROMEO:`) and the model writes that character's speech. Otherwise it writes the next character's reply.
