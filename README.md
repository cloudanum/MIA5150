# MIA5150 · Generative AI & LLMs — Code Labs

Hands-on, self-contained **Colab notebooks** for *MIA5150 — Generative AI & Large Language Models* (University of Ottawa, Fall 2026, Instructor: Imran Ahmad).

**How to use:** click any **Open in Colab** badge below, then *Runtime → Run all*. No local setup. Everything a notebook needs either ships with Colab (PyTorch, NumPy, Matplotlib) or is installed/downloaded inside the notebook itself.

---

## Module: Sequential Models — RNNs, GRUs & LSTMs

Six notebooks, one learning arc: build it by hand → watch it fail → see why the fix works → use it on real data → generate with it → apply it to the real world.

| # | Notebook | What you build | What you *see* |
|---|----------|----------------|----------------|
| 1 | **An RNN from scratch (NumPy only)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cloudanum/MIA5150/blob/main/notebooks/sequential-models-rnn-gru-lstm/01_rnn_from_scratch.ipynb) | A vanilla RNN — forward pass *and* backpropagation through time — in ~100 lines of NumPy, trained to memorize a repeating text | The hidden state is just $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$; the *same* weights reused at every step; gradient-check your own BPTT against finite differences |
| 2 | **Why RNNs forget: the vanishing gradient, live** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cloudanum/MIA5150/blob/main/notebooks/sequential-models-rnn-gru-lstm/02_vanishing_gradients.ipynb) | The classic *adding problem* at short vs long sequence lengths, with RNN vs LSTM vs GRU | Gradient norms decaying *exponentially* with distance (semilogy plots); an RNN that solves lag-20 but is helpless at lag-200 while LSTM/GRU sail through |
| 3 | **LSTM & GRU under the hood: watch the gates** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cloudanum/MIA5150/blob/main/notebooks/sequential-models-rnn-gru-lstm/03_lstm_gru_gates.ipynb) | LSTM and GRU cells from scratch, verified line-for-line against `nn.LSTMCell` / `nn.GRUCell` | Forget/input/output gate activations plotted over time on a "carry this signal 30 steps" task — watch the forget gate clamp open and the cell state act like a conveyor belt |
| 4 | **RNN vs GRU vs LSTM on real text (IMDb sentiment)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cloudanum/MIA5150/blob/main/notebooks/sequential-models-rnn-gru-lstm/04_sentiment_imdb.ipynb) | A full NLP pipeline (tokenize → vocab → embed → recurrent core → classify) with one trainer and three swappable cores | Side-by-side accuracy/params/speed comparison; GRU ≈ LSTM in accuracy with ~25% fewer parameters; vanilla RNN stumbling on long reviews |
| 5 | **Teaching a GRU to write: character-level text generation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cloudanum/MIA5150/blob/main/notebooks/sequential-models-rnn-gru-lstm/05_char_text_generation.ipynb) | A character-level language model on Tiny Shakespeare with temperature sampling | The *same next-token objective* as GPT, one generation earlier — and what temperature does to the coherence/diversity trade-off |
| 6 | **Sequence models meet the real world: time-series forecasting** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cloudanum/MIA5150/blob/main/notebooks/sequential-models-rnn-gru-lstm/06_time_series_forecasting.ipynb) | An LSTM forecaster for airline-passenger traffic, with proper chronological splits | Why you must **never shuffle** time series (leakage!), RMSE vs naive baselines, and how recursive multi-step forecasts degrade with horizon |

### Suggested path

`01 → 02 → 03` is the theory spine (mechanism → failure mode → the gated fix). `04`, `05`, `06` are independent application tracks — assign any or all after 03.

---

## Runtime tips

- Everything runs on Colab's free **CPU** tier in minutes; notebook 5 is smoother on a GPU (*Runtime → Change runtime type → T4*).
- All notebooks are deterministic (fixed seeds), so your plots should match a classmate's.
- Each notebook ends with **Try it yourself** exercises — those are the assignment-ready bits.

## Companion material

- Course videos: the **AI Algorithms** playlist on the [neurals.ca YouTube channel](https://www.youtube.com/@neurals_ca) — episode *How Machines Remember: RNNs, GRUs & LSTMs* walks this same arc.

## Roadmap

Coming next to this repo, following the course: attention & transformers, LLM pre-training, VAEs, GANs, diffusion.

---

*Notebooks are tested against Colab's current PyTorch. If a Colab update breaks something, open an issue.*
