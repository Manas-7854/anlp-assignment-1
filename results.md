# ANLP Assignment 1 — Final Experimental Results

## 1. Executive summary

This document reports the final results for all five encoder-decoder Transformer
configurations. Every reported test metric was computed from greedy-decoded
predictions produced by the checkpoint with the lowest validation loss. The
models use the same data split, random seed, global Transformer dimensions,
optimizer, learning-rate schedule, batch size, training duration, and decoding
limit wherever applicable.

The main findings are:

1. **C5 (BLT/raw bytes) is strongest overall** on bit-level accuracy and mean
   Levenshtein distance. It obtains **91.6518% bit accuracy** and a mean edit
   distance of **103.882**.
2. **C2 (RoPE) is the strongest learned-tokenizer model** after fixing RoPE in
   decoder cross-attention. It obtains **68.8562% bit accuracy**, **0.413740
   BLEU**, and the highest ROUGE scores among C1-C4.
3. C1 is the second-best tokenized model. C4 is close to C1, showing that
   RMSNorm preserves most baseline quality while training faster in this run.
4. C3 trades quality for fewer parameters and higher throughput through GQA.
5. Exact sequence accuracy is extremely low for every model because an entire
   long example is counted wrong after even one character error.

## 2. Configuration definitions

| Configuration | Positional method | Attention | Normalization | Input processing |
|---|---|---|---|---|
| C1 | Fixed sinusoidal | MHA | LayerNorm | Shared learned binary BPE |
| C2 | RoPE | MHA | LayerNorm | Shared learned binary BPE |
| C3 | Fixed sinusoidal | GQA | LayerNorm | Shared learned binary BPE |
| C4 | Fixed sinusoidal | MHA | RMSNorm | Shared learned binary BPE |
| C5 | Fixed sinusoidal globally | MHA | LayerNorm | Raw bytes with BLT local encoder/decoder |

C2 applies RoPE independently to Q and K in encoder self-attention, decoder
self-attention, and decoder cross-attention. Cross-attention Q and K may have
different lengths; each is rotated with its own positional range. C2 does not
receive additive sinusoidal embeddings. The earlier C2 run without RoPE in
cross-attention is invalid and superseded by the final C2 run reported here.

## 3. Shared hyperparameters

| Hyperparameter | Value |
|---|---:|
| Global model dimension | 256 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Global attention heads | 8 |
| Feed-forward hidden dimension | 1,024 |
| Dropout | 0.1 |
| Batch size | 2 |
| Epochs | 20 |
| Optimizer | AdamW |
| Peak learning rate | 0.0003 |
| Minimum learning rate | 0.00001 |
| Warmup | Linear, first 5% of optimizer steps |
| Warmup steps | 2,000 |
| Post-warmup schedule | Cosine decay |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Numerical precision | FP16 mixed precision |
| Random seed | 42 |
| Maximum model sequence length | 4,096 |
| Maximum greedy decode length | 4,096 |
| DataLoader workers | 0 |
| Training/validation/test ratio | 0.8 / 0.1 / 0.1 |
| Training examples | 4,000 |
| Validation examples | 500 |
| Test examples | 500 |
| Training batches per epoch | 2,000 |
| Total optimizer steps | 40,000 |
| Gradient/loss objective | Token-level cross entropy, padding ignored |
| Decoding | Autoregressive greedy argmax |

All five runs trained on all 4,000 training examples. The short-looking C1
runtime is not evidence of early termination: its log contains all 20 epochs,
2,000 batches per epoch, and 40,000 total optimizer updates.

## 4. Configuration-specific hyperparameters

| Config | Query heads | KV heads | Source vocab | Target vocab | Parameters |
|---|---:|---:|---:|---:|---:|
| C1 | 8 | 8 | 1,025 | 131 | 7,703,427 |
| C2 | 8 | 8 | 1,025 | 131 | 7,703,427 |
| C3 | 8 | 2 | 1,025 | 131 | 6,519,171 |
| C4 | 8 | 8 | 1,025 | 131 | 7,697,795 |
| C5 | 8 global | 8 global | 259 | 259 | 8,200,707 |

C5-only local parameters:

| Hyperparameter | Value |
|---|---:|
| Patch size | 4 bytes |
| Local model dimension | 128 |
| Local heads | 4 |
| Local FFN dimension | 512 |
| Local layers | 1 |

The small C1/C4 parameter difference comes from RMSNorm having fewer learned
normalization parameters than LayerNorm. C3 is smaller because GQA shares K/V
heads. C5 is larger because it adds local byte encoders and a local decoder.

## 5. Dataset and tokenization

The aligned corpus contains 5,000 plaintext/ciphertext pairs. Ciphertext lines
contain only `0` and `1`, and each ciphertext has exactly eight bits per
plaintext byte. The split is produced once using seed 42, so all configurations
receive identical examples in each partition.

### 5.1 C1-C4 learned-tokenizer lengths

| Split | Examples | Batches | Source min/max | Target min/max |
|---|---:|---:|---:|---:|
| Train | 4,000 | 2,000 | 16 / 1,938 | 23 / 2,672 |
| Validation | 500 | 250 | 19 / 1,697 | 27 / 2,420 |
| Test | 500 | 250 | 20 / 1,361 | 31 / 1,876 |

The shared binary BPE tokenizer begins with individual `0` and `1` symbols and
contains 1,022 learned variable-length merge rules. This gives 1,024 content
tokens; adding the source padding ID produces the recorded source vocabulary of
1,025. The tokenizer was trained only on the training split and then frozen for
C1-C4. The decoder uses 128 ASCII IDs plus PAD, BOS, and EOS, for 131 IDs.

### 5.2 C5 raw-byte lengths

| Split | Examples | Batches | Source min/max | Target min/max |
|---|---:|---:|---:|---:|
| Train | 4,000 | 2,000 | 21 / 2,670 | 23 / 2,672 |
| Validation | 500 | 250 | 25 / 2,418 | 27 / 2,420 |
| Test | 500 | 250 | 29 / 1,874 | 31 / 1,876 |

C5 uses byte IDs 0-255 and PAD/BOS/EOS IDs 256-258. Its global Transformer sees
pooled four-byte patches, so the effective global-attention sequence is roughly
one quarter of the raw-byte length. C5 does not use the BPE merge rules.

## 6. Evaluation protocol and metric definitions

The checkpoint with minimum validation loss is loaded after training. Every
test source is then decoded autoregressively from BOS. At each step, the model's
highest-probability output is appended. Decoding stops at EOS or at 4,096 output
positions. No teacher forcing or beam search is used for the reported sequence
metrics.

- **Bit-level accuracy:** decoded prediction and target strings are converted
  to bytes and then bits. Matching bit positions are counted; missing or extra
  tail bits are penalized through the maximum prediction/target length.
- **Sequence accuracy:** fraction of the 500 predictions exactly equal to the
  complete target string.
- **Levenshtein distance:** character-level insertions, deletions, and
  substitutions required to transform prediction into target, averaged across
  the test set.
- **BLEU:** word-tokenized corpus BLEU-4 with uniform 1-4-gram weights, clipped
  counts, brevity penalty, and add-one smoothing. Reported only for C1-C4.
- **ROUGE-1/2:** mean per-example unigram/bigram overlap F1 after whitespace
  tokenization. Reported only for C1-C4.
- **ROUGE-L:** mean per-example longest-common-subsequence F1 over whitespace
  tokens. Reported only for C1-C4.
- **Test token accuracy:** teacher-forced next-token accuracy. This diagnoses
  the conditional model but is not the assignment's greedy sequence metric.
- **Test loss:** teacher-forced cross-entropy on non-padding target positions.

BLEU and ROUGE are intentionally absent for C5 because the assignment requests
them for tokenized models. Bit accuracy, exact sequence accuracy, and edit
distance are directly based on decoded strings and are shared across C1-C5.

## 7. Final greedy-decoding test results

| Config | Bit accuracy ↑ | Exact sequence ↑ | Mean edit distance ↓ | BLEU ↑ | ROUGE-1 F1 ↑ | ROUGE-2 F1 ↑ | ROUGE-L F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 66.9379% | 0.20% (1/500) | 228.092 | 0.247765 | 0.577761 | 0.358195 | 0.566949 |
| C2 | **68.8562%** | **0.40% (2/500)** | **143.894** | **0.413740** | **0.677726** | **0.497954** | **0.673646** |
| C3 | 61.8402% | 0.20% (1/500) | 315.952 | 0.127038 | 0.481853 | 0.230759 | 0.463845 |
| C4 | 64.9490% | 0.20% (1/500) | 245.870 | 0.226044 | 0.565474 | 0.339020 | 0.554313 |
| C5 | **91.6518% overall** | 0.00% (0/500) | **103.882 overall** | N/A | N/A | N/A | N/A |

Bold values within C2 indicate the best C1-C4/tokenized result. C5 has the best
cross-configuration bit accuracy and edit distance.

## 8. Teacher-forced validation and test results

| Config | Best epoch | Best validation loss | Validation accuracy | Test loss | Test token accuracy |
|---|---:|---:|---:|---:|---:|
| C1 | 20 | 0.4156 | 87.80% | 0.417948 | 87.7397% |
| C2 | 18 | **0.328091** | **90.9194%** | **0.331955** | **90.6972%** |
| C3 | 20 | 0.6629 | 80.54% | 0.662453 | 80.5547% |
| C4 | 20 | 0.4628 | 86.44% | 0.465674 | 86.2866% |
| C5 | 20 | 0.2469 | 92.01% | 0.245980 | 92.0368% |

C5 loss is not strictly comparable with C1-C4 loss because it predicts over a
different vocabulary. Within C1-C4, C2 is strongest. Validation and test values
are close for every model, which suggests the seeded validation and test splits
have similar difficulty and that gross validation overfitting is absent.

## 9. Full epoch-by-epoch training history

Each cell is `training loss / validation loss / validation token accuracy`.

| Epoch | C1 | C2 | C3 | C4 | C5 |
|---:|---|---|---|---|---|
| 1 | 2.6483 / 2.4627 / 27.01% | 2.4448 / 1.9667 / 41.87% | 2.6584 / 2.4603 / 27.10% | 2.6457 / 2.4615 / 27.12% | 2.7529 / 2.2910 / 32.66% |
| 2 | 2.4217 / 2.3193 / 31.06% | 1.8331 / 1.6388 / 51.48% | 2.4221 / 2.3202 / 30.90% | 2.4154 / 2.3080 / 31.41% | 2.2590 / 2.1425 / 36.50% |
| 3 | 2.2402 / 2.0353 / 39.33% | 1.6049 / 1.4138 / 58.58% | 2.2592 / 2.0810 / 37.79% | 2.2154 / 2.0026 / 40.37% | 2.1680 / 2.0602 / 38.88% |
| 4 | 1.9636 / 1.7028 / 49.76% | 1.3733 / 1.1808 / 65.56% | 2.0389 / 1.8155 / 46.43% | 1.9411 / 1.6982 / 49.94% | 2.0906 / 1.9785 / 41.39% |
| 5 | 1.6608 / 1.4182 / 58.42% | 1.1989 / 1.0461 / 69.61% | 1.7744 / 1.5271 / 55.37% | 1.6599 / 1.4089 / 58.80% | 2.0115 / 1.8448 / 44.70% |
| 6 | 1.4313 / 1.2266 / 63.98% | 1.0755 / 0.9375 / 72.82% | 1.5395 / 1.3051 / 61.90% | 1.4328 / 1.2095 / 64.66% | 1.8127 / 1.4563 / 55.21% |
| 7 | 1.2684 / 1.0631 / 68.79% | 0.9861 / 0.8756 / 74.60% | 1.3711 / 1.1757 / 65.65% | 1.2706 / 1.0742 / 68.52% | 1.5171 / 1.1866 / 63.11% |
| 8 | 1.1460 / 0.9498 / 72.12% | 0.9069 / 0.8166 / 76.34% | 1.2497 / 1.0303 / 69.97% | 1.1510 / 0.9581 / 71.96% | 1.3035 / 0.9594 / 69.69% |
| 9 | 1.0489 / 0.8708 / 74.39% | 0.8313 / 0.7558 / 78.04% | 1.1570 / 0.9898 / 71.16% | 1.0602 / 0.8569 / 74.86% | 1.1015 / 0.7330 / 76.77% |
| 10 | 0.9584 / 0.7626 / 77.62% | 0.7379 / 0.6634 / 80.96% | 1.0804 / 0.9129 / 73.37% | 0.9719 / 0.8336 / 75.45% | 0.9162 / 0.5660 / 81.95% |
| 11 | 0.8785 / 0.6985 / 79.40% | 0.6627 / 0.5252 / 84.84% | 1.0145 / 0.8575 / 74.96% | 0.9003 / 0.7133 / 79.03% | 0.7886 / 0.4678 / 85.08% |
| 12 | 0.8084 / 0.6323 / 81.40% | 0.5924 / 0.4888 / 86.01% | 0.9568 / 0.7800 / 77.16% | 0.8367 / 0.6631 / 80.51% | 0.7045 / 0.4012 / 87.17% |
| 13 | 0.7421 / 0.5650 / 83.31% | 0.5249 / 0.3973 / 88.61% | 0.9068 / 0.8427 / 75.47% | 0.7758 / 0.6089 / 82.08% | 0.6392 / 0.3554 / 88.59% |
| 14 | 0.6900 / 0.5311 / 84.36% | 0.4784 / 0.4109 / 88.38% | 0.8618 / 0.6980 / 79.59% | 0.7245 / 0.5656 / 83.36% | 0.5891 / 0.3194 / 89.76% |
| 15 | 0.6446 / 0.4914 / 85.55% | 0.4355 / 0.3959 / 88.95% | 0.8262 / 0.7965 / 76.79% | 0.6833 / 0.5445 / 83.94% | 0.5497 / 0.2941 / 90.50% |
| 16 | 0.6058 / 0.4766 / 85.98% | 0.4031 / 0.3593 / 89.99% | 0.7941 / 0.7280 / 78.72% | 0.6491 / 0.5266 / 84.47% | 0.5212 / 0.2761 / 91.12% |
| 17 | 0.5763 / 0.4389 / 87.06% | 0.3781 / 0.3594 / 90.11% | 0.7704 / 0.6690 / 80.38% | 0.6233 / 0.4881 / 85.67% | 0.4995 / 0.2637 / 91.47% |
| 18 | 0.5544 / 0.4305 / 87.33% | 0.3601 / **0.3281 / 90.92%** | 0.7519 / 0.6739 / 80.22% | 0.6028 / 0.4793 / 85.93% | 0.4856 / 0.2562 / 91.71% |
| 19 | 0.5391 / 0.4205 / 87.63% | 0.3486 / 0.3745 / 89.90% | 0.7386 / 0.6711 / 80.34% | 0.5892 / 0.4724 / 86.13% | 0.4763 / 0.2501 / 91.91% |
| 20 | 0.5292 / **0.4156 / 87.80%** | 0.3410 / 0.3406 / 90.69% | 0.7310 / **0.6629 / 80.54%** | 0.5815 / **0.4628 / 86.44%** | 0.4696 / **0.2469 / 92.01%** |

C1, C3, C4, and C5 select epoch 20. C2 selects epoch 18; its validation
loss rises at epochs 19-20, so evaluating the final weights rather than the
best-validation checkpoint would unfairly lower its result.

## 10. Efficiency results

| Config | Parameters | Training time | Peak allocated GPU memory | Final-epoch throughput |
|---|---:|---:|---:|---:|
| C1 | 7.703M | 3,569.50 s (59.49 min) | 8,930.47 MB | 14,141 tokens/s |
| C2 | 7.703M | 3,221.89 s (53.70 min) | 8,927.35 MB | 15,623 tokens/s |
| C3 | 6.519M | 3,046.09 s (50.77 min) | 8,911.84 MB | 16,420 tokens/s |
| C4 | 7.698M | 2,631.63 s (43.86 min) | 8,828.52 MB | 19,202 tokens/s |
| C5 | 8.201M | 3,898.56 s (64.98 min) | 1,080.94 MB | 10,958 byte positions/s |

C3 contains about 15.4% fewer parameters than C1 due to grouped K/V heads.
C4 was fastest in this set of runs. Wall-clock and throughput comparisons can
also reflect GPU/node conditions and therefore should not be treated as pure
architectural measurements without repeated controlled timing runs.

C5's throughput unit is not directly comparable to C1-C4 because its input is
raw bytes and its global network sees pooled patches. Its much smaller recorded
peak memory is consistent with reducing global-attention length by patching,
despite having the largest parameter count.

## 11. Ablation analysis

### 11.1 C2: sinusoidal positions to RoPE

Relative to C1, corrected C2 improves:

- bit accuracy by 1.9183 absolute percentage points;
- exact reconstruction from 1 to 2 test sequences;
- mean edit distance by 84.198 edits (36.9% lower);
- BLEU from 0.247765 to 0.413740;
- ROUGE-L from 0.566949 to 0.673646; and
- teacher-forced test token accuracy by 2.9575 percentage points.

This is a coherent improvement across every quality metric. It supports the
conclusion that RoPE is beneficial for this encrypted-sequence task **when it
is applied consistently to encoder self-attention, decoder self-attention, and
decoder cross-attention**. The earlier incomplete cross-attention integration
produced a much worse C2 result and must not appear in final comparisons.

### 11.2 C3: MHA to GQA

Relative to C1, C3 reduces parameters from 7.703M to 6.519M and improves
throughput, but bit accuracy falls by 5.0977 percentage points, mean edit
distance rises by 87.860, and BLEU nearly halves. This is a conventional
efficiency-quality tradeoff: shared K/V heads reduce capacity and computation.
The C3 validation curve is noisier than the other curves, with regressions at
epochs 13 and 15, but it ends at its best validation loss.

### 11.3 C4: LayerNorm to RMSNorm

C4 stays close to C1. Bit accuracy falls by only 1.9889 percentage points,
ROUGE-L changes from 0.566949 to 0.554313, and mean edit distance increases by
17.778. It records the highest training throughput and shortest wall-clock
training time. These results suggest RMSNorm is a credible lighter alternative
with a modest quality cost in this setup.

### 11.4 C5: learned BPE to BLT/raw bytes

C5 improves over the best tokenized model C2 by 22.7956 absolute bit-accuracy
points and lowers mean edit distance from 143.894 to 103.882. This strongly
supports byte-local modeling for byte-aligned encrypted data. C5 nevertheless
has zero exact reconstructions, compared with two for C2. That is not a
contradiction: C5 can make fewer errors overall while distributing at least one
error across every long sequence.

C5 is not a one-line tokenizer substitution. The required BLT design adds local
encoders/decoder, patch pooling, a different vocabulary, and a different
effective global sequence length. Its result should be described as the full
token-free/BLT configuration rather than attributing the gain solely to removal
of BPE.

## 12. Why exact sequence accuracy is near zero

Test targets range from 31 to 1,876 decoder IDs. Exact sequence accuracy is an
all-or-nothing metric. Even a model with independent 92% per-position accuracy
would have an extremely small probability of making no error over hundreds of
positions. Errors are not truly independent, but the calculation illustrates
why high token or bit accuracy can coexist with almost zero exact sequence
accuracy. The exact metric remains important because it measures complete
reconstruction, but it should be interpreted together with bit accuracy and
edit distance.

## 13. Validity, caveats, and limitations

1. **One seed:** all results use seed 42. Differences are observed outcomes,
   not confidence intervals. Multiple seeds would be needed for formal claims
   of statistical significance.
2. **Fixed training budget:** four configurations achieve their best validation
   loss at epoch 20 and may improve with longer training. The comparison is fair
   under an equal 20-epoch/40,000-step budget, not necessarily at convergence.
3. **Teacher forcing versus generation:** test loss and token accuracy condition
   on correct previous tokens. Greedy decoding conditions on the model's own
   earlier predictions and accumulates errors. Greedy metrics are therefore the
   primary assignment results.
4. **Different vocabularies:** C5 cross-entropy is not directly comparable to
   C1-C4 cross-entropy because the class spaces have sizes 259 and 131.
5. **Different throughput units:** C5 processes raw byte positions while C1-C4
   use learned source tokens and plaintext decoder IDs.
6. **BLEU/ROUGE tokenization:** these scores use whitespace-delimited words.
   Alternate tokenization, stemming, smoothing, or library defaults can produce
   different absolute scores. The same implementation is used for C1-C4.
7. **Sequence length weighting:** token loss/accuracy weight individual target
   positions, while BLEU/ROUGE/edit distance are aggregated per sequence or
   corpus as defined above. Long sequences therefore influence metrics in
   different ways.
8. **Timing:** jobs can run on different physical GPUs/nodes even within the
   same partition. Architectural timing claims should remain descriptive.
9. **No beam search:** all sequence results intentionally use greedy decoding,
   satisfying the assignment and keeping C1-C5 comparable.
10. **No test-set model selection:** validation loss chooses the checkpoint.
    The test set is evaluated only after training and does not affect selection.

## 14. Reproducibility and artifacts

### Final metrics

- `outputs/metrics/C1_test_metrics.json`
- `outputs/metrics/C2_test_metrics.json`
- `outputs/metrics/C3_test_metrics.json`
- `outputs/metrics/C4_test_metrics.json`
- `outputs/metrics/C5_test_metrics.json`

Each JSON records the full resolved configuration, vocabulary sizes, parameter
count, data-split statistics, chosen checkpoint epoch, evaluation protocol, and
all applicable final metrics.

### Checkpoints

- `outputs/checkpoints/C1/best.pt` and `final.pt`
- `outputs/checkpoints/C2/best.pt` and `final.pt`
- `outputs/checkpoints/C3/best.pt` and `final.pt`
- `outputs/checkpoints/C4/best.pt` and `final.pt`
- `outputs/checkpoints/C5/best.pt` and `final.pt`

The `best.pt` files are the proper evaluation checkpoints. `final.pt` stores
epoch-20 weights, which differ from the selected checkpoint for C2.

### WandB runs

| Config | Run ID | URL |
|---|---|---|
| C1 | `gau3ctor` | https://wandb.ai/manasagrawal206-iiit-hyderabad/anlp-assignment-1/runs/gau3ctor |
| C2 corrected | `apdcwnn6` | https://wandb.ai/manasagrawal206-iiit-hyderabad/anlp-assignment-1/runs/apdcwnn6 |
| C3 | `b0axlr97` | https://wandb.ai/manasagrawal206-iiit-hyderabad/anlp-assignment-1/runs/b0axlr97 |
| C4 | `g6y614rp` | https://wandb.ai/manasagrawal206-iiit-hyderabad/anlp-assignment-1/runs/g6y614rp |
| C5 | `clawcyb1` | https://wandb.ai/manasagrawal206-iiit-hyderabad/anlp-assignment-1/runs/clawcyb1 |

### Logs and launch scripts

- Logs: `c1_log.txt` through `c5_log.txt`
- Launchers: `train_c1.sh` through `train_c5.sh`
- Shared tokenizer: `outputs/merge_rules.json`

All five logs end with successful completion messages and contain final test
metrics. The corrected C2 checkpoint contains RoPE buffers under decoder
cross-attention, providing a direct artifact-level check that the fixed model
was actually trained.

## 15. Final conclusions

Under a controlled 20-epoch budget, C5 gives the strongest bitwise
reconstruction and lowest average edit distance, demonstrating the value of
byte-local modeling for this task. Among learned-tokenizer configurations, C2
is best: correctly integrated RoPE improves all quality metrics over the
sinusoidal C1 baseline. C4 retains most C1 quality and is the fastest observed
run, while C3 reduces parameter count and improves throughput at a noticeable
quality cost. Exact full-sequence reconstruction remains unsolved by all five
models, emphasizing that high local accuracy does not guarantee error-free
generation over long encrypted sequences.
