Assignment Context

ANLP Assignment 1: build an encoder-decoder Transformer from scratch in PyTorch and compare C1–C5.

Configurations

C1: Sinusoidal + MHA + LayerNorm + learned subword tokenizer

C2: RoPE + MHA + LayerNorm + learned subword tokenizer

C3: Sinusoidal + GQA + LayerNorm + learned subword tokenizer

C4: Sinusoidal + MHA + RMSNorm + learned subword tokenizer

C5: Sinusoidal + MHA + LayerNorm + BLT/token-free byte processing

C2–C5 must each change only one component relative to C1. Keep shared hyperparameters consistent where applicable.

Hard Requirements

Use Python + PyTorch.

Implement core Transformer components from scratch.

Do not use nn.Transformer, nn.MultiheadAttention, or equivalent ready-made Transformer blocks.

Implement MHA, GQA, sinusoidal positions, RoPE, LayerNorm/Pre-LN as required, RMSNorm, FFN, and BLT local encoder/decoder modules.

C1–C4 ciphertext must use a learned subword tokenizer.

Fixed-width chunking such as 8 bits = 1 token is not allowed as the tokenizer.

Tokenizer must be implemented from scratch; do not use a prebuilt BPE/SentencePiece tokenizer library.

Standard libraries are allowed for BLEU, ROUGE, and Levenshtein, but understand how they work.

Evaluate with greedy decoding.

Log runs with WandB.

Host pretrained model checkpoints on Hugging Face.

Final submission is a single <rollnumber>_assignment1.zip on Moodle, not a GitHub submission.

Current deadline: 3 Sep 2026, 10:59 AM IST.

Project Decisions

Use one tokenizer shared by C1–C4 so tokenization is not another changing variable.

Train the tokenizer on the training split only, then freeze it before model training.

For our implementation, start BPE from individual binary symbols (0, 1) and learn variable-length bit-pattern merges.

Do not use 8-bit byte chunks as the basis/final tokens for C1–C4.

Save tokenizer state as:

outputs/tokenizer/vocab.json

outputs/tokenizer/merges.json

Also upload vocab.json and merges.json alongside HF checkpoints for reproducibility. This is our choice, not an explicit assignment requirement.

C5 does not use the C1–C4 tokenizer; it follows the BLT/raw-byte path.

Use a private GitHub repo only for development/syncing across HPC accounts.

Train configurations in parallel on different authorized HPC accounts when possible.

Suggested Training Flow

Split dataset into train/val/test.

Train tokenizer once on train only.

Save/freeze tokenizer artifacts.

Sync code + tokenizer artifacts to all HPC environments.

Train C1–C4 using the exact same tokenizer.

Train C5 separately with BLT.

Collect metrics, WandB runs, checkpoints, plots, and report results.

Build the required folder structure and submit the ZIP to Moodle.

File-Structure Guidance

Keep close to the assignment structure:

src/models/attention.py

src/models/positional.py

src/models/norm.py

src/models/blt.py

src/dataset.py

src/train.py

src/utils.py

outputs/

README.md

Report.pdf

Prefer putting tokenizer implementation/loading logic in src/dataset.py and learned tokenizer artifacts under outputs/tokenizer/.

Do Not

Do not retrain a separate tokenizer for C1, C2, C3, or C4.

Do not use val/test data to learn tokenizer merges.

Do not use fixed 8-bit chunks as C1–C4 Transformer tokens.

Do not use prebuilt tokenizer implementations.

Do not change extra architectural/hyperparameter choices between ablation configs without necessity.
