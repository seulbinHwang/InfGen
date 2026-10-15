# Paper audit and reproduction decisions

This repository implements **UT Austin InfGen**, by Xiuyu Yang, Shuhan Tan, and
Philipp Krahenbuhl: *Long-term Traffic Simulation with Interleaved Autoregressive
Motion and Scenario Generation*. It is independent of UCLA's former InfGen,
renamed SceneStreamer. The authoritative configuration is
[`configs/paper_reproduction.yaml`](../configs/paper_reproduction.yaml).

The audit used the supplied 20-page
[project preprint PDF](https://orangesodahub.github.io/InfGen/assets/infgen_preprint.pdf),
including its supplement. Its SHA-256 is
`ea0cd73fd9b91e186aadff0e963f6ccbcae1949016b7898e5901773328ad994e`.
Page numbers below count from the PDF's first page. Public primary sources:
[paper record](https://arxiv.org/abs/2506.17213),
[paper HTML](https://arxiv.org/html/2506.17213v2), and
[authors' project](https://orangesodahub.github.io/InfGen/).
The supplied PDF, rather than another paper called InfGen, governs this profile.
Upstream baseline: `68f643c0cd464e908fac0ef79f82d55b5e6edc90`.

## Algorithm checklist

| Requirement | Evidence | Implementation interpretation |
| --- | --- | --- |
| A 1.1 s logged seed and static map condition future traffic. | Sections 3, 5; pp. 3, 5-6 | 11 historical frames; training uses 80 future frames at 10 Hz. |
| Motion simulation and scene generation alternate. | Eq. 2, Section 4.2; pp. 3-5 | Predict active agents in parallel, then add agents one at a time until BEGIN MOTION. |
| Agent population changes through ADD, KEEP, REMOVE, BEGIN MOTION. | Section 4.1; p. 4 | Existing state aliases are retained; `invalid` in the spatial head means stop generation. |
| Each insertion conditions on existing agents and earlier insertions. | Section 4.2; pp. 4-5 | Hybrid attention allows existing context and generated prefix, never future insertions. |
| Motion tokens cover 0.5 s and are learned by k-disks clustering. | Section 4.1; p. 3 | Use the released SMART-derived 2048-entry agent vocabulary, not an invented kinematic vocabulary. |
| Agent poses use ego-centered position and heading tokens. | Section 4.1, Table 4; pp. 3, 11 | Released circular 75 m grid, 3 m spacing; 120 bins at 3 degrees. Grid-count discrepancy below. |
| Motion attention uses own history, agents, and map. | Eqs. 3-5; p. 4 | Temporal attention is causal; current active-agent predictions are parallel. |
| Scene attention uses occupancy, agents, and map. | Eq. 6, Section 4.3; pp. 4-5 | Preserve occupancy feature construction even when auxiliary occupancy losses are disabled. |
| Agent features fuse motion, position, validity, and shape/type through a one-layer MLP. | Eq. 9 and Appendix B.1; pp. 11-12 | Paper mode uses one affine projection after concatenation; special query/invalid embeddings are retained. |
| Relative attention includes distance, relative direction, relative heading, and temporal delta. | Eqs. 10-12; p. 12 | Fourier embeddings; context must obey radius and causal masks. |
| New positions are sampled from top-K. Heading is decoded at that position. | Section B.2; p. 12 | Heading refinement uses agent/map attention, no occupancy attention. |
| Headings and types use the highest-probability token. | Section B.2; p. 12 | Argmax for heading and type; these are not extra stochastic motion samples. |
| New shape is continuous and predicted by a three-layer MLP. | Section B.2; p. 12 | Paper mode uses three linear layers with two ReLU activations; MSE and hidden widths are explicit implementation choices. |
| Training is teacher-forced and tasks train in parallel. | Sections 4.4, D; pp. 5, 13 | Interleaved autoregression is the inference procedure, not a claim that training must serially roll out. |
| Insertions are ordered near-to-far from ego. | Section 4.4, Appendix C; pp. 5, 13 | Distance sorting replaces the release's direction-based order in the paper profile. |
| A token is valid only when both 0.5 s endpoints are valid. | Eq. 13 and prose; p. 13 | Internal visibility must not be inferred from only one endpoint. |
| First/last valid token gives ADD/REMOVE; the final log token cannot imply removal. | Eq. 14 and exceptions; p. 13 | Initial valid tokens get ADD; a right-censored last token gets KEEP. |
| Loss masks align to predictions, not GT query tokens. | Appendix D.1 footnote, Eq. 17; pp. 13-14 | REMOVE and BEGIN MOTION are supervised from the preceding query; EOS queries are excluded. |
| Temporal class weights are KEEP 0.1, REMOVE 0.9. | Appendix D.1; p. 14 | Invalid targets are excluded; the retained three-class alias has zero invalid weight. |
| Spatial class weights are ADD 0.1, BEGIN MOTION 0.9. | Appendix D.2; p. 14 | The released spatial class order is stop, add, so its weight vector is `[0.9, 0.1]`. |
| Training uses 32 agents and 10 spatial queries; inference can grow to 128. | Appendix D.2; pp. 14-15 | `max_num=32`, ten seed features, `buffer_size=128`. |
| REMOVE is terminal for an agent identity and discards its next motion. | Section 4.2; pp. 4-5 | Retired rows are excluded from subsequent context and cannot return through a KEEP prediction. Rows are retained only for exporting past trajectories. |
| Short-term WOSAC fixes the population; long-term evaluation permits population changes. | Sections 5.1-5.2; p. 6 | 8 s motion-only versus 300 future 10 Hz frames / 30 s dynamic rollout. |

## Configuration and objective

Table 4 (p. 11) states width 128, eight heads of dimension 16, 64 Fourier bands,
six motion blocks, three scene blocks, and three map blocks. The scene block
count is already three in the released decoder; no unused configuration key is
introduced. The temporal window is 12 token steps. The release expresses this as
`time_span=60` raw frames and divides by its five-frame shift. Radii in meters
are agent-agent 60, map-agent 30, map-map 10, query-agent 10, and query-map 75 for
position / 10 for heading. The motion and map vocabulary sizes are 2048 and 1024.

Equation 8 (p. 5) and training settings (p. 6) give:

```text
loss = 1 * motion_CE + 10 * position_CE + 1 * heading_CE
     + 10 * (temporal_control_CE + spatial_control_CE)
     + 0.2 * shape_regression + 5 * type_CE
```

`paper_objective: true` selects ordinary unsmoothed categorical cross-entropy,
unweighted type CE, and omits offset-regression, map-token, and occupancy
auxiliary losses. The release's position coefficient is 1; the canonical
profile corrects it to the paper's 10. The release's temporal class weights
`[0.1, 0.1, 0.8]` differ from the stated KEEP/REMOVE weights. The profile uses
`[0, 0.1, 0.9]` in the existing invalid/KEEP/REMOVE class order. This zero weight
is not a substitute for validity/lifecycle masks.

The paper specifies AdamW with initial learning rate 0.0005, cosine annealing,
and total batch eight on eight NVIDIA A5000 GPUs. The configuration therefore
uses one training example per GPU and eight devices, with no accumulation.
The shared `run.py` CLI and `train.py` wrapper honor the YAML's eight devices
unless `--devices` is explicitly provided. Local smoke runs may explicitly
reduce the device count; that is not the paper's training batch.

`predict_map: false` does not remove the static map encoder. It disables the
extra map-token prediction task. The separate released
`configs/pretrain_scalable_map.yaml` recipe remains available. `predict_occ:
true` retains occupancy features and tensors required by scene generation;
zero auxiliary coefficients and `paper_objective` exclude those auxiliary
objectives. DDP permits unused parameters because inactive auxiliary heads are
still retained for compatibility.

The profile also changes two modules to the layer counts explicitly stated in
Appendix B: the concatenated feature fusion is one linear layer and the shape
head is three linear layers. The legacy release instead has three linear
layers in its fusion embedding and two in its shape head. Here, a "layer" is
interpreted as one affine/linear transformation. For the shape head the chosen
hidden widths are 128 and 128, separated by ReLU; neither these activations nor
the intermediate widths are specified by the paper.

These architecture changes alter parameter keys/shapes. Consequently an
official legacy checkpoint is **not** a strict-load checkpoint for this paper
profile. Use the matching legacy configuration for unmodified checkpoint
inference, or explicitly request partial initialization, inspect its reported
missing/incompatible parameters, and train the changed modules. `infer.py`
reconstructs the saved profile and requires a strict state-dictionary match;
partial initialization is not presented as a complete resume or a trained
paper-profile model.

## Explicit implementation choices and discrepancies

| Item | Evidence or discrepancy | Decision / limit |
| --- | --- | --- |
| Position vocabulary count | Table 4 says 1849, R=75 m, spacing=3 m. Released `Attr_Tokenizer` uses a 51-by-51 lattice clipped to the 75 m circle, giving **1961**, verified by instantiating it. | Preserve released coordinates and compatible head shapes. Do not silently crop to 1849 arbitrary cells or claim an exact match to that table entry. |
| Eq. 18 mask | The printed formula omits the offset between existing context and generated sequence, and its boundary disagrees with the prose. | Follow the prose: all existing agents plus a causal generated prefix, also applying scene/batch/radius restrictions. |
| Eq. 16 invalid geometry | It prints invalid-invalid as `-z_invalid`, with `z_invalid=-2`; taken literally that is +2. The released implementation's `invalid_motion_value` and `invalid_head_value` are -2. | Retain released -2 constants for invalid pairs; valid/invalid transition constants are +/-1. This resolves a paper/code sign discrepancy in favor of the reference implementation, rather than claiming that -2 is the printed equation's literal arithmetic. |
| Map and motion token construction | Section 4.1 and Appendix B delegate important details to SMART. | Use released codebooks and road encoder. Full independent tokenizer retraining is a separate data-dependent stage. |
| Fusion and shape layer counts | Appendix B.1 says one-layer fusion; Appendix B.2 says three-layer shape MLP. Released helpers implement three and two linear layers respectively. | Paper mode constructs one fusion projection and a three-linear-layer shape head. This deliberately changes checkpoint keys/shapes; the legacy profile preserves its original construction. |
| Heading quantization | Section 4.1 says the closest heading interval but does not define bin representatives, rounding, or ties. The release uses floor/bin membership. | Paper mode interprets each decoded angle `-180 + 3*i` degrees as a representative and rounds to the nearest one modulo 120; exact half-bin ties follow PyTorch round-to-even. This is an explicit interpretation, not an unambiguous correction stated by the paper. |
| AdamW beta/epsilon/weight decay | Not specified. Released optimizer omits explicit options. | Retain PyTorch AdamW defaults: betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`. Record PyTorch version with results. |
| Duration and schedule | Paper gives cosine but no epochs, warmup, minimum LR, or schedule unit. | Retain release's 32 epochs, zero warmup, epoch-based cosine to zero. These are choices, not measured facts from the paper. |
| Dropout | Not specified. | Retain released 0.1. |
| Regression criterion | Paper specifies continuous shape prediction without a loss formula. | Retain MSE. Offset regression is absent from Eq. 8 and disabled in this objective. |
| Control-loss reduction | Paper does not define how temporal/spatial losses are combined or normalized jointly. | Sum the two separately reduced weighted CE losses under coefficient 10, as in the released model. |
| Sampling K / temperature | Position top-K is stated without K. No numeric motion K or temperature is given. | Retain released position K=10, motion K=5, temperature 1. Heading/type remain argmax. |
| Control sampling | Section 4.2 says sample control; released scene control uses argmax. | Paper mode uses categorical controls at temperature 1. Active temporal controls are restricted to KEEP/REMOVE; spatial controls to BEGIN MOTION/ADD. NULL remains a training placeholder. The exact temperature is an explicit implementation choice. |
| Learned position offset | Released inference adds a learned sub-grid offset; paper B.2 describes grid-point placement and Eq. 8 has no offset loss. | Paper mode disables both the offset objective and its inference correction. A randomly initialized unused offset head cannot shift the sampled grid point. |
| Occupied-cell filtering | Release defaults to occupied-cell rejection; Appendix F says no explicit fine-grained regularization was imposed. | Paper mode disables the rejection heuristic. It does not promise collision-free placements. |
| Query-agent radius | Table 4 states 10 m. The release uses 75 m for coarse position-query agent attention and 10 m for heading refinement. | Paper mode applies 10 m to query-agent attention in both stages; position-query map context and occupancy still cover the 75 m scene. |
| Attention order | Eqs. 3-6 describe temporal/grid, then agent attention, then map attention. The release applies map attention before agent attention. | Paper mode follows temporal/grid -> agent -> map in training and inference. The legacy profile retains its released order. |
| Occupancy feature shape | Section 4.3 describes an `[Np,D]` occupancy feature matrix, while the released `seed_agent_occ_embed` compresses the entire binary `[Np]` grid to one `[D]` token per query. Eq. 6 permits an efficiency transformation `Gamma`. | Retain the reference implementation's compressed occupancy token as an interpretation of `Gamma`; do not claim it is literally the stated per-cell feature matrix. This remains a documented representational ambiguity. |
| Inference capacity | Appendix D.2 permits more than ten spatial generations and caps inference at 128; the release hardcodes ten insertions per step. | Paper mode permits up to 128 active agents and stops on BEGIN MOTION or available capacity. The trajectory archive can contain more than 128 identities over time because retired rows are kept for output only. |
| Seed lifecycle censoring | Full-log REMOVE labels and backward extrapolation from future observations can reveal future visibility at the seed boundary. | Training retains full-log supervision. Inference uses original observed visibility and separately constructed history-only states, right-censors the last historical control to KEEP, and clears future generated-state inputs. It cannot admit an agent solely because extrapolation made it appear at the current frame. |
| Input jitter | Release perturbs map-codebook matching among eight nearest candidates, including in evaluation. | Retain this undocumented augmentation during training. Paper-mode evaluation uses nearest map-codebook matching; other legacy behavior remains available outside the profile. |
| Dataset filtering | Release filters to at least eight agents and fewer than the buffer size; training selects 32. | With generated metadata, preserve the release's scenario filter. Without metadata the portable loader warns and loads all scenario pickles; generate metadata for the declared experiment rather than silently mixing these datasets. |
| Seed and resource settings | Seed 2024, one loader worker, pinning, precision 32, and gradient clip 0.5 come from release/entrypoints. | Reproducibility/resource choices; not stated paper hyperparameters. Debug overfit duration 6000 is not used for the paper recipe. |
| Validation sample count | Default one rollout is convenient for development. | It does not establish official WOSAC or paper performance. Use a separately declared evaluation protocol. |
| Initial velocity | No separate new-agent velocity head is specified; Appendix F identifies stationary insertions as a failure. | Infer subsequent motion using the motion model. Do not claim an extra velocity head reproduces the paper. |

The audit deliberately distinguishes an executable scientific implementation
from a demonstrated reproduction of numerical results. Official benchmark
claims require WOMD, sufficient training compute, a trained checkpoint, and the
declared sampling/evaluation protocol. The main Table 2 and supplementary
all-components Table 5 also contain different reported scores; tests do not
assert either as an achievable local target.

## Evaluation and known limits

Section 5.2 uses sliding eight-second windows over long rollouts and compares
agents to a distribution learned over the validation set, because generated
agents have no one-to-one logged counterparts. Placement statistics are entry
count, exit count, and their distances to ego. Mean Agent Count Error and its
growth slope supplement these statistics. A short synthetic rollout is not a
substitute for this evaluation. The sliding-window interval and some histogram
details are not completely specified in the PDF; any evaluation must state its
chosen settings and provenance.

Appendix F reports boundary placements, incorrectly stationary new agents,
and poorly modeled visibility flicker. Agents visible for less than 0.5 s are
discarded; sub-token validity patterns are not modeled. The 30 s rollout can
already exhaust WOMD map coverage, and durations above five minutes were not
evaluated. No map generation or reinforcement-learning training is part of
this reproduction.
