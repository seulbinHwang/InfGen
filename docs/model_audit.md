# Model and loss audit

Target: **InfGen**, Yang, Tan and Krähenbühl, *Long-term Traffic Simulation with
Interleaved Autoregressive Motion and Scenario Generation* (ICCV 2025),
[paper](https://arxiv.org/abs/2506.17213). This is distinct from UCLA's renamed
SceneStreamer. The implementation starts from official upstream commit
`68f643c0cd464e908fac0ef79f82d55b5e6edc90`.

## Paper profile versus released defaults

`Model.paper_objective: true` selects the audited paper profile. The default is
false to retain the released model's training choices. The paper profile uses a
single-linear fusion layer and a three-linear shape head in accordance with the
paper; the released profile has a three-linear fusion and two-linear shape head.
Their checkpoints therefore differ for these heads. Full checkpoint loading must
use the checkpoint's saved profile; official released weights can initialize only
matching parameters through an explicit partial pretrain load. Neither loading
path establishes equal outputs or reproduced benchmark scores.

| Topic | Paper evidence | Implementation |
|---|---|---|
| Objective | Eq. 7–8: motion, pose, controls, attributes | Plain cross entropy without released label smoothing; unweighted type CE; configured scalar and control-class weights. Offset, map-token and occupancy auxiliary losses do not enter the paper objective. |
| Fusion and shape heads | Appendix B.1 Eq. 9: one-layer fusion; B.2: three-layer shape MLP | Paper mode uses one Linear fusion and a three-Linear shape MLP with 128-wide hidden layers and ReLU. Hidden widths and activations are explicit implementation choices. |
| Attention order | Eq. 3–6: temporal, agent, map; scene grid, agent, map | The paper profile applies agent attention before map attention throughout training and inference. Released default remains map before agent. |
| Query context | Table 4: query–agent 10 m; query–map 75 m for position, 10 m for heading | Coarse and refined agent attention use 10 m in paper mode. The occupancy grid independently includes the allowed prefix over the full 75 m region. |
| Spatial order | Sec. 4.4: insert from near to far | Paper mode sorts entering agents by distance; released default sorts by angle. |
| Temporal controls | Appendix D.1: exclude EOS and later NULL supervision | Paper masks exclude EOS and its suffix; legacy masks retain the released NULL-suffix objective. |
| Pose and attributes | Appendix B.2: sample position; highest-probability heading and type | Position retains top-K sampling; heading and type use argmax. Paper mode omits the released learned offset correction and grid-overlap rejection rule. |
| Heading quantization | Sec. 4.1: the closest heading interval | Paper mode uses nearest circular representatives at -180 + 3k degrees, rounded modulo 120. This makes the choice of representatives explicit; legacy mode retains floor-based interval assignment. |
| Controls at rollout | Appendix B.2 describes categorical sampling alongside pose | Paper mode samples controls at temperature 1. Active agents choose KEEP or REMOVE, excluding the NULL placeholder. Legacy uses argmax. |
| Insertion length | Appendix D.2: ten positions supervised, generation can continue to 128 agents | The paper profile can insert more than ten agents in a step, stopping on BEGIN_MOTION or the live-agent capacity. Retired IDs remain archived as output rows and do not consume live capacity. |
| Static map input | Sec. 4.1: discretized map context | Paper inference chooses nearest map vocabulary entries. Training retains the released random choice among eight nearest entries as an explicit upstream augmentation choice. |

The released motion top-K value 5, position top-K value 10, categorical temperature
1 and the exact discrete grid construction are implementation choices rather
than verified universal paper requirements. The released grid has **1,961**
positions with radius 75 m and 3 m spacing, whereas paper Table 4 lists 1,849.
The actual grid and corresponding vocabulary tensor shapes are preserved. The
fusion and shape-head layer counts are corrected specifically in the paper
profile, without claiming that every unspecified architectural detail has been
independently reconstructed.

## Correctness fixes applying to both profiles

- Empty masked CE, MSE and BCE supervision contributes a differentiable zero.
  Nonfinite losses on nonempty supervision are not silently replaced by zero.
- Spatial causal masks convert local per-scene ordering to global batch indices.
  Prefix occupancy is built per scene and prefix, preventing other scenes or
  later insertions from appearing in a query's occupancy context.
- Native radius searches sort scene-batch vectors before searching and map
  resulting indices back to original rows, including interleaved seed rows.
  Batched map-token polygon IDs receive their scene's global polygon offset.
- Training-only seed placeholders are excluded from temporal attention during
  training. Inference has no placeholder rows at that stage; all real agents
  retain their own causal history rather than losing the final ten rows.
- Invalid headings are replaced by the declared constant before vector encoding,
  without mutating the caller's tensor. Valid-to-invalid relative-heading
  transitions use the source/target condition rather than an impossible condition.
- Shared occupancy grids broadcast over multiple ego headings. Heading indices
  remain within vocabulary bounds at the angle wrap boundary.
- The `disable_insertion` spelling is corrected. CPU execution avoids CUDA-only
  autocast/memory reporting, and filtered inference uses the actual retained-agent
  count. Requested horizons shorter than the log are truncated consistently.
- An agent's REMOVE control is terminal for that ID. Its decoded motion at the
  removal transition is discarded, the EXIT marker is retained, and subsequent
  states are NULL. Its motion and interaction context are masked and it cannot
  reappear through a later KEEP prediction. A new ADD operation receives a new ID.
- Inference consumes `observed_valid_mask` and `history_state_idx`, constructed
  from the observed prefix by the data pipeline. For legacy data, its last
  observed REMOVE is right-censored to KEEP, so future disappearance labels cannot
  reveal future visibility. Historical exported controls use the same causal seed.
- Dynamic inference explicitly supports one scene per batch. Multi-rollout export
  refuses incompatible agent-ID layouts instead of silently stacking mismatched
  rows. Checkpoint pretraining refuses a load with zero matched trainable tensors.
- Core model import and offline rollout export do not load TensorFlow/Waymo
  scientific metrics. Those dependencies are loaded when online metrics are used.

## Verification and limits

`tests/test_model_tokenizer.py` checks rotated-grid broadcasting, position round
trips, empty input and angle bounds. `tests/test_model_masks.py` covers empty-loss
gradients, weighted losses, paper attention order, temporal causality, spatial
prefixes across scenes, scene-separated occupancy, causal history and terminal
removal, and native search on unsorted scene batches. `tests/test_model_rollout.py`
checks paper fusion/shape layer counts and runs the actual decoder on synthetic data
with forced ADD/KEEP/REMOVE controls, including more than ten insertions in one
step, capacity reuse after removal, archived IDs and a short rollout horizon.

These are functional and causality checks, not scientific evaluations. Synthetic
data and short training runs do not establish the paper's realism, robustness or
WOSAC/long-horizon metric results. Root-level reproduction verification separately
exercises training, checkpoint resume and deterministic offline rollout export.
