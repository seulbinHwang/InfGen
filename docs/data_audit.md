# Data and tokenization audit

Source: the requested [InfGen preprint](https://orangesodahub.github.io/InfGen/assets/infgen_preprint.pdf), especially Appendix C (Eqs. 13–16), Appendix B, and Appendix D. This audit is against the official OrangeSodahub/InfGen implementation at commit `68f643c`. A passing synthetic regression test is evidence about executable behavior, not a claim that published WOMD scores have been reproduced.

## Facts specified by the paper

| Paper statement | Implementation |
| --- | --- |
| Appendix C: a WOMD record has 91 frames at 10 Hz; temporal stride δ = 5 (0.5 s), producing 18 tokens. | `TokenProcessor.shift = 5`; token matching uses endpoints 0→5, 5→10, …, 85→90. |
| Appendix C: a segment is valid iff both its start and end frames are valid. | Endpoint conjunction of the unfolded raw validity mask. |
| Eq. 14: first valid segment is ADD; last valid segment is REMOVE; intermediate controls are KEEP. A REMOVE at the final token becomes KEEP. | `control_states_from_validity`; tracks with no valid segment remain invalid. |
| Appendix B: attributes cover vehicle, pedestrian, cyclist. | Raw preprocessing drops unsupported/unset object classes instead of creating an all-zero trajectory vocabulary for them. |
| Appendix C: when arranged spatially, ADD agents are ordered by distance to ego. | This is handled by the model decoder rather than by changing raw track IDs in preprocessing. |

## Inherited upstream choices that the paper does not fully specify

These are kept to avoid silently substituting a different experiment. They are not claimed as facts from the paper.

- Raw preprocessing retains tracks whose first-to-last valid span is at least 15 frames, interpolates missing interior positions and velocities, and unwraps/interpolates headings. Test split observations are left as supplied because future labels are withheld. Filtering is based on span, not the number of observed frames.
- First observations are extrapolated back to the previous five-frame boundary with constant observed velocity. Agents first appearing at frame 10 may be extended to frame 5. Heading changes larger than 1.5 radians between valid observations are cleaned using the preceding heading.
- Motion matching uses the shipped per-class vocabulary, endpoint bounding-box corner distance, and the preceding *quantized* pose. Matching template widths/lengths remain upstream values (vehicle 2/4.8 m, pedestrian 1/2 m, cyclist 1/1 m), which differ from actual per-track shape attributes. No new vocabulary or clustering algorithm is invented.
- Map preprocessing keeps the released 0.5 m interpolation, 5 m segments and three sampled points per segment; disconnected gaps over 10 m, or heading jumps over 0.1 rad with gaps over 3 m, split polylines. This segmentation detail is inherited code, not asserted as a paper requirement.
- When `meta_infos.json` is present beside the split directory, the released scenario filter `8 <= num_agents < buffer_size` remains active. The paper does not establish that exact filter. Without metadata, all `.pkl` scenarios load; provide metadata to reproduce the upstream filter. An explicitly supplied missing metadata path is an error.

## Repairs and explicit implementation decisions

| Issue or unspecified case | Decision and reason |
| --- | --- |
| Upstream inferred ADD/KEEP for an entirely invalid track. | Keep every segment invalid: there is no first valid segment satisfying Eq. 14. |
| A track has exactly one valid segment, making ADD and REMOVE coincide. | Give ADD precedence. A single control token cannot represent both; ADD gives a well-defined birth, whereas REMOVE would refer to an agent never added. The upstream 15-frame span filter usually avoids this degenerate case. Review this convention if data preparation changes. |
| One category, including vehicles, may be absent at the current frame. | Preserve per-category current-frame height means when present; fall back to observed **historical** scene height, then zero only when no finite observed height exists. Height imputation is an engineering fallback, not a paper claim. It avoids NaNs when creating agents. |
| Raw `city` is absent in this repository's preprocessing output. | Remove it only when present; the released unconditional deletion crashed its own data pipeline. |
| The tokenizer cast a batched ego-index vector to one integer and aliased local/global indices. | Preserve an independent local `av_idx`, clone the mutable `av_index`, and use graph offsets only to gather each scene's ego positions/headings. A regression test checks two scenes with different agent counts and local ego indices. |
| Released configs contain both strings and lists of dataset directories, and a hardcoded `~/infgen/…` metadata path. | Support both path forms and resolve metadata beside each split, with optional explicit `meta_infos_path`. No home-directory layout is required. |
| Adjacent duplicate map vertices and headings crossing ±π. | Skip zero-length interpolation intervals and use wrapped adjacent heading difference. This prevents NaNs and artificial discontinuities. Empty/fully degenerate maps produce typed zero-length tensors. |
| Sequential converter passed the TFRecord output directory as the `existing_files` positional argument; worker processes depended on parent globals. | Use named options and explicit split/validity arguments in both sequential and multiprocessing paths. Workers are compatible with spawned Python processes. |
| Traffic-light time indices were strings but compared with integer frame indices. | Keep time indices as integers so current-frame lights are selected. Signal states still use upstream STOP/GO/CAUTION/UNKNOWN collapsing. |
| Interrupted writes could leave partial scenario pickle files, later treated as completed. | Write to a process-specific temporary file and atomically rename after serialization. Ignore non-pickle files in metadata generation. |
| Full-sequence REMOVE labels and backward extrapolation could reveal a future-only agent at the inference boundary. | Preserve `observed_valid_mask` before extrapolation and derive separate `history_state_idx` from frames 0–10 alone. Historical extrapolation uses only observations in that prefix; the prefix's last control is right-censored (KEEP rather than a future-derived REMOVE). Full `state_idx` remains available as training ground truth. Inference must consume the history fields, not full-sequence lifecycle labels. |
| Released shape means and scene-height fallback read future annotations. | Visible seeds use shapes estimated from historical observations and historical scene-height fallback. Future-only agents retain future shape attributes only as scene-generation training targets; they are not visible inference seeds. Already preprocessed upstream pickles with full-track shape means must be regenerated to remove that baked-in dependence. The paper does not specify the estimator, so this is an explicit causal-inference decision. |
| Shape validation only checked the final agent and could index an empty set. | Require every retained agent to have at least one finite positive length/width/height observation, with an informative failure when data cannot be represented. |
| General `wrap_angle(min,max)` used a formula only correct for a symmetric interval. | Use the standard `(angle - min) mod (max-min) + min`; default [−π,π) behavior is preserved. |

TensorFlow and the Waymo package are loaded only for TFRecord decoding. Reading processed data, unit tests, metadata generation, and converter help do not need them. Worker count is configurable (`--num_workers`, default up to 8), and map debug plotting is opt-in via `DEBUG_MAP=1`. These are portability/resource choices and do not modify the learning objective.

## Verification and remaining limits

`tests/test_data_pipeline.py` checks the lifecycle table (including empty/gapped tracks and configurable state IDs), actual shipped vocabulary tokenization, absent city/category height handling, finite duplicate-vertex map interpolation, empty maps, supported track filtering and angle interpolation, withheld test futures, integer traffic lights, multiple dataset roots and local metadata, optional metadata, sequential converter argument routing, and invariance of history tokens, controls, shapes, and height fallback when future trajectories/validity/attributes are perturbed. Synthetic tests do not validate TFRecord decoding against a licensed WOMD dataset, large multiprocess conversion, or published metrics. Those require the actual dataset and the official evaluation dependencies. The height fallback is a pooled scalar per category for a batch, inherited from upstream; scene-specific height export is used only by the supported single-scene inference path. Multi-scene rollout export would need per-graph height handling.
