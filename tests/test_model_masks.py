import math
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn

from infgen.modules.agent_decoder import InfGenAgentDecoder
from infgen.modules.layers import masked_loss, batched_radius, batched_radius_graph


class ModelMaskTests(unittest.TestCase):
    def test_radius_search_preserves_ids_for_unsorted_scene_batches(self):
        points = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [1., 0.]])
        batch = torch.tensor([0, 1, 0, 1])
        expected = {(0, 2), (2, 0), (1, 3), (3, 1)}
        edges = batched_radius_graph(points, 1.1, batch=batch, loop=False)
        self.assertEqual(set(map(tuple, edges.t().tolist())), expected)
        edges = batched_radius(points, points, 1.1, batch_x=batch, batch_y=batch)
        cross_edges = edges[:, edges[0] != edges[1]]
        self.assertEqual(set(map(tuple, cross_edges.t().tolist())), expected)

    def test_paper_attention_applies_agents_before_map(self):
        calls = []

        def agent(features, relative, edges):
            calls.append('agent')
            return features + 1

        def map_layer(pair, relative, edges):
            calls.append('map')
            return pair[1] * 2

        decoder = SimpleNamespace(paper_objective=True)
        output = InfGenAgentDecoder._attend_spatial(
            decoder, torch.tensor(0), None, agent, map_layer, None, None, None, None
        )
        self.assertEqual(calls, ['agent', 'map'])
        self.assertEqual(output.item(), 2)
        calls.clear()
        decoder.paper_objective = False
        output = InfGenAgentDecoder._attend_spatial(
            decoder, torch.tensor(0), None, agent, map_layer, None, None, None, None
        )
        self.assertEqual(calls, ['map', 'agent'])
        self.assertEqual(output.item(), 1)

    def test_empty_supervision_has_finite_zero_gradient(self):
        for criterion, shape, target in [
            (nn.CrossEntropyLoss(), (3, 4), torch.tensor([0, 1, 2])),
            (nn.MSELoss(), (3, 2), torch.zeros(3, 2)),
            (nn.BCEWithLogitsLoss(), (3, 2), torch.zeros(3, 2)),
        ]:
            prediction = torch.randn(*shape, requires_grad=True)
            mask = torch.zeros(3, dtype=torch.bool)
            loss = masked_loss(criterion, prediction, target, mask)
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_nonempty_mask_preserves_weighted_objective(self):
        prediction = torch.randn(4, 3, requires_grad=True)
        target = torch.tensor([0, 2, 1, 2])
        mask = torch.tensor([False, True, True, False])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.0, 0.1, 0.9]))
        torch.testing.assert_close(
            masked_loss(criterion, prediction, target, mask),
            criterion(prediction[mask], target[mask]),
        )

    def test_spatial_prefix_mask_uses_global_batch_indices(self):
        decoder = SimpleNamespace(num_seed_feature=2, seed_attn_to_av=True)
        data = {'agent': {'ptr': torch.tensor([0, 3, 6])}}
        # Both scenes use local insertion order agent 1, then 2; AV is 0.
        order = torch.tensor([[1], [2], [0], [1], [2], [0]])
        mask, _ = InfGenAgentDecoder._build_seq(
            decoder, 'cpu', data, 6, 1, torch.tensor([0, 3]), order
        )
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[1, 2])
        self.assertFalse(mask[2, 4])
        self.assertTrue(mask[3, 4])
        self.assertFalse(mask[3, 5])
        self.assertTrue(mask[:, [0, 3]].all())
        self.assertFalse(mask[:, 6:].any())

    def test_prefix_occupancy_keeps_two_scenes_and_their_future_agents_separate(self):
        decoder = SimpleNamespace(num_seed_feature=2, seed_attn_to_av=True,
                                  attr_tokenizer=SimpleNamespace(grid_size=6))
        data = type('TwoScenes', (dict,), {'num_graphs': 2})({
            'agent': {'ptr': torch.tensor([0, 2, 4]),
                      'batch': torch.tensor([0, 0, 1, 1]),
                      'state_idx': torch.ones(4, 2, dtype=torch.long),
                      'pt_grid_token_idx': torch.tensor([[4, 5], [4, 5]])},
            'pt_token': {'ptr': torch.tensor([0, 1, 2])},
        })
        order = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0]])
        prefix, _ = InfGenAgentDecoder._build_seq(
            decoder, 'cpu', data, 4, 2, torch.tensor([0, 2]), order
        )
        grid = torch.arange(4)[:, None].expand(-1, 2)
        InfGenAgentDecoder._build_occ_gt(decoder, data, prefix, grid, mode='prefix')
        for query, expected in enumerate(({0}, {0, 1}, {2}, {2, 3})):
            for step in range(2):
                occupied = data['agent']['agent_occ'][query, step].nonzero().flatten().tolist()
                self.assertEqual(set(occupied), expected)
        self.assertEqual(data['agent']['map_occ'][0, 0].nonzero().item(), 4)
        self.assertEqual(data['agent']['map_occ'][2, 0].nonzero().item(), 5)

    def test_paper_temporal_control_mask_excludes_eos_and_null_suffix(self):
        decoder = SimpleNamespace(enter_state=2, exit_state=3, paper_objective=True)
        states = torch.tensor([[0, 2, 1, 1, 3, 0, 0]])
        valid = torch.tensor([[False, True, True, True, True, False, False]])
        motion, controls = InfGenAgentDecoder._next_token_masks(decoder, states, valid)
        expected = torch.tensor([[False, True, True, True, False, False, False]])
        torch.testing.assert_close(motion, expected)
        torch.testing.assert_close(controls, expected)

    def test_enter_at_end_does_not_assign_an_empty_lookahead_slice(self):
        decoder = SimpleNamespace(enter_state=2, exit_state=3, paper_objective=True)
        states = torch.tensor([[0, 0, 2, 1]])
        valid = states != 0
        motion, controls = InfGenAgentDecoder._next_token_masks(decoder, states, valid)
        self.assertTrue(motion[0, 2])
        self.assertFalse(controls[0, -1])

    def test_paper_control_sampling_excludes_temporal_null(self):
        decoder = SimpleNamespace(paper_objective=True)
        logits = torch.tensor([[1000.0, 0.0, 0.0]]).expand(1000, -1)
        with torch.random.fork_rng():
            torch.manual_seed(7)
            actions = InfGenAgentDecoder._decode_control(decoder, logits, temporal=True)
        self.assertEqual(set(actions.tolist()), {1, 2})
        decoder.paper_objective = False
        self.assertTrue((InfGenAgentDecoder._decode_control(decoder, logits, temporal=True) == 0).all())

    def test_remove_is_terminal_and_keep_cannot_resurrect_the_agent(self):
        decoder = SimpleNamespace(paper_objective=False, invalid_state=0, valid_state=1,
                                  exit_state=3, valid_state_type=['invalid', 'valid', 'exit'],
                                  use_state_token=True, disable_insertion=False)
        decoder._decode_control = MethodType(InfGenAgentDecoder._decode_control, decoder)
        previous = torch.tensor([1, 1])
        remove = torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]])
        states = InfGenAgentDecoder._advance_agent_states(decoder, remove, previous, 0)
        torch.testing.assert_close(states, torch.tensor([1, 3]))
        keep = torch.tensor([[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]])
        states = InfGenAgentDecoder._advance_agent_states(decoder, keep, states, 0)
        torch.testing.assert_close(states, torch.tensor([1, 0]))
        states = InfGenAgentDecoder._advance_agent_states(decoder, keep, states, 0)
        torch.testing.assert_close(states, torch.tensor([1, 0]))

    def test_seed_boundary_control_cannot_reveal_future_removal(self):
        decoder = SimpleNamespace(num_historical_steps=11, shift=5, invalid_state=0,
                                  valid_state=1, exit_state=3)
        stays = torch.tensor([[2, 1, 1, 1]])
        leaves = torch.tensor([[2, 3, 0, 0]])
        seed_stays = InfGenAgentDecoder._history_states(decoder, {'agent': {'state_idx': stays}})
        seed_leaves = InfGenAgentDecoder._history_states(decoder, {'agent': {'state_idx': leaves}})
        torch.testing.assert_close(seed_stays, seed_leaves)
        torch.testing.assert_close(seed_leaves, torch.tensor([[2, 1, 0, 0]]))
        torch.testing.assert_close(leaves, torch.tensor([[2, 3, 0, 0]]))

    def test_seed_prefers_history_only_states_over_full_log_labels(self):
        decoder = SimpleNamespace(num_historical_steps=11, shift=5, invalid_state=0,
                                  valid_state=1, exit_state=3)
        data = {'agent': {'state_idx': torch.tensor([[2, 1, 1, 1]]),
                          'history_state_idx': torch.tensor([[0, 0]])}}
        seed = InfGenAgentDecoder._history_states(decoder, data)
        self.assertFalse(seed.any())

    def test_invalid_heading_is_constant_without_mutating_caller(self):
        decoder = SimpleNamespace(input_dim=2, invalid_state=0, enter_state=2,
                                  invalid_motion_value=-2.0, invalid_head_value=-2.0,
                                  motion_gap=1.0)
        position = torch.zeros(1, 3, 2)
        heading = torch.tensor([[0.1, 2.3, -0.7]])
        original = heading.clone()
        states = torch.tensor([[1, 0, 1]])
        _, vectors = InfGenAgentDecoder._build_vector_a(decoder, position, heading, states)
        torch.testing.assert_close(heading, original)
        torch.testing.assert_close(vectors[0, 1], torch.tensor([math.cos(-2.0), math.sin(-2.0)]))

    def test_temporal_attention_is_causal_and_sanitizes_valid_to_invalid(self):
        decoder = SimpleNamespace(input_dim=2, invalid_state=0, enter_state=2,
                                  num_seed_feature=1, temporal_attn_to_invalid=True,
                                  temporal_attn_seed=False, hist_mask=False,
                                  time_span=10, shift=5, motion_gap=1.0,
                                  heading_gap=1.0, invalid_motion_value=-2.0,
                                  invalid_head_value=-2.0, r_t_emb=lambda **kw: kw['continuous_inputs'])
        positions = torch.zeros(2, 4, 2)
        headings = torch.tensor([[0.1, 2.3, -0.7, 0.0], [0.0, 0.0, 0.0, 0.0]])
        vectors = torch.stack([headings.cos(), headings.sin()], dim=-1)
        states = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]])
        edges, relative = InfGenAgentDecoder._build_temporal_edge(
            decoder, SimpleNamespace(num_graphs=1), positions, headings, states,
            vectors, torch.ones(2, 4, dtype=torch.bool)
        )
        lag = edges[1] - edges[0]
        self.assertTrue(((lag > 0) & (lag <= 2)).all())
        self.assertTrue((edges < 4).all())
        target_edge = (edges[0] == 0) & (edges[1] == 1)
        self.assertEqual(relative[target_edge, 2].item(), 1.0)

    def test_inference_temporal_edges_keep_history_for_all_real_agents(self):
        decoder = SimpleNamespace(input_dim=2, invalid_state=0, enter_state=2,
                                  num_seed_feature=10, temporal_attn_to_invalid=False,
                                  temporal_attn_seed=False, hist_mask=False,
                                  time_span=10, shift=5, motion_gap=1.0,
                                  heading_gap=1.0, invalid_motion_value=-2.0,
                                  invalid_head_value=-2.0, r_t_emb=lambda **kw: kw['continuous_inputs'])
        positions = torch.zeros(7, 4, 2)
        headings = torch.zeros(7, 4)
        states = torch.ones(7, 4, dtype=torch.long)
        vectors = torch.stack((headings.cos(), headings.sin()), dim=-1)
        inference_mask = torch.zeros(7, 4, dtype=torch.bool)
        inference_mask[:, -1] = True
        original_mask = inference_mask.clone()
        edges, _ = InfGenAgentDecoder._build_temporal_edge(
            decoder, SimpleNamespace(num_graphs=1), positions, headings, states,
            vectors, torch.ones(7, 4, dtype=torch.bool), inference_mask,
        )
        self.assertEqual(set((edges[1] // 4).tolist()), set(range(7)))
        self.assertTrue((edges[0] < edges[1]).all())
        torch.testing.assert_close(inference_mask, original_mask)


if __name__ == '__main__':
    unittest.main()
