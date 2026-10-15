from pathlib import Path
import unittest

import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Batch

from infgen.datasets.scalable_dataset import WaymoTargetBuilder
from infgen.model.infgen import InfGen
from scripts.make_synthetic_data import make_scenario


ROOT = Path(__file__).resolve().parents[1]


class ConstantHead(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, features):
        return features.new_tensor(self.values).expand(*features.shape[:-1], len(self.values))


class KeepRemoveKeepHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.step = 0

    def forward(self, features):
        values = [-100., -100., 100.] if self.step == 1 else [-100., 100., -100.]
        self.step += 1
        return features.new_tensor(values).expand(*features.shape[:-1], 3)


@unittest.skipUnless((ROOT / 'infgen/tokens/agent_vocab_555_s2.pkl').is_file()
                     and (ROOT / 'infgen/tokens/map_traj_token5.pkl').is_file(),
                     'Download the official vocabulary assets to run model rollout tests')
class DynamicRolloutTests(unittest.TestCase):
    def test_insertion_capacity_and_permanent_removal_in_full_decoder(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng():
                torch.manual_seed(2024)
                with (ROOT / 'configs/paper_reproduction.yaml').open() as stream:
                    config = EasyDict(yaml.safe_load(stream))['Model']
                config.num_recurrent_steps_val = 15
                model = InfGen(config).eval()
                decoder = model.encoder.agent_encoder
                self.assertEqual(sum(isinstance(layer, torch.nn.Linear) for layer in decoder.fusion_emb.modules()), 1)
                self.assertEqual(sum(isinstance(layer, torch.nn.Linear) for layer in decoder.seed_shape_predict_head.modules()), 3)
                transform = WaymoTargetBuilder(11, 80, max_num=32, training=False)
                second = make_scenario(1)
                second['map_polygon']['light_type'].fill_(3)
                map_batch = Batch.from_data_list([transform(make_scenario()), transform(second)])
                with torch.inference_mode():
                    map_batch = model.match_token_map(map_batch)
                map_edges = map_batch['pt_token', 'to', 'map_polygon']['edge_index']
                torch.testing.assert_close(
                    map_batch['pt_token']['batch'][map_edges[0]],
                    map_batch['map_polygon']['batch'][map_edges[1]],
                )
                decoder.buffer_size = 21
                decoder.seed_state_predict_head = ConstantHead([-100., 100.])
                decoder.state_predict_head = KeepRemoveKeepHead()
                scenario = WaymoTargetBuilder(11, 80, max_num=32, training=False)(make_scenario())
                data = Batch.from_data_list([scenario])
                with torch.inference_mode():
                    data = model.token_processer(data)
                    data = model.match_token_map(data)
                    data = model.sample_pt_pred(data)
                    data = model._fetch_enterings(data)
                    data['batch_size_a'] = data['agent']['ptr'].diff()
                    data['batch_size_pl'] = data['pt_token']['ptr'].diff()
                    data['agent']['av_index'] += data['agent']['ptr'][:-1]
                    result = model.encoder.inference(data)

                # Three real model steps: KEEP, REMOVE, then KEEP. Capacity is
                # live agents, while output rows retain earlier retired IDs.
                self.assertGreater(result['agent_id'].numel(), decoder.buffer_size)
                self.assertLessEqual(result['pred_valid'].sum(0).max().item(), decoder.buffer_size)
                self.assertGreater((result['next_state_idx'][:, 2] == decoder.enter_state).sum().item(), 10)
                self.assertEqual(result['next_state_idx'][1, -2:].tolist(), [decoder.exit_state, decoder.invalid_state])
                self.assertFalse(result['pred_valid'][1, -10:].any())
                self.assertFalse(result['pred_traj'][1, -10:].any())
                self.assertEqual(result['pred_traj'].shape[1], 26)
                self.assertTrue(torch.isfinite(result['pred_traj']).all())
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
