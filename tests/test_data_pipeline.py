"""Paper lifecycle equations and regression cases for the released data path."""
import json
import math
import pickle

import numpy as np
import pytest
import torch

from data_preprocess import (
    get_agent_features, get_map_features, main, process_dynamic_map,
)
from infgen.datasets.preprocess import (
    TokenProcessor, control_states_from_validity, interplating_polyline,
)
from infgen.datasets.scalable_dataset import MultiDataset
from infgen.utils.func import wrap_angle


STATES = {'invalid': 0, 'valid': 1, 'enter': 2, 'exit': 3}


def test_control_lifecycle_matches_appendix_c():
    validity = torch.tensor([
        [False, False, True, True, True, False],
        [True, True, True, True, True, True],
        [False, False, False, False, False, False],
        [False, True, False, True, False, False],
        [False, True, False, False, False, False],
    ])
    assert control_states_from_validity(validity, STATES).tolist() == [
        [0, 0, 2, 1, 3, 0],  # first ADD, last REMOVE
        [2, 1, 1, 1, 1, 1],  # last-horizon REMOVE becomes KEEP
        [0, 0, 0, 0, 0, 0],  # no valid segment cannot create an agent
        [0, 2, 1, 3, 0, 0],  # Appendix C defines controls between endpoints
        [0, 2, 0, 0, 0, 0],  # documented single-segment convention
    ]


def test_control_states_respect_configured_ids():
    states = {'invalid': 7, 'valid': 8, 'enter': 9, 'exit': 10}
    actual = control_states_from_validity(torch.tensor([[0, 1, 1, 0]]), states)
    assert actual.tolist() == [[7, 9, 10, 7]]


def test_wrap_angle_supports_general_interval():
    values = torch.tensor([-1.0, 0.0, 3.0, 6.0])
    torch.testing.assert_close(wrap_angle(values, 0.0, 5.0), torch.tensor([4.0, 0.0, 3.0, 1.0]))
    with pytest.raises(ValueError):
        wrap_angle(values, 5.0, 0.0)


def test_map_interpolation_duplicate_points_and_heading_wrap_are_finite():
    xy = np.array([[0., 0.], [0., 0.], [4., 0.], [9., 0.]])
    headings = np.array([math.pi - .01, math.pi - .01, -math.pi + .01, -math.pi + .01])
    result = interplating_polyline(xy, headings)
    assert result.shape[-2:] == (3, 3)
    assert torch.isfinite(result).all()
    assert result[0, -1, 0] == 5.0  # no artificial split at the +-pi boundary
    assert interplating_polyline(np.zeros((2, 2)), np.zeros(2)) is None
    assert interplating_polyline(np.empty((0, 2)), np.empty(0)) is None


def test_empty_map_is_well_shaped(monkeypatch):
    monkeypatch.delenv('DEBUG_MAP', raising=False)
    map_infos = {key: [] for key in ['lane', 'crosswalk', 'road_edge', 'road_line']}
    map_infos['all_polylines'] = np.empty((0, 5), dtype=np.float32)
    lights = process_dynamic_map({'lane_id': [], 'state': []})
    data = get_map_features(map_infos, lights)
    assert data['map_point']['position'].shape == (0, 3)
    data = TokenProcessor._tokenize_map(data)
    assert data['pt_token']['num_nodes'] == 0
    assert data['map_save']['traj_pos'].shape == (0, 3, 2)


def test_traffic_light_time_matches_integer_current_step():
    lights = process_dynamic_map({
        'lane_id': [np.array([[123]])],
        'state': [np.array([['LANE_STATE_ARROW_STOP']])],
    })
    current = lights.loc[lights['time_step'] == 0]
    assert len(current) == 1
    assert current['lane_id'].item() == '123'
    assert current['state'].item() == 'LANE_STATE_STOP'


def _track_infos():
    states = np.zeros((3, 91, 9), dtype=np.float32)
    states[..., 3:6] = [4., 2., 1.]
    states[0, :, 0] = np.arange(91)
    states[0, :, 6] = np.where(np.arange(91) < 45, 3.1, -3.1)
    valid = np.zeros((3, 91), dtype=bool)
    valid[0, ::2] = True
    valid[2] = True
    return {
        'object_id': np.array([10, 20, 30]), 'object_type': np.array([0, 1, 3]),
        'valid': valid, 'states': states,
        'role': np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool),
        'tracks_to_predict': np.array([0]),
    }


def test_preprocess_interpolates_only_supported_nonempty_tracks():
    result = get_agent_features(_track_infos(), 10, 11, 91)
    assert result['num_nodes'] == 1
    assert result['av_idx'] == 0
    assert result['valid_mask'].all()
    torch.testing.assert_close(result['position'][0, :, 0], torch.arange(91, dtype=torch.float32))
    assert (result['heading'][0, 1:] - result['heading'][0, :-1]).abs().max() < .1


def test_unlabeled_test_future_is_not_interpolated():
    tracks = _track_infos()
    tracks['valid'][0, 11:] = False
    result = get_agent_features(tracks, 10, 11, 91, split='testing', disable_invalid=True)
    assert not result['valid_mask'][0, 11:].any()
    assert not result['valid_mask'][0, 1]  # test history gaps are left as supplied


def test_dataset_accepts_list_paths_and_local_metadata(tmp_path):
    dirs = [tmp_path / 'one' / 'training', tmp_path / 'two' / 'training']
    for index, directory in enumerate(dirs):
        directory.mkdir(parents=True)
        scenario = f'scene_{index}'
        with (directory / f'{scenario}.pkl').open('wb') as handle:
            pickle.dump({'scenario_id': scenario}, handle)
        (directory.parent / 'meta_infos.json').write_text(json.dumps({
            'training': {scenario: {'num_agents': 8}},
        }))
    dataset = MultiDataset('train', raw_dir=dirs)
    assert len(dataset) == 2
    assert dataset.get(1)['scenario_id'] == 'scene_1'


def test_dataset_without_metadata_and_explicit_missing_metadata(tmp_path):
    with (tmp_path / 'only.pkl').open('wb') as handle:
        pickle.dump({'scenario_id': 'only'}, handle)
    dataset = MultiDataset('val', raw_dir=str(tmp_path))
    assert len(dataset) == 1
    with pytest.raises(FileNotFoundError, match='Explicit metadata'):
        MultiDataset('val', raw_dir=str(tmp_path), meta_infos_path=str(tmp_path / 'missing.json'))


def test_sequential_preprocess_passes_named_options(tmp_path, monkeypatch):
    raw_dir, processed_dir = tmp_path / 'raw', tmp_path / 'processed'
    (raw_dir / 'validation').mkdir(parents=True)
    (raw_dir / 'validation' / 'shard.tfrecord').touch()
    calls = []
    monkeypatch.setattr('data_preprocess.wm2argo', lambda *args, **kwargs: calls.append((args, kwargs)))
    main(['--no_batch', '--split', 'validation', '--input_dir', str(raw_dir), '--output_dir', str(processed_dir)])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 3
    assert kwargs['existing_files'] == []
    assert kwargs['output_dir_tfrecords_splitted'] == str(processed_dir / 'validation_tfrecords_splitted')
    assert kwargs['split'] == 'validation'


def test_real_vocab_tokenization_without_city_and_without_vehicles():
    processor = TokenProcessor(2048, training=True, predict_state=True, state_token=STATES)
    assert processor.training
    agent_count = 2
    agent = {
        'num_nodes': agent_count, 'av_idx': 0,
        'valid_mask': torch.zeros((agent_count, 91), dtype=torch.bool),
        'position': torch.zeros((agent_count, 91, 3)),
        'velocity': torch.zeros((agent_count, 91, 2)),
        'heading': torch.zeros((agent_count, 91)),
        'type': torch.ones(agent_count, dtype=torch.uint8),
        'category': torch.zeros(agent_count, dtype=torch.uint8),
        'shape': torch.ones((agent_count, 91, 3)),
    }
    agent['valid_mask'][0] = True
    agent['position'][0, :, 2] = 7.0
    data = processor({'agent': agent, 'map_polygon': {}})
    assert data['agent']['state_idx'][0].tolist() == [2] + [1] * 17
    assert data['agent']['state_idx'][1].tolist() == [0] * 18
    assert data['agent']['token_idx'][1].tolist() == [-1] * 18
    assert all(torch.isfinite(z) for z in data['agent']['raw_height'].values())
    assert data['agent']['raw_height']['veh'] == 7.0
    assert data['agent']['token_pos'].shape == (agent_count, 18, 2)


def test_visible_seed_shapes_are_estimated_only_from_history():
    tracks = _track_infos()
    baseline = get_agent_features(tracks, 10, 11, 91)
    tracks['states'][0, 11:, 3:6] = [100., 200., 300.]
    perturbed = get_agent_features(tracks, 10, 11, 91)
    torch.testing.assert_close(baseline['shape'], perturbed['shape'])


def test_history_tokenization_cannot_use_future_validity_or_attributes():
    import copy

    processor = TokenProcessor(2048, predict_state=True, state_token=STATES)
    agent = {
        'num_nodes': 2, 'av_idx': 0,
        'valid_mask': torch.zeros((2, 91), dtype=torch.bool),
        'position': torch.zeros((2, 91, 3)),
        'velocity': torch.zeros((2, 91, 2)),
        'heading': torch.zeros((2, 91)),
        'type': torch.tensor([1, 0], dtype=torch.uint8),
        'category': torch.zeros(2, dtype=torch.uint8),
        'shape': torch.ones((2, 91, 3)),
    }
    agent['valid_mask'][0] = True
    agent['valid_mask'][1, 11:] = True  # extrapolation would fabricate visibility at t=1s
    agent['position'][0, :, 2] = 7.0
    baseline = processor({'agent': copy.deepcopy(agent), 'map_polygon': {}})['agent']
    agent['valid_mask'][0, 11:] = False  # makes full-sequence GT control REMOVE at token 1
    agent['position'][:, 11:] += 1000
    agent['velocity'][:, 11:] += 500
    agent['heading'][:, 11:] += 1.0
    agent['shape'][:, 11:] += 100
    perturbed = processor({'agent': copy.deepcopy(agent), 'map_polygon': {}})['agent']
    assert baseline['state_idx'][0, 1] != perturbed['state_idx'][0, 1]
    assert baseline['history_state_idx'].tolist() == [[2, 1], [0, 0]]
    torch.testing.assert_close(baseline['history_state_idx'], perturbed['history_state_idx'])
    for key in ('token_pos', 'token_heading', 'token_idx'):
        torch.testing.assert_close(baseline[key][0, :2], perturbed[key][0, :2])
    torch.testing.assert_close(baseline['shape'][0], perturbed['shape'][0])
    for category in ('veh', 'ped', 'cyc'):
        torch.testing.assert_close(baseline['raw_height'][category], perturbed['raw_height'][category])
    assert not baseline['observed_valid_mask'][1, 10]
    assert baseline['valid_mask'][1, 10]  # legacy GT extrapolation remains separate
    assert not agent['valid_mask'][1, 10]  # source fixture was not accidentally modified


def test_batched_tokenizer_uses_per_graph_ego_offsets_without_aliasing():
    from torch_geometric.data import Batch, HeteroData

    def scene(count, ego_index, offset):
        positions = torch.zeros((count, 91, 3))
        positions[..., 0] = offset + torch.arange(count)[:, None] * 10
        return HeteroData({
            'agent': {
                'num_nodes': count, 'av_idx': ego_index,
                'valid_mask': torch.ones((count, 91), dtype=torch.bool),
                'position': positions,
                'velocity': torch.zeros((count, 91, 2)),
                'heading': torch.zeros((count, 91)),
                'type': torch.zeros(count, dtype=torch.uint8),
                'category': torch.zeros(count, dtype=torch.uint8),
                'shape': torch.ones((count, 91, 3)),
            },
            'map_polygon': {'num_nodes': 0},
        })

    batch = Batch.from_data_list([scene(2, 1, 0), scene(3, 0, 100)])
    processor = TokenProcessor(2048, predict_state=True, state_token=STATES)
    actual = processor(batch)
    assert actual['agent']['av_idx'].tolist() == [1, 0]
    assert actual['agent']['av_index'].tolist() == [1, 0]
    assert actual['ego_pos'].shape == (2, 18, 2)
    global_ego = torch.tensor([1, 2])
    torch.testing.assert_close(actual['ego_pos'], actual['agent']['token_pos'][global_ego])
    torch.testing.assert_close(actual['ego_heading'], actual['agent']['token_heading'][global_ego])
    actual['agent']['av_index'] += actual['agent']['ptr'][:-1]
    assert actual['agent']['av_index'].tolist() == [1, 2]
    assert actual['agent']['av_idx'].tolist() == [1, 0]
