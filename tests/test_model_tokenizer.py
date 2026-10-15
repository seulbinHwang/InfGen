import math
import unittest

import torch

from infgen.modules.attr_tokenizer import Attr_Tokenizer


class PositionTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Attr_Tokenizer(150.0, 3.0, 75.0, 3.0)

    def test_shared_grid_broadcasts_over_different_ego_headings(self):
        centers = torch.tensor([[1.0, 2.0], [30.0, -12.0]])
        headings = torch.tensor([math.pi / 2, -math.pi / 3])
        grids = self.tokenizer.get_grid(centers, headings)
        self.assertEqual(grids.shape, (2, self.tokenizer.grid_size, 2))
        for row in range(2):
            encoded, offsets = self.tokenizer.encode_pos(
                grids[row], centers[row:row + 1], headings[row:row + 1]
            )
            torch.testing.assert_close(encoded, torch.arange(self.tokenizer.grid_size))
            torch.testing.assert_close(offsets, torch.zeros_like(offsets), atol=2e-5, rtol=0)

    def test_position_decode_encode_round_trip(self):
        centers = torch.tensor([[2.0, 10.0], [-5.0, 2.0]])
        headings = torch.tensor([0.0, math.pi / 4])
        indices = torch.tensor([100, 1250])
        positions = self.tokenizer.decode_pos(indices, centers, headings)
        actual, offsets = self.tokenizer.encode_pos(positions, centers, headings)
        torch.testing.assert_close(actual, indices)
        torch.testing.assert_close(offsets, torch.zeros_like(offsets), atol=1e-5, rtol=0)

    def test_empty_position_batch_and_wrapped_heading_bounds(self):
        indices, offsets = self.tokenizer.encode_pos(
            torch.empty(0, 2), torch.zeros(1, 2), torch.zeros(1)
        )
        self.assertEqual(indices.shape, (0,))
        self.assertEqual(offsets.shape, (0, 2))
        headings = torch.tensor([-math.pi, math.pi, 0, 20 * math.pi, math.pi - 1e-7])
        indices = self.tokenizer.encode_heading(headings)
        self.assertTrue(torch.all((indices >= 0) & (indices < 120)))
        reconstructed = self.tokenizer.decode_heading(indices)
        error = (headings - reconstructed + math.pi).remainder(2 * math.pi) - math.pi
        self.assertTrue(torch.all(error.abs() <= math.radians(3.0) + 1e-5))

    def test_paper_heading_chooses_nearest_circular_bin_representative(self):
        tokenizer = Attr_Tokenizer(150., 3., 75., 3., nearest_heading=True)
        heading = torch.deg2rad(torch.tensor([-178.4, 179.2, 1.6]))
        indices = tokenizer.encode_heading(heading)
        decoded = tokenizer.decode_heading(indices)
        torch.testing.assert_close(decoded, torch.deg2rad(torch.tensor([-177., -180., 3.])))
        error = (heading - decoded + math.pi).remainder(2 * math.pi) - math.pi
        self.assertTrue((error.abs() <= math.radians(1.5) + 1e-6).all())


if __name__ == '__main__':
    unittest.main()
