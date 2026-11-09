"""Tiny real PyTorch backward and denoising smoke test; no pretrained weights."""
import importlib.util
import math
import random
import unittest

import masked_diffusion as md

HAS_DEPS = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'))


@unittest.skipUnless(HAS_DEPS, 'PyTorch/Transformers отсутствуют в этой среде')
class TorchSmokeTests(unittest.TestCase):
    def test_backward_optimizer_and_reverse_unmask(self):
        import torch
        from transformers import BertConfig, BertForMaskedLM

        torch.manual_seed(7)
        config = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64,
                            max_position_embeddings=64)
        model = BertForMaskedLM(config)
        model.train()
        prefix, target = [1, 20, 2], [10, 11, 4, 12, 13]
        ids, labels = md.corrupt(prefix, target, [0, 1, 3, 4], 3,
                                 random.Random(3), 0.75)
        ids.append(2)
        labels.append(-100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        weight_before = model.bert.embeddings.word_embeddings.weight.detach().clone()
        loss = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(weight_before, model.bert.embeddings.word_embeddings.weight))

        model.eval()
        sequence = prefix + [3, 3, 4, 3, 3] + [2]
        def predict(current, positions):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([current])).logits[0]
            result = {}
            for pos in positions:
                row = logits[len(prefix) + pos].clone()
                row[:6] = -float('inf')
                probability = torch.softmax(row, dim=0)
                choice = int(probability.argmax())
                result[pos] = choice, float(probability[choice])
            return result
        decoded = md.reverse_unmask(sequence, len(prefix), {0, 1, 3, 4}, 3, 3, predict)
        self.assertEqual(decoded[len(prefix) + 2], 4)
        self.assertTrue(all(decoded[len(prefix) + pos] != 3 for pos in (0, 1, 3, 4)))


if __name__ == '__main__':
    unittest.main()
