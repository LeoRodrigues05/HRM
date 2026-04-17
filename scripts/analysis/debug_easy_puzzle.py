#!/usr/bin/env python3
"""Debug script to understand why given cells are being changed."""

import numpy as np
import torch
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

NPZ_PATH = 'data/step_0_all_preds.npz'
CHECKPOINT_PATH = 'checkpoints/sapientinc-sudoku-extreme/checkpoint.pt'
CONFIG_PATH = 'checkpoints/sapientinc-sudoku-extreme/all_config.yaml'

def id2num(i):
    if 2 <= i <= 10:
        return str(i - 1)
    return '.'

def main():
    # Load data
    z = np.load(NPZ_PATH, allow_pickle=True)
    labels = z['labels']

    # Create one easy puzzle
    rng = np.random.default_rng(42)
    sol_idx = 348156
    solution = labels[sol_idx]
    puzzle = solution.copy()
    blank_idx = rng.choice(81, size=1, replace=False)[0]
    puzzle[blank_idx] = 1  # blank token

    print('Original solution (first 20 cells):')
    print([id2num(int(x)) for x in solution[:20]])
    print(f'\nBlanked index: {blank_idx}')
    print(f'Value at blanked index in solution: {id2num(int(solution[blank_idx]))}')
    print(f'Value at blanked index in puzzle: {id2num(int(puzzle[blank_idx]))}')

    # Now load model and run
    from pretrain import PretrainConfig, init_train_state, create_dataloader

    with open(CONFIG_PATH, 'r') as f:
        config = PretrainConfig(**yaml.safe_load(f))

    train_loader, train_metadata = create_dataloader(
        config, 'train', test_set_mode=False, epochs_per_iter=1, 
        global_batch_size=config.global_batch_size, rank=0, world_size=1
    )

    train_state = init_train_state(config, train_metadata, world_size=1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')
    train_state.model.load_state_dict(
        {k.removeprefix('_orig_mod.'): v for k, v in checkpoint.items()},
        assign=True
    )
    train_state.model.eval()
    model = train_state.model

    # Run inference
    batch = {
        'inputs': torch.tensor(puzzle[None, :], dtype=torch.int32).cuda(),
        'labels': torch.tensor(solution[None, :], dtype=torch.int32).cuda(),
        'puzzle_identifiers': torch.tensor([0], dtype=torch.int32).cuda()
    }

    print('\n--- Running model ---')
    print(f'Input puzzle (missing cell at {blank_idx}):')
    input_chars = [id2num(int(x)) for x in puzzle]
    for r in range(9):
        print(f'Row {r+1}:', ''.join(input_chars[r*9:(r+1)*9]))

    with torch.inference_mode():
        with torch.device('cuda'):
            carry = model.initial_carry(batch)
        
        for step in range(3):
            carry, loss, metrics, outputs, all_finish = model(
                carry=carry, 
                batch=batch,
                return_keys=['logits', 'intermediate_preds_step']
            )
            
            pred = outputs['logits'].argmax(-1).cpu().numpy()[0]
            pred_chars = [id2num(int(x)) for x in pred]
            
            # Check how many cells differ from input
            input_np = puzzle
            changed_from_input = [(i, id2num(int(input_np[i])), pred_chars[i]) 
                                   for i in range(81) 
                                   if pred[i] != input_np[i] and input_np[i] != 1]
            
            print(f'\nStep {step+1}:')
            print(f'  Prediction at blank idx {blank_idx}: {pred_chars[blank_idx]} (expected: {id2num(int(solution[blank_idx]))})')
            print(f'  Given cells changed by model: {len(changed_from_input)}')
            
            if changed_from_input:
                print('  Examples of changed given cells:')
                for i, inp, pred_val in changed_from_input[:10]:
                    row, col = i // 9, i % 9
                    print(f'    Cell ({row+1},{col+1}): input={inp}, pred={pred_val}')

if __name__ == '__main__':
    main()
