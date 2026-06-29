from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import os
import json
import shutil
import pickle
import hashlib
import numpy as np
from glob import glob

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata, dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    # ARC-1
    dataset_dirs: List[str] = ["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"]
    output_dir: str = "data/arc-aug-1000"
    
    # ARC-2
    # dataset_dirs: List[str] = ["dataset/raw-data/ARC-AGI-2/data"]
    # output_dir: str = "data/arc-2-aug-1000"

    seed: int = 42
    num_aug: int = 1000
    
    
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5
    

@dataclass
class ARCPuzzle:
    id: str

    examples: List[Tuple[np.ndarray, np.ndarray]]

    
def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    def _grid_hash(grid: np.ndarray):
        buffer = [x.to_bytes(1, "big") for x in grid.shape]
        buffer.append(grid.tobytes())
        
        return hashlib.sha256(b"".join(buffer)).hexdigest()
    
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{_grid_hash(input)}|{_grid_hash(label)}")
            
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def convert_single_arc_puzzle(results: dict, default_name: str, puzzle: dict, aug_count: int, dest_mapping: Dict[str, Tuple[str, str]], spill_handle):
    # Remove "name"
    name = puzzle.pop("name", default_name)
    
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([(arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"])) for example in examples])

    group = [converted]
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            # Augment plan
            trans_id = np.random.randint(0, 8)
            mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])  # Permute colors, Excluding "0" (black)
            
            aug_repr = f"t{trans_id}_{''.join(str(x) for x in mapping)}"

            def _map_grid(grid: np.ndarray):
                return dihedral_transform(mapping[grid], trans_id)
            
            # Check duplicate
            augmented = {dest: ARCPuzzle(f"{puzzle.id}_{aug_repr}", [(_map_grid(input), _map_grid(label)) for (input, label) in puzzle.examples]) for dest, puzzle in converted.items()}
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break
            
        if len(group) < aug_count + 1:
            print (f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append (memory-lean): keep only the per-aug id strings in `results`; spill the
    # actual grids to a per-(split,set) pickle stream so we never hold the whole
    # augmented dataset in RAM. The RNG above is untouched, so the spilled content
    # and identifier strings are byte-identical to the in-memory build.
    for dest in dests:
        dest_split, dest_set = dest
        dest_group = [converted[dest] for converted in group]

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([p.id for p in dest_group])

        pickle.dump([p.examples for p in dest_group],
                    spill_handle(dest_split, dest_set),
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_puzzles_arcagi(results: dict, dataset_path: str, config: DataProcessConfig, spill_handle):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        "evaluation": [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))]
    }
    
    total_puzzles = 0
    # Deterministic traversal: os.scandir / glob return filesystem order, which
    # varies by machine and perturbs the augmentation RNG (and thus the unique-aug
    # count) for borderline small-grid puzzles. Sorting makes the build reproducible.
    for subdir in sorted(os.scandir(dataset_path), key=lambda e: e.name):
        if subdir.is_dir():
            # Load all puzzles in this directory
            puzzles = []
            for filename in sorted(glob(os.path.join(subdir.path, "*.json"))):
                with open(filename, "r") as f:
                    puzzles.append((Path(filename).stem, json.load(f)))
                    
            # Shuffle puzzles
            np.random.shuffle(puzzles)
            
            # Assign by fraction
            for idx, (default_name, puzzle) in enumerate(puzzles):
                fraction = idx / len(puzzles)
                test_examples_dest = None
                for f, dest in test_examples_map.get(subdir.name, test_examples_map["_default"]):
                    if fraction < f:
                        test_examples_dest = dest
                        break
                        
                assert test_examples_dest is not None
                
                convert_single_arc_puzzle(results, default_name, puzzle, config.num_aug, {"train": train_examples_dest, "test": test_examples_dest}, spill_handle)
                total_puzzles += 1

    print (f"[{dataset_path}] total puzzles: {total_puzzles}")


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    # Memory-lean build: phase 1 augments puzzles (RNG identical to the original)
    # but spills the grids to per-(split,set) pickle streams on disk, holding only
    # the lightweight per-aug id strings in RAM. Phase 2 reads the grids back in the
    # same order, applies the translational-augment RNG, and writes the .npy arrays.
    # Peak RAM is one subset's output arrays instead of the whole augmented dataset.
    os.makedirs(config.output_dir, exist_ok=True)
    spill_dir = os.path.join(config.output_dir, "_spill")
    if os.path.exists(spill_dir):
        shutil.rmtree(spill_dir)
    os.makedirs(spill_dir)

    spill_files: Dict[Tuple[str, str], object] = {}
    spill_paths: Dict[Tuple[str, str], str] = {}

    def spill_handle(split_name: str, subset_name: str):
        key = (split_name, subset_name)
        if key not in spill_files:
            path = os.path.join(spill_dir, f"{split_name}__{subset_name}.pkl")
            spill_paths[key] = path
            spill_files[key] = open(path, "wb")
        return spill_files[key]

    # Phase 1 — read + augment (grids -> disk, id strings -> memory)
    data: Dict[str, Dict[str, list]] = {}
    for dataset_dir in config.dataset_dirs:
        load_puzzles_arcagi(data, dataset_dir, config, spill_handle)
    for f in spill_files.values():
        f.close()

    # Map global puzzle identifiers (split-major, group order, aug order — identical
    # to the original in-memory iteration, so indices match the trained checkpoint).
    num_identifiers = 1  # 0 is blank
    identifier_map = {}
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for pid in group:            # group is now a list of id strings
                    if pid not in identifier_map:
                        identifier_map[pid] = num_identifiers
                        num_identifiers += 1

    print (f"Total puzzle IDs (including <blank>): {num_identifiers}", flush=True)

    # Phase 2 — save (grids streamed back from disk)
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Translational augmentations
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():
            # Construct subset
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            with open(spill_paths[(split_name, subset_name)], "rb") as spill_f:
                for group in subset:                       # group = list of id strings
                    group_examples = pickle.load(spill_f)  # list (per aug) of examples
                    for pid, examples in zip(group, group_examples):
                        # Push puzzle
                        no_aug_id = np.random.randint(0, len(examples))
                        for _idx_ex, (inp, out) in enumerate(examples):
                            inp, out = np_grid_to_seq_translational_augment(inp, out, do_translation=enable_translational_augment and _idx_ex != no_aug_id)

                            results["inputs"].append(inp)
                            results["labels"].append(out)
                            example_id += 1

                            total_examples += 1

                        results["puzzle_indices"].append(example_id)
                        results["puzzle_identifiers"].append(identifier_map[pid])

                        puzzle_id += 1

                        total_puzzles += 1

                    # Push group
                    results["group_indices"].append(puzzle_id)
                    total_groups += 1

            for k in list(results.keys()):
                v = results.pop(k)        # drop the source list ref before stacking
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)

                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)
                del v                       # free the stacked array before the next key

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"

            pad_id=0,
            ignore_label_id=0,

            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,

            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            sets=list(split.keys())
        )

        # Save metadata as JSON.
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}

        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)

    # Clean up the grid spill.
    shutil.rmtree(spill_dir, ignore_errors=True)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
