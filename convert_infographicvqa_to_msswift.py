#!/usr/bin/env python3
"""
Convert InfographicVQA dataset to ms-swift format.

This script converts the InfographicVQA dataset from its original JSON format
to the format required by ms-swift for training vision-language models.

Usage:
    python convert_infographicvqa_to_msswift.py \
        --dataset_dir /path/to/InfographicVQA \
        --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def format_answer(answers):
    """
    Format the answer from the answers list.
    Uses the first answer if multiple answers exist.
    """
    if isinstance(answers, list):
        if len(answers) > 0:
            return str(answers[0]).strip()
        return ''
    elif isinstance(answers, str):
        return answers.strip()
    return str(answers).strip()


def convert_to_msswift_format(entry,
                              dataset_dir,
                              images_dir,
                              variant: str = 'sft'):
    """
    Convert a single entry to ms-swift format.
    
    Args:
        entry: Dictionary with entry data from InfographicVQA JSON
        dataset_dir: Path to the dataset root directory
        images_dir: Path to the images directory (absolute)
        variant: Output variant ('sft', 'sft_reason', 'grpo')
    
    Returns:
        Dictionary in ms-swift format or None if conversion fails
    """
    try:
        # Get question
        question = entry.get('question', '').strip()
        if not question:
            return None
        
        # Get answer
        answers = entry.get('answers', [])
        if not answers or len(answers) == 0:
            return None
        
        answer = format_answer(answers)
        if not answer:
            return None
        
        # Get image path
        image_local_name = entry.get('image_local_name', '')
        if not image_local_name:
            return None
        
        # Construct absolute image path
        image_path = os.path.join(images_dir, image_local_name)
        
        # Verify image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
        
        # Get question ID for unique identifier
        question_id = entry.get('questionId', '')
        entry_id = f"infographicvqa_{question_id}"
        
        # SFT variants (default)
        if variant in {'sft', 'sft_reason'}:
            # Standard InfoVQA prompt format
            prompt = (
                f"<image>{question}\n"
                "Answer the question using a single word or phrase."
            )
            
            response = answer
            
            # Note: InfographicVQA doesn't have reasoning, but keeping structure for consistency
            if variant == 'sft_reason':
                # Could potentially extract reasoning from 'operation/reasoning' if needed
                operation_reasoning = entry.get('operation/reasoning', [])
                if operation_reasoning and len(operation_reasoning) > 0:
                    reasoning_text = ' '.join(str(r) for r in operation_reasoning if r)
                    if reasoning_text.strip():
                        response = f"<think>{reasoning_text}</think>{answer}"
            
            result = {
                'id': entry_id,
                'image': image_path,  # Absolute path
                'conversations': [
                    {
                        'from': 'human',
                        'value': prompt
                    },
                    {
                        'from': 'gpt',
                        'value': response
                    }
                ]
            }
        
        elif variant == 'grpo':
            # GRPO-style sample: user prompt only; solution used by reward
            prompt = (
                f"<image>{question}\n"
                "Answer the question using a single word or phrase. "
                "Think step by step, but put the final answer inside <answer></answer> tags."
            )
            
            result = {
                'id': entry_id,
                'images': [image_path],  # Absolute path
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }],
                # Used by GRPO reward; keep raw answer (no tags)
                'solution': answer
            }
            
            # Extract reasoning if available
            operation_reasoning = entry.get('operation/reasoning', [])
            if operation_reasoning and len(operation_reasoning) > 0:
                reasoning_text = ' '.join(str(r) for r in operation_reasoning if r)
                if reasoning_text.strip():
                    result['reasoning'] = reasoning_text.strip()
        
        else:
            raise ValueError(f'Unknown variant: {variant}')
        
        return result
    
    except Exception as e:
        print(f"Error converting entry {entry.get('questionId', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_dataset(json_file,
                   dataset_dir,
                   images_dir,
                   output_file,
                   split_name,
                   variant: str = 'sft',
                   max_samples: Optional[int] = None):
    """
    Convert a dataset split (train or val) to ms-swift format.
    
    Args:
        json_file: Path to the input JSON file
        dataset_dir: Path to the dataset root directory
        images_dir: Absolute path to the images directory
        output_file: Path to the output JSONL file
        split_name: Name of the split (for progress display)
        variant: Output variant ('sft', 'sft_reason', 'grpo')
        max_samples: Maximum number of samples to convert (None = all)
    """
    print(f"\nConverting {split_name} dataset...")
    print(f"  Input: {json_file}")
    print(f"  Output: {output_file}")
    if max_samples:
        print(f"  Max samples: {max_samples}")
    
    # Load JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract entries from the data structure
    if isinstance(data, dict) and 'data' in data:
        entries = data['data']
        print(f"  Dataset: {data.get('dataset_name', 'Unknown')}")
        print(f"  Version: {data.get('dataset_version', 'Unknown')}")
    elif isinstance(data, list):
        entries = data
    else:
        print(f"Error: Unexpected data format in {json_file}")
        return {
            'converted': 0,
            'skipped': 0,
            'unique_images': 0
        }
    
    print(f"  Total entries: {len(entries)}")
    
    # Limit samples if requested
    if max_samples and max_samples < len(entries):
        entries = entries[:max_samples]
        print(f"  Limited to: {len(entries)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Convert entries
    converted_count = 0
    skipped_count = 0
    unique_images = set()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in tqdm(entries, desc=f"Converting {split_name}"):
            # Convert to ms-swift format
            converted = convert_to_msswift_format(entry, dataset_dir, images_dir, variant)
            
            if converted:
                out_f.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # Track unique images
                if variant == 'grpo':
                    # GRPO format uses 'images' list
                    images = converted.get('images', [])
                    unique_images.update(images)
                else:
                    # SFT format uses 'image' string
                    image = converted.get('image')
                    if image:
                        unique_images.add(image)
            else:
                skipped_count += 1
    
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Unique images: {len(unique_images)}")
    print(f"  Output saved to: {output_file}")
    
    return {
        'converted': converted_count,
        'skipped': skipped_count,
        'unique_images': len(unique_images)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert InfographicVQA dataset to ms-swift format'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to the InfographicVQA dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory for converted datasets'
    )
    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Only convert training set'
    )
    parser.add_argument(
        '--val_only',
        action='store_true',
        help='Only convert validation set'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['sft', 'sft_reason', 'grpo'],
        default='sft',
        help='Output variant: sft (default), sft_reason (prepend reasoning if available), grpo (messages+solution for GRPO)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to convert (useful for testing). If not specified, converts all samples.'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Verify dataset directory exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        return
    
    # Verify annotations directory exists
    annotations_dir = dataset_dir / 'annotations'
    if not annotations_dir.exists():
        print(f"Error: Annotations directory does not exist: {annotations_dir}")
        return
    
    # Verify images directory exists
    images_dir = dataset_dir / 'images'
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir}")
        return
    
    # Get absolute path to images directory
    images_dir_abs = str(images_dir.resolve())
    
    # Verify required files exist
    train_json = annotations_dir / 'infographicsVQA_train_v1.0.json'
    val_json = annotations_dir / 'infographicsVQA_val_v1.0_withQT.json'
    
    if not train_json.exists() and not args.val_only:
        print(f"Warning: Training JSON not found: {train_json}")
        print("  Attempting alternative filename: infographicsVQA_train_v1.0.json")
        if not train_json.exists():
            print(f"Error: Training annotations not found")
            return
    
    if not val_json.exists() and not args.train_only:
        print(f"Warning: Validation JSON not found: {val_json}")
        # Try alternative filename
        val_json_alt = annotations_dir / 'infographicsVQA_val_v1.0.json'
        if val_json_alt.exists():
            val_json = val_json_alt
            print(f"  Using alternative: {val_json}")
        else:
            print(f"Error: Validation annotations not found")
            return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert training set
    train_stats = None
    if not args.val_only and train_json.exists():
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        # Add size suffix if max_samples is specified
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        train_output = output_dir / f'infographicvqa_train{suffix}.jsonl'
        train_stats = convert_dataset(
            str(train_json),
            str(dataset_dir),
            images_dir_abs,
            str(train_output),
            'train',
            args.variant,
            args.max_samples
        )
    
    # Convert validation set
    val_stats = None
    if not args.train_only and val_json.exists():
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        # Add size suffix if max_samples is specified
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        val_output = output_dir / f'infographicvqa_val{suffix}.jsonl'
        val_stats = convert_dataset(
            str(val_json),
            str(dataset_dir),
            images_dir_abs,
            str(val_output),
            'val',
            args.variant,
            args.max_samples
        )
    
    print("\n" + "="*60)
    print("Conversion completed!")
    print("="*60)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print("="*60)
    if train_stats:
        print(f"Training Set:")
        print(f"  Total QAs: {train_stats['converted']}")
        print(f"  Unique Images: {train_stats['unique_images']}")
        print(f"  Skipped: {train_stats['skipped']}")
    if val_stats:
        print(f"\nValidation Set:")
        print(f"  Total QAs: {val_stats['converted']}")
        print(f"  Unique Images: {val_stats['unique_images']}")
        print(f"  Skipped: {val_stats['skipped']}")
    
    print(f"\nOutput files:")
    if not args.val_only and train_stats:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        train_file = f'infographicvqa_train{suffix}.jsonl'
        print(f"  Train: {output_dir / train_file}")
    if not args.train_only and val_stats:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        val_file = f'infographicvqa_val{suffix}.jsonl'
        print(f"  Val: {output_dir / val_file}")
    
    print(f"\nTo use in ms-swift, update your training script:")
    if not args.val_only and train_stats:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        train_file = f'infographicvqa_train{suffix}.jsonl'
        print(f"  --dataset '{output_dir / train_file}'")


if __name__ == '__main__':
    main()

