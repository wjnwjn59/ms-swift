#!/usr/bin/env python3
"""
Convert NarrativeInfoVQA dataset to ms-swift format.

This script converts the NarrativeInfoVQA dataset from its original format
to the format required by ms-swift for training vision-language models.

Usage:
    python convert_narrativeinfovqa_to_msswift.py \
        --dataset_dir /path/to/NarrativeInfoVQA \
        --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def load_qa_file(qa_file_path):
    """Load a QA JSON file."""
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_qa_pair(qa_data, qa_type, qa_index):
    """Extract a specific QA pair from the QA data."""
    if qa_type == 'original':
        qa_pairs = qa_data.get('original_qa_pairs', [])
    elif qa_type == 'generated':
        qa_pairs = qa_data.get('generated_qa_pairs', [])
    else:
        return None
    
    if qa_index < len(qa_pairs):
        return qa_pairs[qa_index]
    return None


def format_answer(answer):
    """Format the answer from the QA pair."""
    if isinstance(answer, str):
        # Handle string answers like "['answer']"
        if answer.startswith('[') and answer.endswith(']'):
            try:
                answer_list = eval(answer)
                if isinstance(answer_list, list) and len(answer_list) > 0:
                    return answer_list[0]
            except:
                pass
        return answer
    elif isinstance(answer, dict) and 'text' in answer:
        # Handle dict format with 'text' key
        text_list = answer['text']
        if isinstance(text_list, list) and len(text_list) > 0:
            return text_list[0]
        return str(text_list)
    return str(answer)


def extract_reasoning(qa_pair: dict) -> str:
    """Best-effort extraction of reasoning text from the QA pair."""
    reasoning = qa_pair.get('reasoning', {}) or {}
    # Prefer the generated_reasoning.think.logical_reasoning field
    gen = reasoning.get('generated_reasoning', {}) if isinstance(reasoning, dict) else {}
    think = gen.get('think', {}) if isinstance(gen, dict) else {}
    logical = think.get('logical_reasoning', '')
    if isinstance(logical, str) and logical.strip():
        return logical.strip()
    # Fallbacks
    for key in ['reasoning_full', 'reasoning_no_bbox', 'reasoning_no_spatial', 'reasoning_short']:
        val = reasoning.get(key, '')
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ''


def convert_to_msswift_format(annotation,
                              qa_data,
                              dataset_dir,
                              use_absolute_paths=False,
                              variant: str = 'sft',
                              exclude_unanswerable: bool = False,
                              unanswerable_only: bool = False):
    """
    Convert a single annotation to ms-swift format.
    
    Args:
        annotation: Dictionary with annotation data
        qa_data: Dictionary with QA data loaded from qas file
        dataset_dir: Path to the dataset directory
        use_absolute_paths: If True, use absolute paths for images
        variant: Output variant ('sft', 'sft_reason', 'grpo')
        exclude_unanswerable: If True, exclude questions marked as unanswerable
        unanswerable_only: If True, include only questions marked as unanswerable
    
    Returns:
        Dictionary in ms-swift format or None if conversion fails
    """
    try:
        # Extract QA pair
        qa_type = annotation.get('qa_type', 'original')
        qa_index = annotation.get('qa_index', 0)
        qa_pair = get_qa_pair(qa_data, qa_type, qa_index)
        
        if qa_pair is None:
            return None
        
        # Get question and answer
        question = qa_pair.get('question', '')
        answer = format_answer(qa_pair.get('answer', ''))
        
        if not question or not answer:
            return None
        
        # Handle unanswerable questions
        is_unanswerable = qa_pair.get('is_unanswerable', False)
        
        # Filter based on unanswerable status
        if exclude_unanswerable and is_unanswerable:
            return None  # Skip unanswerable questions
        if unanswerable_only and not is_unanswerable:
            return None  # Skip answerable questions (only keep unanswerable)
        
        if is_unanswerable:
            answer = 'unanswerable'
        
        # Get image path
        image_id = annotation.get('image_id')
        if image_id is None:
            return None
        
        image_filename = f"{image_id}.png"
        image_path = os.path.join('images', image_filename)
        
        # Verify image exists
        full_image_path = os.path.join(dataset_dir, image_path)
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            return None
        
        # Use absolute path if requested
        if use_absolute_paths:
            image_path = os.path.abspath(full_image_path)
        
        reasoning_text = extract_reasoning(qa_pair)

        # SFT variants (default)
        if variant in {'sft', 'sft_reason'}:
            prompt = (
                "<image>Answer the question according to the image using a single word or phrase. "
                "If the image does not contain enough evidence, answer exactly: unanswerable. "
                "Do not use outside knowledge.\n"
                f"Question: {question}"
            )

            if variant == 'sft_reason' and reasoning_text:
                response = f"<think>{reasoning_text}</think>{answer}"
            else:
                response = answer

            result = {
                'id': annotation.get('id', ''),
                'image': image_path,  # Relative or absolute path
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
                "<image>Answer the question according to the image using a single word or phrase. "
                "If insufficient evidence, respond with: unanswerable. "
                "Think step by step, but put the final answer inside <answer></answer> tags."
                f"\nQuestion: {question}"
            )

            result = {
                'id': annotation.get('id', ''),
                'images': [image_path],
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }],
                # Used by GRPO reward; keep raw answer (no tags)
                'solution': answer
            }
            if reasoning_text:
                result['reasoning'] = reasoning_text

        else:
            raise ValueError(f'Unknown variant: {variant}')
        
        return result
    
    except Exception as e:
        print(f"Error converting annotation {annotation.get('id', 'unknown')}: {e}")
        return None


def convert_dataset(annotations_file,
                   dataset_dir,
                   output_file,
                   split_name,
                   use_absolute_paths=False,
                   variant: str = 'sft',
                   max_samples: Optional[int] = None,
                   exclude_unanswerable: bool = False,
                   unanswerable_only: bool = False):
    """
    Convert a dataset split (train or val) to ms-swift format.
    
    Args:
        annotations_file: Path to the annotations JSONL file
        dataset_dir: Path to the dataset root directory
        output_file: Path to the output JSONL file
        split_name: Name of the split (for progress display)
        use_absolute_paths: If True, use absolute paths for images
        variant: Output variant ('sft', 'sft_reason', 'grpo')
        max_samples: Maximum number of samples to convert (None = all)
        exclude_unanswerable: If True, exclude questions marked as unanswerable
        unanswerable_only: If True, include only questions marked as unanswerable
    """
    print(f"\nConverting {split_name} dataset...")
    print(f"  Input: {annotations_file}")
    print(f"  Output: {output_file}")
    if max_samples:
        print(f"  Max samples: {max_samples}")
    if exclude_unanswerable:
        print(f"  Excluding unanswerable questions")
    if unanswerable_only:
        print(f"  Including only unanswerable questions")
    
    # Load all annotations
    annotations = []
    with open(annotations_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    
    print(f"  Total annotations: {len(annotations)}")
    
    # Limit samples if requested
    if max_samples and max_samples < len(annotations):
        annotations = annotations[:max_samples]
        print(f"  Limited to: {len(annotations)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Track QA files we've loaded to avoid reloading
    qa_cache = {}
    
    # Convert annotations
    converted_count = 0
    skipped_count = 0
    unanswerable_count = 0
    answerable_count = 0
    unique_images = set()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for annotation in tqdm(annotations, desc=f"Converting {split_name}"):
            # Get QA file path
            qas_file = annotation.get('qas_file')
            if not qas_file:
                skipped_count += 1
                continue
            
            qa_file_path = os.path.join(dataset_dir, qas_file)
            if not os.path.exists(qa_file_path):
                print(f"Warning: QA file not found: {qa_file_path}")
                skipped_count += 1
                continue
            
            # Load QA file (with caching)
            if qa_file_path not in qa_cache:
                try:
                    qa_cache[qa_file_path] = load_qa_file(qa_file_path)
                except Exception as e:
                    print(f"Error loading QA file {qa_file_path}: {e}")
                    skipped_count += 1
                    continue
            
            qa_data = qa_cache[qa_file_path]
            
            # Check unanswerable status before conversion for statistics
            qa_type = annotation.get('qa_type', 'original')
            qa_index = annotation.get('qa_index', 0)
            qa_pair = get_qa_pair(qa_data, qa_type, qa_index)
            is_unanswerable = qa_pair.get('is_unanswerable', False) if qa_pair else False
            
            # Convert to ms-swift format
            converted = convert_to_msswift_format(annotation, qa_data, dataset_dir, use_absolute_paths, variant, exclude_unanswerable, unanswerable_only)
            
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
                # Track why it was skipped
                if exclude_unanswerable and is_unanswerable:
                    unanswerable_count += 1
                elif unanswerable_only and not is_unanswerable:
                    answerable_count += 1
    
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    if exclude_unanswerable:
        print(f"  Excluded (unanswerable): {unanswerable_count}")
    if unanswerable_only:
        print(f"  Excluded (answerable): {answerable_count}")
    print(f"  Unique images: {len(unique_images)}")
    print(f"  Output saved to: {output_file}")
    
    return {
        'converted': converted_count,
        'skipped': skipped_count,
        'unanswerable_excluded': unanswerable_count if exclude_unanswerable else 0,
        'answerable_excluded': answerable_count if unanswerable_only else 0,
        'unique_images': len(unique_images)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert NarrativeInfoVQA dataset to ms-swift format'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to the NarrativeInfoVQA dataset directory'
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
        '--use_absolute_paths',
        action='store_true',
        help='Use absolute paths for images (recommended if images are in a different location)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['sft', 'sft_reason', 'grpo'],
        default='sft',
        help='Output variant: sft (default), sft_reason (prepend reasoning), grpo (messages+solution for GRPO)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to convert (useful for testing). If not specified, converts all samples.'
    )
    parser.add_argument(
        '--exclude_unanswerable',
        action='store_true',
        help='Exclude questions marked as unanswerable from the dataset'
    )
    parser.add_argument(
        '--unanswerable_only',
        action='store_true',
        help='Include only questions marked as unanswerable (exclude answerable questions)'
    )
    
    args = parser.parse_args()
    
    # Validate that both options are not set at the same time
    if args.exclude_unanswerable and args.unanswerable_only:
        parser.error('--exclude_unanswerable and --unanswerable_only cannot be used together')
    
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Verify dataset directory exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        return
    
    # Verify required files exist
    train_annotations = dataset_dir / 'train_annotations.jsonl'
    val_annotations = dataset_dir / 'val_annotations.jsonl'
    
    if not train_annotations.exists() and not args.val_only:
        print(f"Error: Training annotations not found: {train_annotations}")
        return
    
    if not val_annotations.exists() and not args.train_only:
        print(f"Error: Validation annotations not found: {val_annotations}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert training set
    train_stats = None
    if not args.val_only:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        # Add size suffix if max_samples is specified
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        # Add suffix if filtering unanswerable
        if args.exclude_unanswerable:
            suffix = f"{suffix}_no_unans" if suffix else "_no_unans"
        elif args.unanswerable_only:
            suffix = f"{suffix}_unans_only" if suffix else "_unans_only"
        train_output = output_dir / f'narrativeinfovqa_train{suffix}.jsonl'
        train_stats = convert_dataset(
            str(train_annotations),
            str(dataset_dir),
            str(train_output),
            'train',
            args.use_absolute_paths,
            args.variant,
            args.max_samples,
            args.exclude_unanswerable,
            args.unanswerable_only
        )
    
    # Convert validation set
    val_stats = None
    if not args.train_only:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        # Add size suffix if max_samples is specified
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        # Add suffix if filtering unanswerable
        if args.exclude_unanswerable:
            suffix = f"{suffix}_no_unans" if suffix else "_no_unans"
        elif args.unanswerable_only:
            suffix = f"{suffix}_unans_only" if suffix else "_unans_only"
        val_output = output_dir / f'narrativeinfovqa_val{suffix}.jsonl'
        val_stats = convert_dataset(
            str(val_annotations),
            str(dataset_dir),
            str(val_output),
            'val',
            args.use_absolute_paths,
            args.variant,
            args.max_samples,
            args.exclude_unanswerable,
            args.unanswerable_only
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
        if args.exclude_unanswerable:
            print(f"  Excluded (unanswerable): {train_stats['unanswerable_excluded']}")
        if args.unanswerable_only:
            print(f"  Excluded (answerable): {train_stats['answerable_excluded']}")
    if val_stats:
        print(f"\nValidation Set:")
        print(f"  Total QAs: {val_stats['converted']}")
        print(f"  Unique Images: {val_stats['unique_images']}")
        if args.exclude_unanswerable:
            print(f"  Excluded (unanswerable): {val_stats['unanswerable_excluded']}")
        if args.unanswerable_only:
            print(f"  Excluded (answerable): {val_stats['answerable_excluded']}")
    
    print(f"\nOutput files:")
    if not args.val_only:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        if args.exclude_unanswerable:
            suffix = f"{suffix}_no_unans" if suffix else "_no_unans"
        elif args.unanswerable_only:
            suffix = f"{suffix}_unans_only" if suffix else "_unans_only"
        train_file = f'narrativeinfovqa_train{suffix}.jsonl'
        print(f"  Train: {output_dir / train_file}")
    if not args.train_only:
        suffix = f"_{args.variant}" if args.variant != 'sft' else ''
        if args.max_samples:
            suffix = f"{suffix}_n{args.max_samples}" if suffix else f"_n{args.max_samples}"
        if args.exclude_unanswerable:
            suffix = f"{suffix}_no_unans" if suffix else "_no_unans"
        elif args.unanswerable_only:
            suffix = f"{suffix}_unans_only" if suffix else "_unans_only"
        val_file = f'narrativeinfovqa_val{suffix}.jsonl'
        print(f"  Val: {output_dir / val_file}")
    print(f"\nTo use in ms-swift, update your training script:")
    if not args.val_only:
        print(f"  --dataset '{output_dir / train_file}'")


if __name__ == '__main__':
    main()

