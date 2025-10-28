import os
import json
import mmap
import pickle
from pathlib import Path
import random
from typing import List, Dict, Optional, Iterator, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import csv
import time
import os
import pickle
import gc
import psutil
import subprocess
import time

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import tiktoken
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TikTokenizer:
    """
    Tokenizer class using tiktoken with GPT-2 encoding (cl100k_base)
    Handles text tokenization, encoding, and decoding for transformer training
    """
    
    # def __init__(self, logger, encoding_name: str = "gpt2", vocab_size: int = 50257):
    #     """
    #     Initialize tokenizer with specified encoding
        
    #     Args:
    #         encoding_name: Tiktoken encoding name ("gpt2", "cl100k_base", etc.)
    #         vocab_size: Vocabulary size (50257 for GPT-2)
    #     """
    #     self.encoding_name = encoding_name
    #     self.vocab_size = vocab_size
    #     self.logger = logger
        
    #     try:
    #         self.encoding = tiktoken.get_encoding(encoding_name)
    #     except Exception as e:
    #         self.logger.warning(f"Failed to load {encoding_name}, falling back to gpt2: {e}")
    #         self.encoding = tiktoken.get_encoding("gpt2")
        
    #     # Special tokens
    #     self.bos_token = "<|endoftext|>"
    #     self.eos_token = "<|endoftext|>"  
    #     self.pad_token = "<|endoftext|>"
    #     self.unk_token = "<|endoftext|>"
        
    #     # Token IDs
    #     self.bos_token_id = self.encoding.encode(self.bos_token)[0]
    #     self.eos_token_id = self.encoding.encode(self.eos_token)[0] 
    #     self.pad_token_id = self.encoding.encode(self.pad_token)[0]
    #     self.unk_token_id = self.encoding.encode(self.unk_token)[0]
        
    #     self.logger.info(f"Tokenizer initialized with {encoding_name}, logger: {self.vocab_size}")
    
    def __init__(self, logger, encoding_name: str = "gpt2", vocab_size: int = 50257):
        """
        Initialize tokenizer with specified encoding
        
        Args:
            encoding_name: Tiktoken encoding name ("gpt2", "cl100k_base", etc.)
            vocab_size: Vocabulary size (50257 for GPT-2)
        """
        self.encoding_name = encoding_name
        self.vocab_size = vocab_size
        self.logger = logger
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            self.logger.warning(f"Failed to load {encoding_name}, falling back to gpt2: {e}")
            self.encoding = tiktoken.get_encoding("gpt2")
        
        # Special tokens
        self.bos_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"  
        self.pad_token = "<|endoftext|>"
        self.unk_token = "<|endoftext|>"
        
        # Token IDs - FIX: Allow special tokens when encoding
        try:
            self.bos_token_id = self.encoding.encode(self.bos_token, allowed_special={self.bos_token})[0]
            self.eos_token_id = self.encoding.encode(self.eos_token, allowed_special={self.eos_token})[0] 
            self.pad_token_id = self.encoding.encode(self.pad_token, allowed_special={self.pad_token})[0]
            self.unk_token_id = self.encoding.encode(self.unk_token, allowed_special={self.unk_token})[0]
        except Exception as e:
            # Fallback: use a known token ID for GPT-2
            self.logger.warning(f"Failed to encode special tokens, using fallback: {e}")
            self.bos_token_id = 50256  # <|endoftext|> token ID in GPT-2
            self.eos_token_id = 50256
            self.pad_token_id = 50256
            self.unk_token_id = 50256
        
        self.logger.info(f"Tokenizer initialized with {encoding_name}, vocab_size: {self.vocab_size}")
        self.logger.info(f"Special token IDs - BOS: {self.bos_token_id}, EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")

    
    # def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
    #     """
    #     Encode text to token IDs
        
    #     Args:
    #         text: Input text string
    #         add_special_tokens: Whether to add BOS/EOS tokens
            
    #     Returns:
    #         List of token IDs
    #     """
    #     token_ids = self.encoding.encode(text)
        
    #     if add_special_tokens:
    #         token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
    #     return token_ids
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # FIX: Allow special tokens in the text being encoded
        token_ids = self.encoding.encode(text, allowed_special={self.bos_token, self.eos_token}, disallowed_special=())
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        return token_ids

    
    # def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
    #     """
    #     Decode token IDs to text
        
    #     Args:
    #         token_ids: List of token IDs
    #         skip_special_tokens: Whether to skip special tokens in output
            
    #     Returns:
    #         Decoded text string
    #     """
    #     if skip_special_tokens:
    #         # Filter out special token IDs
    #         special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
    #         token_ids = [tid for tid in token_ids if tid not in special_ids]
        
    #     try:
    #         return self.encoding.decode(token_ids)
    #     except Exception as e:
    #         self.logger.warning(f"Decode error: {e}")
    #         return ""

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            # Filter out special token IDs
            special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        try:
            return self.encoding.decode(token_ids)
        except Exception as e:
            self.logger.warning(f"Decode error: {e}")
            # Fallback: try decoding without error checking
            try:
                return self.encoding.decode(token_ids, errors='ignore')
            except:
                return ""

    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size
    
    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text"""
        return len(self.encoding.encode(text))


class DatasetShardCreator:
    """
    Creates tokenized dataset shards from text files for efficient loading
    Supports multi-processing for fast tokenization of large datasets
    """
    
    def __init__(
        self,
        logger,
        tokenizer: TikTokenizer,
        shard_size: int = 1000000,  # 1M tokens per shard
        max_length: int = 1024,
        num_processes: int = None
    ):
        """
        Initialize shard creator
        
        Args:
            tokenizer: TikTokenizer instance
            shard_size: Number of tokens per shard
            max_length: Maximum sequence length
            num_processes: Number of processes for parallel processing
        """
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length
        self.logger = logger
        self.num_processes = num_processes or min(mp.cpu_count(), 8)
        
        self.logger.info(f"Shard creator initialized - shard_size: {shard_size}, max_length: {max_length}")
    
    def create_shards(
        self,
        input_file: str,
        output_dir: str,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Create dataset shards from input text file
        
        Args:
            input_file: Path to input text file
            output_dir: Directory to save shards
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with shard creation statistics
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        self.logger.info(f"Creating shards from {input_file}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Read and split text into chunks
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into paragraphs/documents
        documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
        
        # If no paragraph splits, split by sentences or use the whole text
        if len(documents) <= 1:
            # Try splitting by sentences
            sentences = [sent.strip() for sent in text.split('.') if sent.strip()]
            if len(sentences) > 2:
                documents = sentences
            else:
                # Use whole text as single document and create artificial split
                documents = [text.strip()]
        
        self.logger.info(f"Found {len(documents)} documents")
        
        # Handle small datasets - ensure at least 1 validation document
        if len(documents) < 10:  # Very small dataset
            if len(documents) == 1:
                # Split the single document roughly in half for train/val
                single_doc = documents[0]
                mid_point = len(single_doc) // 2
                # Find a good split point (space or punctuation)
                for i in range(mid_point - 100, mid_point + 100):
                    if i >= 0 and i < len(single_doc) and single_doc[i] in [' ', '.', '!', '?', '\n']:
                        mid_point = i
                        break
                
                train_docs = [single_doc[:mid_point].strip()]
                val_docs = [single_doc[mid_point:].strip()]
                
                # Filter out empty docs
                train_docs = [doc for doc in train_docs if doc]
                val_docs = [doc for doc in val_docs if doc]
                
                self.logger.info(f"Split single document: {len(train_docs)} train, {len(val_docs)} val")
            else:
                # Small dataset: use last document for validation
                val_docs = documents[-1:]
                train_docs = documents[:-1]
        else:
            # Normal split for larger datasets
            val_size = max(1, int(len(documents) * validation_split))  # At least 1 validation doc
            val_docs = documents[:val_size]
            train_docs = documents[val_size:]
        
        self.logger.info(f"Final split: {len(train_docs)} train documents, {len(val_docs)} validation documents")
        
        # Process train and validation sets
        train_stats = self._process_documents(
            train_docs, output_path / "train", "train"
        )
        val_stats = self._process_documents(
            val_docs, output_path / "val", "val"
        )
        
        # Save metadata
        metadata = {
            'tokenizer_config': {
                'encoding_name': self.tokenizer.encoding_name,
                'vocab_size': self.tokenizer.vocab_size,
                'max_length': self.max_length
            },
            'data_config': {
                'shard_size': self.shard_size,
                'validation_split': validation_split,
                'input_file': str(input_path),
                'total_documents': len(documents)
            },
            'statistics': {
                'train': train_stats,
                'validation': val_stats
            }
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Shard creation completed. Metadata saved to {metadata_path}")
        return metadata

    
    def _process_documents(
        self,
        documents: List[str],
        output_dir: Path,
        split_name: str
    ) -> Dict[str, Any]:
        """Process documents and create shards"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tokenize documents in parallel
        self.logger.info(f"Tokenizing {len(documents)} {split_name} documents...")
        
        chunk_size = max(1, len(documents) // self.num_processes)
        doc_chunks = [
            documents[i:i + chunk_size] 
            for i in range(0, len(documents), chunk_size)
        ]
        
        all_tokens = []
        total_tokens = 0
        total_sequences = 0
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                executor.submit(self._tokenize_chunk, chunk)
                for chunk in doc_chunks
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Tokenizing {split_name}"):
                chunk_tokens = future.result()
                all_tokens.extend(chunk_tokens)
                total_tokens += sum(len(seq) for seq in chunk_tokens)
                total_sequences += len(chunk_tokens)
        
        self.logger.info(f"Tokenized {total_sequences} sequences with {total_tokens} tokens")
        
        # Create shards
        shard_idx = 0
        current_shard = []
        current_shard_tokens = 0
        
        for sequence in tqdm(all_tokens, desc=f"Creating {split_name} shards"):
            current_shard.append(sequence)
            current_shard_tokens += len(sequence)
            
            if current_shard_tokens >= self.shard_size:
                # Save current shard
                shard_path = output_dir / f"shard_{shard_idx:04d}.pkl"
                with open(shard_path, 'wb') as f:
                    pickle.dump(current_shard, f)
                
                shard_idx += 1
                current_shard = []
                current_shard_tokens = 0
        
        # Save remaining tokens
        if current_shard:
            shard_path = output_dir / f"shard_{shard_idx:04d}.pkl"
            with open(shard_path, 'wb') as f:
                pickle.dump(current_shard, f)
            shard_idx += 1
        
        return {
            'num_shards': shard_idx,
            'total_sequences': total_sequences,
            'total_tokens': total_tokens,
            'avg_sequence_length': total_tokens / total_sequences if total_sequences > 0 else 0
        }
    
    def _tokenize_chunk(self, documents: List[str]) -> List[List[int]]:
        """Tokenize a chunk of documents"""
        tokenized_sequences = []
        
        for doc in documents:
            # Tokenize document
            token_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            
            # Split into sequences of max_length
            for i in range(0, len(token_ids), self.max_length):
                sequence = token_ids[i:i + self.max_length]
                
                # Pad sequence if necessary
                if len(sequence) < self.max_length:
                    sequence.extend([self.tokenizer.pad_token_id] * (self.max_length - len(sequence)))
                
                tokenized_sequences.append(sequence)
        
        return tokenized_sequences


class StreamingDataset(IterableDataset):
    """Memory-efficient streaming dataset with lazy loading of shards"""
    
    def __init__(
        self,
        logger,
        shard_dir: str,
        shuffle: bool = True,
        infinite: bool = True,
        max_length: int = 1024,
        allow_empty: bool = False
    ):
        self.shard_dir = Path(shard_dir)
        self.shuffle = shuffle
        self.infinite = infinite
        self.max_length = max_length
        self.allow_empty = allow_empty
        self.logger = logger
        
        if not self.shard_dir.exists():
            if allow_empty:
                self.logger.warning(f"Shard directory not found: {shard_dir} (allowing empty)")
                self.shard_files = []
            else:
                raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
        else:
            self.shard_files = sorted([f for f in self.shard_dir.glob("shard_*.pkl")])
            
            if not self.shard_files and not allow_empty:
                raise ValueError(f"No shard files found in {shard_dir}")
        
        self.logger.info(f"Found {len(self.shard_files)} shards in {shard_dir}")
        
        # Load metadata if available
        metadata_path = self.shard_dir.parent / "metadata.json"
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def __iter__(self):
        """Iterate through dataset with lazy loading"""
        if not self.shard_files:
            if self.allow_empty:
                dummy_sequence = [0] * self.max_length
                input_ids = torch.tensor(dummy_sequence[:-1], dtype=torch.long)
                target_ids = torch.tensor(dummy_sequence[1:], dtype=torch.long)
                
                yield {
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                    'attention_mask': torch.ones_like(input_ids)
                }
            return
        
        iteration_count = 0
        while True:
            shard_order = list(range(len(self.shard_files)))
            if self.shuffle:
                np.random.shuffle(shard_order)
            
            for shard_idx in shard_order:
                shard_file = self.shard_files[shard_idx]
                
                try:
                    with open(shard_file, 'rb') as f:
                        sequences = pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load shard {shard_file}: {e}")
                    continue
                
                if self.shuffle:
                    np.random.shuffle(sequences)
                
                for sequence in sequences:
                    if len(sequence) < self.max_length:
                        sequence.extend([0] * (self.max_length - len(sequence)))
                    else:
                        sequence = sequence[:self.max_length]
                    
                    input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                    target_ids = torch.tensor(sequence[1:], dtype=torch.long)
                    
                    yield {
                        'input_ids': input_ids,
                        'target_ids': target_ids,
                        'attention_mask': (input_ids != 0).long()
                    }
                    
                    iteration_count += 1
            
            if not self.infinite:
                break
    
    def __len__(self):
        """Return exact length by counting all sequences in all shards."""
        if not self.shard_files:
            return 1 if self.allow_empty else 0

        total = 0
        for shard_file in self.shard_files:
            try:
                with open(shard_file, 'rb') as f:
                    sequences = pickle.load(f)
                    # Clip/pad here just like in __iter__
                    total += len(sequences)
            except Exception:
                continue

        return total



def load_tokens(filename):
    """Load tokens from a shard file"""
    with open(filename, 'rb') as f:
        tokens = pickle.load(f)
    return torch.tensor(tokens, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, data_dir, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        
        # Get the shard filenames
        shard_dir = os.path.join(data_dir, split)
        shards = os.listdir(shard_dir)
        shards = [s for s in shards if s.endswith('.pkl')]
        shards = sorted(shards)
        shards = [os.path.join(shard_dir, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        
        # Handle case where we don't have enough tokens
        if len(buf) < B*T + 1:
            # Move to next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
            buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        
        # advance the position in the tensor
        self.current_position += B * T
        
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
            
        return x, y


class DatasetLoader:
    """
    High-level interface for loading and managing datasets
    Supports both regular and streaming data loading with optimal configurations
    """
    
    def __init__(
        self,
        logger,
        data_dir: str,
        batch_size: int = 32,
        max_length: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = True
    ):
        """
        Initialize dataset loader 
        
        Args:
            data_dir: Directory containing train/val shards
            batch_size: Batch size for data loading
            max_length: Maximum sequence length
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            shuffle_train: Whether to shuffle training data
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.logger = logger
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.logger.warning(f"No metadata found at {metadata_path}")
            self.metadata = {}
        
        self.logger.info(f"Dataset loader initialized - batch_size: {batch_size}, max_length: {max_length}")
        
    def get_train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        train_dataset = StreamingDataset(
            logger = self.logger,
            shard_dir=self.data_dir / "train",
            shuffle=self.shuffle_train,
            infinite=True,
            max_length=self.max_length
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # DISABLE multiprocessing for IterableDataset
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def get_val_dataloader(self) -> DataLoader:
        """Get validation data loader"""
        val_dataset = StreamingDataset(
            logger = self.logger,
            shard_dir=self.data_dir / "val",
            shuffle=False,
            infinite=False,
            max_length=self.max_length,
            allow_empty=True
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # DISABLE multiprocessing for IterableDataset
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        target_ids = torch.stack([item['target_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            'metadata': self.metadata,
            'train_shards': len(list((self.data_dir / "train").glob("shard_*.pkl"))),
            'val_shards': len(list((self.data_dir / "val").glob("shard_*.pkl"))),
            'batch_size': self.batch_size,
            'max_length': self.max_length
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics
    Calculates perplexity, BLEU, top-k accuracy, and other LM metrics
    """
    
    def __init__(
        self,
        logger,
        model: torch.nn.Module,
        tokenizer: TikTokenizer,
        device: str = "cuda",
        generation_config: Optional[Dict] = None,
        mixed_precision = False,
    ):
        """
        Initialize evaluator
        
        Args:
            model: Transformer model to evaluate
            tokenizer: Tokenizer for text processing
            device: Device for computation
            generation_config: Configuration for text generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.device = device
        self.mixed_precision = mixed_precision
        self.generation_config = generation_config or {
            'max_new_tokens': 50,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True
        }
        
        self.logger.info("Model evaluator initialized")
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from cross-entropy loss"""
        return math.exp(min(loss, 20))  # Cap to prevent overflow
    
    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate perplexity on validation data
        
        Args:
            dataloader: Validation data loader
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with perplexity metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        # FIX: Better tqdm setup for IterableDataset
        if max_batches:
            pbar = tqdm(total=max_batches, desc="Computing perplexity", leave=False)
        else:
            pbar = tqdm(desc="Computing perplexity", leave=False)
        
        try:
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                with torch.amp.autocast(self.device, enabled=self.mixed_precision):
                    logits, loss = self.model(input_ids, target_ids)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()
                num_batches += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'batches': num_batches})
                
                # Clear memory
                del input_ids, target_ids, attention_mask, logits, loss
                torch.cuda.empty_cache()
        
        finally:
            pbar.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = self.calculate_perplexity(avg_loss)
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_tokens': total_tokens
        }
        
        self.logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics

    @torch.no_grad()
    def evaluate_top_k_accuracy(
        self,
        dataloader: DataLoader,
        k_values: List[int] = [1, 5, 10],
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate top-k accuracy"""
        self.model.eval()
        
        correct_predictions = {f'top_{k}': 0 for k in k_values}
        total_predictions = 0
        
        # FIX: Better tqdm setup
        if max_batches:
            pbar = tqdm(total=max_batches, desc="Computing top-k accuracy", leave=False)
        else:
            pbar = tqdm(desc="Computing top-k accuracy", leave=False)
        
        try:
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                logits, _ = self.model(input_ids, target_ids)
                
                # Get predictions for each position
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = target_ids.view(-1)
                mask_flat = attention_mask.view(-1)
                
                # Only consider non-padded tokens
                valid_indices = mask_flat.bool()
                if valid_indices.sum() == 0:
                    pbar.update(1)
                    continue
                
                valid_logits = logits_flat[valid_indices]
                valid_targets = targets_flat[valid_indices]
                
                # Calculate top-k accuracy
                for k in k_values:
                    _, top_k_indices = torch.topk(valid_logits, k, dim=1)
                    correct = top_k_indices.eq(valid_targets.unsqueeze(1)).any(dim=1)
                    correct_predictions[f'top_{k}'] += correct.sum().item()
                
                total_predictions += valid_indices.sum().item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'total_preds': total_predictions})
                
                # Clear memory
                del input_ids, target_ids, attention_mask, logits
                torch.cuda.empty_cache()
        
        finally:
            pbar.close()
        
        # Calculate accuracies
        accuracies = {}
        for k in k_values:
            if total_predictions > 0:
                accuracies[f'top_{k}_accuracy'] = correct_predictions[f'top_{k}'] / total_predictions
            else:
                accuracies[f'top_{k}_accuracy'] = 0.0
        
        accuracies['total_predictions'] = total_predictions
        return accuracies

    @torch.no_grad()
    def generate_and_evaluate_text(
        self,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None,
        max_new_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Generate text and evaluate with BLEU score if references provided
        
        Args:
            prompts: List of input prompts
            reference_texts: Optional reference texts for BLEU calculation
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with generated texts and metrics
        """
        self.model.eval()
        
        generated_texts = []
        generation_times = []
        
        for prompt in tqdm(prompts, desc="Generating text"):
            start_time = time.time()
            
            # Tokenize prompt
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt, add_special_tokens=True),
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
            # Generate text
            generated_text = self._generate_text(input_ids, max_new_tokens)
            generated_texts.append(generated_text)
            
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # Clear memory
            del input_ids
            torch.cuda.empty_cache()
        
        results = {
            'generated_texts': generated_texts,
            'avg_generation_time': np.mean(generation_times),
            'prompts': prompts
        }
        
        # Calculate BLEU scores if references provided
        if reference_texts and len(reference_texts) == len(generated_texts):
            bleu_scores = self._calculate_bleu_scores(generated_texts, reference_texts)
            results.update(bleu_scores)
        
        return results
    
    def _generate_text(self, input_ids: torch.Tensor, max_new_tokens: int) -> str:
        """Generate text from input IDs using the model"""
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.amp.autocast('cuda', enabled=True):
                logits, _ = self.model(generated_ids[:, -self.model.context_length:])
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if self.generation_config['temperature'] != 1.0:
                next_token_logits = next_token_logits / self.generation_config['temperature']
            
            # Apply top-k filtering
            if self.generation_config.get('top_k', 0) > 0:
                top_k = min(self.generation_config['top_k'], next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p filtering
            if self.generation_config.get('top_p', 1.0) < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.generation_config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
                self.logger.debug(f'{torch.sum(indices_to_remove) = }')
            
            # Sample or take argmax
            if self.generation_config.get('do_sample', True):
                probs = torch.softmax(next_token_logits, dim=-1)
                self.logger.debug(f'{probs[:100] = }')
                self.logger.debug(f'{probs[-100:] = }')
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            self.logger.debug(f' generated {_} token as {next_token.item()}')
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            self.logger.debug(f' generated {_} token')
        self.logger.debug(f' [BREAKED] generated {_} tokens')
        # Decode generated text
        generated_tokens = generated_ids[0].tolist()
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def _calculate_bleu_scores(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU scores between generated and reference texts"""
        
        smoothing = SmoothingFunction()
        bleu_1_scores = []
        bleu_4_scores = []
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # Tokenize texts
            gen_tokens = nltk.word_tokenize(gen_text.lower())
            ref_tokens = nltk.word_tokenize(ref_text.lower())
            
            # Calculate BLEU-1
            bleu_1 = sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=smoothing.method1
            )
            bleu_1_scores.append(bleu_1)
            
            # Calculate BLEU-4
            bleu_4 = sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing.method1
            )
            bleu_4_scores.append(bleu_4)
        
        return {
            'bleu_1': np.mean(bleu_1_scores),
            'bleu_4': np.mean(bleu_4_scores),
            'bleu_1_std': np.std(bleu_1_scores),
            'bleu_4_std': np.std(bleu_4_scores)
        }
    
    def comprehensive_evaluation(
        self,
        dataloader: DataLoader,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        max_eval_batches: int = 100
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics
        
        Args:
            dataloader: Validation data loader
            prompts: Text generation prompts
            reference_texts: Reference texts for BLEU calculation
            output_file: CSV file to save detailed results
            max_eval_batches: Maximum batches for evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        # Calculate perplexity
        perplexity_metrics = self.evaluate_perplexity(dataloader, max_eval_batches)
        
        # Calculate top-k accuracy
        accuracy_metrics = self.evaluate_top_k_accuracy(dataloader, [1, 5, 10], max_eval_batches)
        
        # Generate and evaluate text
        generation_metrics = self.generate_and_evaluate_text(
            prompts, reference_texts, self.generation_config['max_new_tokens']
        )
        
        # Calculate additional metrics
        bpc = perplexity_metrics['avg_loss'] / math.log(2)  # Bits per character (approximation)
        neg_log_likelihood = perplexity_metrics['avg_loss']
        
        # Combine all metrics
        comprehensive_metrics = {
            'perplexity': perplexity_metrics['perplexity'],
            'neg_log_likelihood': neg_log_likelihood,
            'bits_per_character': bpc,
            'top_1_accuracy': accuracy_metrics['top_1_accuracy'],
            'top_5_accuracy': accuracy_metrics['top_5_accuracy'], 
            'top_10_accuracy': accuracy_metrics['top_10_accuracy'],
            'avg_generation_time': generation_metrics['avg_generation_time'],
            'total_eval_tokens': perplexity_metrics['total_tokens']
        }
        
        # Add BLEU scores if available
        if 'bleu_1' in generation_metrics:
            comprehensive_metrics.update({
                'bleu_1': generation_metrics['bleu_1'],
                'bleu_4': generation_metrics['bleu_4']
            })
        
        # Save detailed results to CSV if requested
        if output_file:
            self._save_evaluation_results(
                comprehensive_metrics,
                generation_metrics,
                output_file
            )
        
        self.logger.info("Comprehensive evaluation completed")
        return comprehensive_metrics
    
    def _save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        generation_results: Dict[str, Any],
        output_file: str
    ):
        """Save evaluation results to CSV file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write metrics and generated samples
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write metrics header
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                writer.writerow([key, f"{value:.4f}" if isinstance(value, (int, float)) else str(value)])
            
            # Write generation samples
            writer.writerow([])  # Empty row
            writer.writerow(['Prompt', 'Generated Text'])
            
            for prompt, generated in zip(
                generation_results['prompts'],
                generation_results['generated_texts']
            ):
                writer.writerow([prompt, generated])
        
        self.logger.info(f"Evaluation results saved to {output_file}")


# Utility functions for data validation
def validate_data_shards(logger, data_dir: str) -> bool:
    """
    Validate that data shards are properly formatted and accessible
    
    Args:
        data_dir: Directory containing data shards
        
    Returns:
        True if validation passes, False otherwise
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Check for train and validation directories
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    if not train_dir.exists():
        logger.error(f"Train directory not found: {train_dir}")
        return False
    
    if not val_dir.exists():
        logger.error(f"Validation directory not found: {val_dir}")
        return False
    
    # Check for shard files
    train_shards = list(train_dir.glob("shard_*.pkl"))
    val_shards = list(val_dir.glob("shard_*.pkl"))
    
    if not train_shards:
        logger.error(f"No training shards found in {train_dir}")
        return False
    
    if not val_shards:
        logger.error(f"No validation shards found in {val_dir}")
        return False
    
    # Test loading a sample shard
    try:
        with open(train_shards[0], 'rb') as f:
            sample_data = pickle.load(f)
        
        if not isinstance(sample_data, list) or not sample_data:
            logger.error(f"Invalid shard format in {train_shards[0]}")
            return False
        
        # Check first sequence
        if not isinstance(sample_data[0], list) or not sample_data[0]:
            logger.error(f"Invalid sequence format in {train_shards[0]}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to load sample shard {train_shards[0]}: {e}")
        return False
    
    logger.info(f"Data validation passed - {len(train_shards)} train shards, {len(val_shards)} val shards")
    return True


def estimate_dataset_size(logger, data_dir: str) -> Dict[str, Any]:
    """
    Estimate dataset size and statistics
    
    Args:
        data_dir: Directory containing data shards
        
    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    
    # Load metadata if available
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('statistics', {})
    
    # Estimate from shard files
    train_shards = list((data_path / "train").glob("shard_*.pkl"))
    val_shards = list((data_path / "val").glob("shard_*.pkl"))
    
    stats = {
        'train': {'num_shards': len(train_shards)},
        'validation': {'num_shards': len(val_shards)}
    }
    
    # Sample a few shards to estimate
    if train_shards:
        try:
            total_sequences = 0
            total_tokens = 0
            
            sample_shards = train_shards[:min(3, len(train_shards))]
            
            for shard_file in sample_shards:
                with open(shard_file, 'rb') as f:
                    sequences = pickle.load(f)
                total_sequences += len(sequences)
                total_tokens += sum(len(seq) for seq in sequences)
            
            # Extrapolate to all shards
            avg_sequences_per_shard = total_sequences / len(sample_shards)
            avg_tokens_per_shard = total_tokens / len(sample_shards)
            
            stats['train'].update({
                'estimated_sequences': int(avg_sequences_per_shard * len(train_shards)),
                'estimated_tokens': int(avg_tokens_per_shard * len(train_shards)),
                'avg_sequence_length': total_tokens / total_sequences if total_sequences > 0 else 0
            })
            
        except Exception as e:
            logger.warning(f"Failed to estimate dataset size: {e}")
    
    return stats

def get_gpu_memory_info(logger):
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return {}
    
    try:
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(0) / (1024**3)   # GB
        free = gpu_memory - reserved  # GB
        
        return {
            'total_gb': gpu_memory,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'utilization_pct': (allocated / gpu_memory) * 100
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return {}

def clear_gpu_cache():
    """Comprehensive GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def find_optimal_batch_size(
    logger,
    model_config: Dict,
    tokenizer,
    data_dir: str,
    device: str = "cuda",
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    context_length: int = 1024,
    test_batch_sizes: List[int] = None,
    target_batch_size: int = None,
    safety_margin: float = 0.15,  # Reserve 15% GPU memory as buffer
    max_test_iterations: int = 5
) -> Dict[str, Any]:
    """
    Find optimal batch size for GPU memory and performance
    
    Args:
        model_config: Model configuration dictionary
        tokenizer: Tokenizer instance
        data_dir: Path to data directory
        device: Training device
        mixed_precision: Use mixed precision
        gradient_accumulation_steps: Gradient accumulation steps
        context_length: Sequence length
        test_batch_sizes: List of batch sizes to test (if None, auto-generate)
        target_batch_size: Specific batch size to test (if provided)
        safety_margin: Fraction of GPU memory to keep as buffer
        max_test_iterations: Max iterations to test per batch size
        
    Returns:
        Dictionary with optimal batch size recommendations and memory analysis
    """
    if device == "cpu":
        logger.warning("CPU training detected, batch size optimization not applicable")
        return {"optimal_batch_size": 32, "device": "cpu", "memory_analysis": {}}
    
    logger.info("🔍 Starting GPU memory profiling and batch size optimization...")
    
    # Clear GPU memory before starting
    clear_gpu_cache()
    
    # Get initial GPU state
    initial_gpu_info = get_gpu_memory_info(logger)
    if not initial_gpu_info:
        return {"error": "Could not access GPU information"}
    
    logger.info(f"📊 Initial GPU State:")
    logger.info(f"   Total Memory: {initial_gpu_info['total_gb']:.2f} GB")
    logger.info(f"   Free Memory: {initial_gpu_info['free_gb']:.2f} GB")
    
    # Determine batch sizes to test
    if target_batch_size:
        batch_sizes_to_test = [target_batch_size]
    elif test_batch_sizes:
        batch_sizes_to_test = test_batch_sizes
    else:
        # Auto-generate batch sizes based on GPU memory
        max_reasonable_bs = min(512, int(initial_gpu_info['free_gb'] * 16))  # Heuristic
        # batch_sizes_to_test = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        batch_sizes_to_test = [16, 64, 128, 192, 256, 384, 512, 1024, 2048]
        batch_sizes_to_test = [bs for bs in batch_sizes_to_test if bs <= max_reasonable_bs]
    
    logger.info(f"🧪 Testing batch sizes: {batch_sizes_to_test}")
    
    # Create test model
    from Transformer import DecoderOnlyTransformer
    
    logger.info("🏗️ Creating test model...")
    test_model = DecoderOnlyTransformer(**model_config)
    test_model = test_model.to(device)
    test_model.train()
    
    # Create test optimizer (for memory calculation)
    test_optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-4)
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
    
    # Get model memory usage
    model_memory = get_gpu_memory_info(logger)
    model_memory_gb = model_memory['allocated_gb'] - initial_gpu_info['allocated_gb']
    
    logger.info(f"🧠 Model loaded - Memory usage: {model_memory_gb:.2f} GB")
    
    # Results storage
    results = {
        'device_info': {
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'total_memory_gb': initial_gpu_info['total_gb'],
            'available_memory_gb': initial_gpu_info['free_gb'],
            'model_memory_gb': model_memory_gb
        },
        'batch_size_analysis': [],
        'optimal_batch_size': None,
        'max_safe_batch_size': None,
        'performance_recommendations': {},
        'memory_breakdown': {}
    }
    
    # Create dummy data loader for testing
    logger.info("📝 Creating test data...")
    try:
        data_loader = DatasetLoader(
            logger = logger,
            data_dir=data_dir,
            batch_size=1,  # Will be overridden
            max_length=context_length,
            num_workers=0,
            pin_memory=False
        )
        test_dataloader = data_loader.get_train_dataloader()
        
        # Get a few test batches
        test_batches = []
        batch_iter = iter(test_dataloader)
        for _ in range(min(3, max_test_iterations)):
            try:
                test_batches.append(next(batch_iter))
            except StopIteration:
                break
        
        if not test_batches:
            raise ValueError("No test batches available")
            
    except Exception as e:
        logger.error(f"Failed to create test data: {e}")
        return {"error": f"Data loading failed: {e}"}
    
    # Test each batch size
    successful_batch_sizes = []
    max_working_batch_size = 0
    
    for batch_size in batch_sizes_to_test:
        logger.info(f"\n🧪 Testing batch size: {batch_size}")
        
        try:
            # Clear cache before each test
            clear_gpu_cache()
            
            # Prepare batch data
            if len(test_batches[0]['input_ids']) < batch_size:
                # Repeat samples to reach desired batch size
                repetitions = (batch_size // len(test_batches[0]['input_ids'])) + 1
                input_ids = test_batches[0]['input_ids'].repeat(repetitions, 1)[:batch_size]
                target_ids = test_batches[0]['target_ids'].repeat(repetitions, 1)[:batch_size]
            else:
                input_ids = test_batches[0]['input_ids'][:batch_size]
                target_ids = test_batches[0]['target_ids'][:batch_size]
            
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            
            # Memory before forward pass
            pre_forward_memory = get_gpu_memory_info(logger)
            
            # Test forward pass
            start_time = time.time()
            
            with torch.amp.autocast('cuda', enabled=mixed_precision):
                logits, loss = test_model(input_ids, target_ids)
                loss = loss / gradient_accumulation_steps
            
            forward_time = time.time() - start_time
            
            # Memory after forward pass
            post_forward_memory = get_gpu_memory_info(logger)
            
            # Test backward pass
            start_time = time.time()
            
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(test_optimizer)
                scaler.update()
            else:
                loss.backward()
                test_optimizer.step()
            
            test_optimizer.zero_grad()
            
            backward_time = time.time() - start_time
            
            # Memory after backward pass (peak usage)
            peak_memory = get_gpu_memory_info(logger)
            
            # Calculate memory usage
            forward_memory_gb = post_forward_memory['allocated_gb'] - pre_forward_memory['allocated_gb']
            total_memory_gb = peak_memory['allocated_gb'] - initial_gpu_info['allocated_gb']
            
            # Calculate available memory with safety margin
            available_with_margin = initial_gpu_info['total_gb'] * (1 - safety_margin)
            memory_utilization = total_memory_gb / available_with_margin
            
            # Calculate throughput metrics
            total_tokens = batch_size * context_length
            tokens_per_sec_forward = total_tokens / forward_time
            tokens_per_sec_total = total_tokens / (forward_time + backward_time)
            
            # Check if this batch size fits safely
            fits_safely = total_memory_gb <= available_with_margin
            
            batch_analysis = {
                'batch_size': batch_size,
                'success': True,
                'fits_safely': fits_safely,
                'memory_usage': {
                    'forward_pass_gb': forward_memory_gb,
                    'total_memory_gb': total_memory_gb,
                    'peak_allocated_gb': peak_memory['allocated_gb'],
                    'memory_utilization_pct': memory_utilization * 100
                },
                'performance': {
                    'forward_time_ms': forward_time * 1000,
                    'backward_time_ms': backward_time * 1000,
                    'total_time_ms': (forward_time + backward_time) * 1000,
                    'tokens_per_sec_forward': tokens_per_sec_forward,
                    'tokens_per_sec_total': tokens_per_sec_total
                },
                'gpu_state': {
                    'allocated_gb': peak_memory['allocated_gb'],
                    'reserved_gb': peak_memory['reserved_gb'],
                    'free_gb': peak_memory['free_gb']
                }
            }
            
            results['batch_size_analysis'].append(batch_analysis)
            successful_batch_sizes.append(batch_size)
            
            if fits_safely:
                max_working_batch_size = batch_size
            
            logger.info(f"   ✅ Success - Memory: {total_memory_gb:.2f}GB ({memory_utilization*100:.1f}% util)")
            logger.info(f"   🚀 Performance: {tokens_per_sec_total:.0f} tok/s")
            
            # Clean up tensors
            del input_ids, target_ids, logits, loss
            clear_gpu_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"   ❌ OOM at batch size {batch_size}")
            
            batch_analysis = {
                'batch_size': batch_size,
                'success': False,
                'error': 'OutOfMemoryError',
                'fits_safely': False
            }
            results['batch_size_analysis'].append(batch_analysis)
            
            clear_gpu_cache()
            # Stop testing larger batch sizes
            break
            
        except Exception as e:
            logger.error(f"   ❌ Error at batch size {batch_size}: {e}")
            
            batch_analysis = {
                'batch_size': batch_size,
                'success': False,
                'error': str(e),
                'fits_safely': False
            }
            results['batch_size_analysis'].append(batch_analysis)
            
            clear_gpu_cache()
            continue
    
    # Analysis and recommendations
    if successful_batch_sizes:
        # Find optimal batch size (largest safe batch size with good performance)
        safe_batch_sizes = [
            analysis for analysis in results['batch_size_analysis'] 
            if analysis.get('success') and analysis.get('fits_safely')
        ]
        
        if safe_batch_sizes:
            # Choose based on performance efficiency
            optimal_analysis = max(safe_batch_sizes, key=lambda x: x.get('performance', {}).get('tokens_per_sec_total', 0))
            results['optimal_batch_size'] = optimal_analysis['batch_size']
            results['max_safe_batch_size'] = max_working_batch_size
        else:
            # No safe batch sizes, recommend the largest working one with warning
            results['optimal_batch_size'] = max(successful_batch_sizes)
            results['max_safe_batch_size'] = max(successful_batch_sizes)
            logger.warning("No batch sizes fit within safety margin!")
    
    # Generate recommendations
    results['performance_recommendations'] = generate_recommendations(results, initial_gpu_info, safety_margin)
    
    # Cleanup test model
    del test_model, test_optimizer
    if scaler:
        del scaler
    clear_gpu_cache()
    
    return results

def generate_recommendations(results: Dict, gpu_info: Dict, safety_margin: float) -> Dict[str, Any]:
    """Generate performance and hardware recommendations"""
    
    recommendations = {
        'optimal_settings': {},
        'hardware_advice': {},
        'memory_optimization': [],
        'performance_tips': []
    }
    
    gpu_total_gb = gpu_info['total_gb']
    optimal_bs = results.get('optimal_batch_size', 1)
    
    # Optimal settings
    if optimal_bs:
        recommendations['optimal_settings'] = {
            'recommended_batch_size': optimal_bs,
            'with_gradient_accumulation': {
                'batch_size': max(1, optimal_bs // 2),
                'grad_accum_steps': 2,
                'effective_batch_size': optimal_bs
            },
            'memory_efficient': {
                'batch_size': max(1, optimal_bs // 4),
                'grad_accum_steps': 4,
                'effective_batch_size': optimal_bs
            }
        }
    
    # Hardware advice based on GPU memory
    if gpu_total_gb < 8:
        recommendations['hardware_advice'] = {
            'category': 'Low-end GPU',
            'suitable_for': ['Small models (tiny/small)', 'Fine-tuning', 'Inference'],
            'limitations': ['Limited batch sizes', 'May need gradient accumulation'],
            'upgrade_recommendation': 'Consider GPU with 12-16GB VRAM for better training'
        }
    elif gpu_total_gb < 16:
        recommendations['hardware_advice'] = {
            'category': 'Mid-range GPU',
            'suitable_for': ['Small to medium models', 'Most training tasks'],
            'limitations': ['Large models may need optimization'],
            'upgrade_recommendation': 'Good for most use cases'
        }
    elif gpu_total_gb < 24:
        recommendations['hardware_advice'] = {
            'category': 'High-end GPU',
            'suitable_for': ['Large models', 'High batch size training', 'Research'],
            'limitations': ['Very large models may still need optimization'],
            'upgrade_recommendation': 'Excellent for most LLM training tasks'
        }
    else:
        recommendations['hardware_advice'] = {
            'category': 'Enterprise GPU',
            'suitable_for': ['Very large models', 'Production training', 'Multi-task training'],
            'limitations': ['None for typical use cases'],
            'upgrade_recommendation': 'Top-tier setup'
        }
    
    # Memory optimization tips
    if optimal_bs and optimal_bs < 32:
        recommendations['memory_optimization'].extend([
            'Enable gradient checkpointing to save memory',
            'Use mixed precision training (FP16)',
            'Reduce sequence length if possible',
            'Use gradient accumulation for effective larger batch sizes'
        ])
    
    if gpu_total_gb < 12:
        recommendations['memory_optimization'].extend([
            'Consider model parallelism for larger models',
            'Use CPU offloading for optimizer states',
            'Enable memory-efficient attention implementations'
        ])
    
    # Performance tips
    recommendations['performance_tips'].extend([
        f'Use batch sizes that are multiples of 8 for optimal GPU utilization',
        f'Consider gradient accumulation if memory-limited',
        f'Monitor GPU utilization during training',
        f'Use pin_memory=True for faster data transfer'
    ])
    
    return recommendations

def print_batch_size_analysis(results: Dict[str, Any]):
    """Print comprehensive batch size analysis results"""
    
    print("=" * 80)
    print("🚀 GPU MEMORY PROFILING & BATCH SIZE OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Device info
    device_info = results.get('device_info', {})
    print(f"\n📱 Device Information:")
    print(f"   GPU: {device_info.get('gpu_name', 'Unknown')}")
    print(f"   Total VRAM: {device_info.get('total_memory_gb', 0):.1f} GB")
    print(f"   Available VRAM: {device_info.get('available_memory_gb', 0):.1f} GB")
    print(f"   Model Memory: {device_info.get('model_memory_gb', 0):.2f} GB")
    
    # Batch size analysis
    print(f"\n📊 Batch Size Analysis:")
    print(f"{'Batch Size':<12} {'Status':<10} {'Memory (GB)':<12} {'Util %':<8} {'Tok/s':<10} {'Safe':<6}")
    print("-" * 70)
    
    for analysis in results.get('batch_size_analysis', []):
        bs = analysis['batch_size']
        
        if analysis.get('success', False):
            memory = analysis['memory_usage']['total_memory_gb']
            util = analysis['memory_usage']['memory_utilization_pct']
            perf = analysis['performance']['tokens_per_sec_total']
            safe = "✅" if analysis.get('fits_safely', False) else "⚠️"
            status = "✅ OK"
            
            print(f"{bs:<12} {status:<10} {memory:<12.2f} {util:<8.1f} {perf:<10.0f} {safe:<6}")
        else:
            error = analysis.get('error', 'Unknown')
            if 'OutOfMemory' in error:
                status = "❌ OOM"
            else:
                status = "❌ ERR"
            print(f"{bs:<12} {status:<10} {'N/A':<12} {'N/A':<8} {'N/A':<10} {'❌':<6}")
    
    # Recommendations
    optimal_bs = results.get('optimal_batch_size')
    max_safe_bs = results.get('max_safe_batch_size')
    
    print(f"\n🎯 Recommendations:")
    if optimal_bs:
        print(f"   Optimal Batch Size: {optimal_bs}")
        print(f"   Max Safe Batch Size: {max_safe_bs}")
    else:
        print("   ❌ No suitable batch size found!")
    
    # Settings recommendations
    recs = results.get('performance_recommendations', {})
    optimal_settings = recs.get('optimal_settings', {})
    
    if optimal_settings:
        print(f"\n⚙️ Optimal Training Settings:")
        print(f"   Direct: batch_size={optimal_settings.get('recommended_batch_size', 'N/A')}")
        
        if 'with_gradient_accumulation' in optimal_settings:
            ga_settings = optimal_settings['with_gradient_accumulation']
            print(f"   With Grad Accum: batch_size={ga_settings['batch_size']}, grad_accum={ga_settings['grad_accum_steps']}")
        
        if 'memory_efficient' in optimal_settings:
            me_settings = optimal_settings['memory_efficient']
            print(f"   Memory Efficient: batch_size={me_settings['batch_size']}, grad_accum={me_settings['grad_accum_steps']}")
    
    # Hardware advice
    hw_advice = recs.get('hardware_advice', {})
    if hw_advice:
        print(f"\n💻 Hardware Assessment:")
        print(f"   Category: {hw_advice.get('category', 'Unknown')}")
        print(f"   Suitable for: {', '.join(hw_advice.get('suitable_for', []))}")
        if hw_advice.get('limitations'):
            print(f"   Limitations: {', '.join(hw_advice.get('limitations', []))}")
        print(f"   Recommendation: {hw_advice.get('upgrade_recommendation', 'N/A')}")
    
    # Optimization tips
    mem_opt = recs.get('memory_optimization', [])
    if mem_opt:
        print(f"\n🛠️ Memory Optimization Tips:")
        for tip in mem_opt:
            print(f"   • {tip}")
    
    perf_tips = recs.get('performance_tips', [])
    if perf_tips:
        print(f"\n🚀 Performance Tips:")
        for tip in perf_tips:
            print(f"   • {tip}")
    
    print("=" * 80)



def find_best_checkpoint(logger, output_dir: str) -> Optional[str]:
    """
    Find the best checkpoint based on validation loss
    
    Args:
        output_dir: Training output directory
        
    Returns:
        Path to best checkpoint or None if not found
    """
    output_path = Path(output_dir)
    
    # Look for checkpoints in both checkpoints and milestones directories
    checkpoint_dirs = []
    for subdir in ['checkpoints', 'milestones']:
        checkpoint_dir = output_path / subdir
        if checkpoint_dir.exists():
            checkpoint_dirs.extend([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')])
    
    if not checkpoint_dirs:
        logger.warning(f"No checkpoints found in {output_dir}")
        return None
    
    best_checkpoint = None
    best_val_loss = float('inf')
    
    # Check each checkpoint for validation loss
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_file = checkpoint_dir / "checkpoint.pt"
        if not checkpoint_file.exists():
            continue
            
        try:
            # Load checkpoint metadata only
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = str(checkpoint_file)
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
            continue
    
    if best_checkpoint:
        logger.info(f"Best checkpoint found: {best_checkpoint} (val_loss: {best_val_loss:.4f})")
    else:
        logger.warning("No valid checkpoints with validation loss found")
        
    return best_checkpoint

def load_prompts_from_file(logger,file_path: str) -> List[str]:
    """Load prompts from a text file (one prompt per line)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    except Exception as e:
        logger.error(f"Failed to load prompts from {file_path}: {e}")
        return []

def save_generation_results(logger,results: Dict, output_file: str):
    """Save generation results to file"""
    output_path = Path(output_file)
    
    if output_path.suffix.lower() == '.json':
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # Save as text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TEXT GENERATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (prompt, generated_text) in enumerate(zip(results['prompts'], results['generated_texts'])):
                f.write(f"PROMPT {i+1}:\n{prompt}\n\n")
                f.write(f"GENERATED:\n{generated_text}\n\n")
                f.write("-" * 80 + "\n\n")
            
            f.write("STATISTICS:\n")
            f.write(f"Total prompts: {len(results['prompts'])}\n")
            f.write(f"Average generation time: {results['avg_generation_time']:.3f}s\n")
            f.write(f"Total generation time: {results['total_generation_time']:.3f}s\n")
            f.write(f"Throughput: {len(results['prompts']) / results['total_generation_time']:.2f} texts/sec\n")
    
    logger.info(f"Generation results saved to {output_path}")

class TextGenerator:
    """Standalone text generator for inference"""
    
    def __init__(
        self,
        logger,
        model,
        tokenizer,
        device: str = 'cuda',
        generation_config: Optional[Dict] = None
    ):
        self.logger = logger
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = generation_config or {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True
        }
        
        # Set model to eval mode
        self.model.eval()
        
        self.logger.info(f"TextGenerator initialized on {device}")
        self.logger.info(f"Generation config: {self.generation_config}")
    
    @torch.no_grad()
    def generate_interactive(self):
        """Interactive generation mode"""
        print("\n" + "="*60)
        print("🤖 INTERACTIVE TEXT GENERATION")
        print("="*60)
        print("Enter prompts to generate text. Type 'quit' to exit.")
        print(f"Config: temp={self.generation_config['temperature']}, "
              f"top_k={self.generation_config['top_k']}, "
              f"top_p={self.generation_config['top_p']}")
        print("-"*60)
        
        while True:
            try:
                prompt = input("\n📝 Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print("🔄 Generating...")
                start_time = time.time()
                
                # Tokenize
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt, add_special_tokens=True)[1:-1],
                    dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                # Generate
                attention_mask = torch.ones_like(input_ids)
                generated_texts = self._generate_text_batch(
                    input_ids, attention_mask, self.generation_config['max_new_tokens']
                )
                
                generation_time = time.time() - start_time
                
                print(f"\n✨ Generated text ({generation_time:.2f}s):")
                print(f"📄 {generated_texts[0]}")
                
                # Cleanup
                del input_ids, attention_mask
                torch.cuda.empty_cache()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _generate_text_batch(
        self, 
        input_ids_batch: torch.Tensor, 
        attention_mask_batch: torch.Tensor,
        max_new_tokens: int
    ) -> List[str]:
        """Batched text generation (same as in ModelEvaluator)"""
        # ... (use the same implementation as provided in the previous response)
        batch_size = input_ids_batch.shape[0]
        device = input_ids_batch.device
        
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        # generated_ids = input_ids_batch.clone()
        attention_mask = attention_mask_batch.clone()
        self.logger.debug(f'{max_new_tokens = }')
        print(f'{input_ids_batch = }')
        print(f'{max_new_tokens = }')
        
        
        
        generated_ids = self.model.generate(input_ids_batch, max_new_tokens)
        # for step in range(max_new_tokens):
            
            # current_length = generated_ids.shape[1]
            # if current_length > self.model.context_length:
            #     context_start = current_length - self.model.context_length
            #     context_ids = generated_ids[:, context_start:]
            # else:
            #     context_ids = generated_ids
            
            # with torch.amp.autocast('cuda', enabled=True):
            #     logits, _ = self.model(context_ids)
            
            # next_token_logits = logits[:, -1, :]
            # self.logger.debug(f'{next_token_logits.shape}')
            # print(f'{next_token_logits.shape}')
            # # Apply temperature
            # if self.generation_config['temperature'] != 1.0:
            #     next_token_logits = next_token_logits / self.generation_config['temperature']
            
            # # Apply top-k
            # if self.generation_config.get('top_k', 0) > 0:
            #     top_k = min(self.generation_config['top_k'], next_token_logits.size(-1))
            #     top_k_values, _ = torch.topk(next_token_logits, top_k, dim=1)
            #     min_top_k = top_k_values[:, -1:]
            #     next_token_logits = torch.where(
            #         next_token_logits < min_top_k,
            #         torch.full_like(next_token_logits, float('-inf')),
            #         next_token_logits
            #     )
            
            # # Apply top-p
            # if self.generation_config.get('top_p', 1.0) < 1.0:
            #     sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=1)
            #     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=1)
                
            #     sorted_indices_to_remove = cumulative_probs > self.generation_config['top_p']
            #     sorted_indices_to_remove[:, 0] = False
            #     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                
            #     indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
            #     indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            #     next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            # if self.generation_config.get('do_sample', True):
            # probs = torch.softmax(next_token_logits, dim=-1)
            # self.logger.debug(f'{probs.shape}')
            # print(f'{probs.shape}')
            # probs = probs / probs.sum(dim=-1, keepdim=True)
            # next_tokens = torch.multinomial(probs, num_samples=1)
            # else:
            #     next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # self.logger.debug(f'{next_tokens = }')
            # print(f'{next_tokens = }')
            
            # generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            # new_attention = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            # attention_mask = torch.cat([attention_mask, new_attention], dim=1)
            
            # eos_generated = (next_tokens.squeeze(1) == self.tokenizer.eos_token_id)
            # active_sequences = active_sequences & ~eos_generated
            
            # del logits, next_token_logits, next_tokens
        # print(f'Out')
        # Decode
        
        # generated_ids = torch.cat([input_ids_batch, next_tokens], dim=1)
        generated_texts = []
        for i in range(batch_size):
            tokens = generated_ids[i].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        """Generate text for multiple prompts"""
        all_results = []
        all_times = []
        
        # Expand prompts if multiple sequences requested
        if num_return_sequences > 1:
            expanded_prompts = []
            original_prompts = []
            for prompt in prompts:
                for _ in range(num_return_sequences):
                    expanded_prompts.append(prompt)
                    original_prompts.append(prompt)
            prompts = expanded_prompts
        
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_end = min(batch_start + batch_size, len(prompts))
            
            batch_prompts = prompts[batch_start:batch_end]
            self.logger.debug(f'{batch_prompts }')
            start_time = time.time()
            
            # Tokenize and pad batch
            batch_input_ids = []
            for prompt in batch_prompts:
                tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                # print(f'staring {len(tokens)} ? Here')
                batch_input_ids.append(tokens)
            
            max_length = max(len(tokens) for tokens in batch_input_ids)
            # print(f'max len {max_length} ? Here')
            self.logger.debug(f'{max_length = }')
            padded_input_ids = []
            padded_token_list = {}
            attention_masks = []
            
            for tokens in batch_input_ids:
                padding_length = max_length - len(tokens)
                rand_token = random.randint(0,self.tokenizer.vocab_size - 2)
                padded_tokens = [rand_token] * padding_length + tokens[1:-1]
                attention_mask = [1] * len(tokens) + [0] * padding_length
                
                padded_input_ids.append(padded_tokens)
                padded_token_list[rand_token] = padding_length
                attention_masks.append(attention_mask)
            
            # print(f'batched')
            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(self.device)
            # print(f'input {input_ids_tensor.shape}')
            
            self.logger.debug(f'{input_ids_tensor.shape = }')
            
            attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
            # print(f'attn')
            
            # Generate
            batch_results = self._generate_text_batch(
                input_ids_tensor,
                attention_mask_tensor,
                self.generation_config['max_new_tokens']
            )
            # print(f'results {batch_results}')
            
            batch_time = time.time() - start_time
            
            final_result = []
            for (pd, pl) , batch_text in zip(padded_token_list.items(), batch_results):
                remov = self.tokenizer.decode([pd])
                final_result.append(batch_text[len(remov *pl):])
            all_results.extend(final_result)
            print(f'all {all_results}')
            all_times.extend([batch_time / len(batch_prompts)] * len(batch_prompts))
            
            del input_ids_tensor, attention_mask_tensor
            torch.cuda.empty_cache()
        
        return {
            'prompts': original_prompts if num_return_sequences > 1 else prompts,
            'generated_texts': all_results,
            'avg_generation_time': np.mean(all_times),
            'total_generation_time': sum(all_times),
            'throughput': len(all_results) / sum(all_times)
        }
