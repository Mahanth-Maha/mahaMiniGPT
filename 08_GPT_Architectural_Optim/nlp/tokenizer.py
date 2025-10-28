from collections import Counter
from heapq import heapify, heappop, heappush
from tqdm import tqdm
import os
import time

def enc(x, chars):
    return [chars.index(c) for c in x]

def dec(x, chars):
    return "".join(chars[i] for i in x)
        
class CharTokenizer:
    def __init__(self, chars_file, special_tokens=None):
        self.special_tokens = special_tokens if special_tokens else {}
        self.vocab = None
        self.chars_file = chars_file 
        if os.path.exists(self.chars_file):
            self.chars = self.load()
        self.vocab_size = self._vocab_size()

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
    
    def encode(self,text):
        return [self.chars.index(c) for c in text]

    def _vocab(self):
        vocab = {idx: charr for idx,charr in enumerate(self.chars)}
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def _vocab_size(self):
        if self.vocab is None:
            return None
        return len(self.vocab)
    
    def save(self):
        open(self.chars_file, "w").write(self.chars)

    def train(self, data_file):
        data = open(data_file, "r").read()
        self.chars = "".join(sorted(set(data)))
        self.save()
        return self.chars
    
    def load(self):
        self.chars = open(self.chars_file, "r").read()
        self.vocab = self._vocab()
        return self.chars
        

class myTokenizer_faster:
    def __init__(self, token_model_file, special_tokens=None):
        self.merges = {}
        self.token_model_file = token_model_file 
        self.special_tokens = special_tokens if special_tokens else {}
        self.vocab = self._vocab()
        
    def _get_bigram_byte_frequency(self, ids):
        return Counter(zip(ids[:-1], ids[1:]))
    
    def train(self, data_file, vocab_size, to_train = None ):
        spl_tkns = len(self.special_tokens)
        num_merges = vocab_size - 256 - spl_tkns
        if to_train:    
            text_size_to_consider = to_train
        
        text_size_to_consider = int(vocab_size ** 1.75)
        
        text = ""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                text = f.read(text_size_to_consider)
                if len(text) == text_size_to_consider:
                    extra_chars = ""
                    while True:
                        char = f.read(1)
                        if not char or char.isspace():
                            break
                        extra_chars += char
                        if len(extra_chars) > 100:
                            break
                    text += extra_chars
                    
        except UnicodeDecodeError as e:
            print(f"Unicode error: {e}")
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(text_size_to_consider)
        
        print(f"🔸 Actually read: {len(text):,} characters")
        
        ids = list(text.encode("utf-8"))
        print(f"🔸 Converted to: {len(ids):,} bytes")
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        print(f"🔸 Starting BPE training...")
        counts = self._get_bigram_byte_frequency(ids)
        heap = [(-freq, source) for source, freq in counts.items()]
        heapify(heap)

        for i in tqdm(range(num_merges), desc="Training", unit="merge"):
            while heap:
                freq, source = heappop(heap)
                if counts[source] == -freq and counts[source] > 0:
                    break
            else:
                break

            dest = 256 + i
            self.merges[source] = dest
            vocab[dest] = vocab[source[0]] + vocab[source[1]]

            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == source[0] and ids[j + 1] == source[1]:
                    new_ids.append(dest)

                    if len(new_ids) >= 2:
                        left = (new_ids[-2], dest)
                        counts[left] += 1
                        heappush(heap, (-counts[left], left))

                    if j + 2 < len(ids):
                        right = (dest, ids[j + 2])
                        counts[right] += 1
                        heappush(heap, (-counts[right], right))

                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
            counts[source] = 0
        print(f"🔸 BPE training Finished")

        self.vocab = vocab
        
        self.save()
        self._analyze_vocab()

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
    
    def encode(self, text, progress=False, legacy=False):
        if not legacy:
            return self.encode_fast(text, progress=progress)
        
        ids = list(text.encode("utf-8"))
        initial_length = len(ids)
        
        if progress:
            pbar = tqdm(desc="Encoding", unit="merges", leave=True)
            pbar.set_postfix({
                'tokens': len(ids), 
                'original': initial_length,
                'ratio': '1.000'
            })
        
        merge_count = 0
        
        while True:
            candidates = [(self.merges[bg], bg) for bg in zip(ids[:-1], ids[1:]) if bg in self.merges]
            if not candidates:
                break
            
            _, pair = min(candidates)
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
                    new_ids.append(self.merges[pair])
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
            merge_count += 1
            
            if progress:
                pbar.update(1)
                pbar.set_postfix({
                    'tokens': len(ids),
                    'original': initial_length, 
                    'ratio': f"{len(ids)/initial_length:.3f}",
                    'merges': merge_count
                })
        
        if progress:
            pbar.close()
            print(f"✅ Encoding complete: {initial_length} -> {len(ids)} tokens ({merge_count} merges)")
        
        return ids

    def encode_fast(self, text, progress=False):
        ids = list(text.encode("utf-8"))
        if len(ids) <= 1:
            return ids

        initial_length = len(ids)

        if progress:
            pbar = tqdm(total=len(ids), desc="Encoding", unit="merge", dynamic_ncols=True)
            start_time = time.time()

        i = 0
        out = []
        merges_applied = 0
        while i < len(ids):
            j = i
            token = ids[j]

            while j + 1 < len(ids) and (token, ids[j + 1]) in self.merges:
                token = self.merges[(token, ids[j + 1])]
                j += 1
                merges_applied += 1

                if progress and merges_applied % 100 == 0:
                    compression = (len(out) + (len(ids) - j)) / initial_length
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (merges_applied + 1)
                    eta = avg_time * (len(ids) - j)
                    pbar.set_postfix({
                        "tokens": f"{len(out):,}",
                        "ratio": f"{compression:.3f}x",
                        "ETA": f"{eta:.1f}s"
                    })
                    pbar.update(100)

            out.append(token)
            i = j + 1

        if progress:
            pbar.close()
            final_compression = len(out) / initial_length
            print(f"✅ Encoded: {initial_length:,} → {len(out):,} tokens "
                f"({final_compression:.3f}x compression)")

        return out

    def _vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self):
        model_file = self.token_model_file
        vocab_file = self.token_model_file.replace('.model', '_vocab.txt')
        
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(self.special_tokens)}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
        
        vocab = self._vocab()
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("# Tokenizer Vocabulary\n")
            f.write(f"# Total vocabulary size: {len(vocab)}\n")
            f.write(f"# Number of merges: {len(self.merges)}\n")
            f.write(f"# Number of special tokens: {len(self.special_tokens)}\n")
            f.write("#" + "="*60 + "\n\n")
            
            for token_id in sorted(vocab.keys()):
                token_bytes = vocab[token_id]
                try:
                    if isinstance(token_bytes, bytes):
                        token_str = token_bytes.decode('utf-8', errors='replace')
                    else:
                        token_str = str(token_bytes)
                    
                    display_str = repr(token_str) if any(ord(c) < 32 or ord(c) > 126 for c in token_str) else token_str
                    f.write(f"{token_id:6d}: {display_str}\n")
                except:
                    f.write(f"{token_id:6d}: {repr(token_bytes)}\n")
        
        print(f"Model saved to: {model_file}")
        print(f"Readable vocab saved to: {vocab_file}")

    def load(self):
        merges = {}
        special_tokens = {}
        idx = 256
        model_file = self.token_model_file
        with open(model_file, 'r', encoding="utf-8") as f:
            
            lines = [line.strip() for line in f if line.strip()]
            
            merge_lines = []
            special_lines = []
            
            for line in lines:
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    merge_lines.append(line)
                else:
                    special_lines.append(line)
                    
            for line in merge_lines:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
            
            for line in special_lines:
                if line:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        special, special_idx = parts
                        special_tokens[special] = int(special_idx)
        
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._vocab()
        
        print(f"Tokenizer loaded from: {model_file}")
        print(f"Loaded {len(merges)} merges and {len(special_tokens)} special tokens")
        print(f"Total vocabulary size: {len(self.vocab)}")

        return self.vocab

    def _vocab_size(self):
        if self.vocab is None:
            return None
        return len(self.vocab)
    
    def _analyze_vocab(self):
        vocab = self._vocab()
        
        print("\n" + "="*60)
        print("VOCABULARY ANALYSIS")
        print("="*60)
        print(f"Total vocabulary size: {len(vocab):,}")
        print(f"Base tokens (0-255): 256")
        print(f"Merge tokens: {len(self.merges):,}")
        print(f"Special tokens: {len(self.special_tokens):,}")
        
        base_tokens = sum(1 for tid in vocab.keys() if tid < 256)
        merge_tokens = sum(1 for tid in vocab.keys() if 256 <= tid < 256 + len(self.merges))
        special_token_ids = sum(1 for tid in vocab.keys() if tid >= 256 + len(self.merges))
        
        print(f"\nToken distribution:")
        print(f"🔸 Base (bytes): {base_tokens}")
        print(f"🔸 Merges: {merge_tokens}")
        print(f"🔸 Special: {special_token_ids}")
        
        print(f"\nFirst 10 merge tokens:")
        merge_start = 256
        for i in range(min(10, len(self.merges))):
            token_id = merge_start + i
            if token_id in vocab:
                token_bytes = vocab[token_id]
                try:
                    token_str = token_bytes.decode('utf-8', errors='replace')
                    display_str = repr(token_str) if any(ord(c) < 32 or ord(c) > 126 for c in token_str) else token_str
                    print(f"🔸 {token_id}: {display_str}")
                except:
                    print(f"🔸 {token_id}: {repr(token_bytes)}")
        
        print(f"\nSpecial tokens:")
        for special, idx in sorted(self.special_tokens.items(), key=lambda x: x[1]):
            print(f"🔸 {special}: {idx}")