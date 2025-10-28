# make_shards.py
import argparse
from train_utils import create_shards_from_hf_txt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)  # HuggingFace txt file input
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='ck100base')
    parser.add_argument('--min-length', type=int, default=32)
    parser.add_argument('--shard-size', type=int, default=10**6)
    args = parser.parse_args()
    create_shards_from_hf_txt(
        hf_txt_path=args.input,
        shard_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
        shard_size=args.shard_size,
        min_length=args.min_length
    )
if __name__ == '__main__':
    main()
