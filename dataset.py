import torch
from typing import Any
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # as SOS and EOS token
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # ONLY SOS token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too small")

        # SOS and EOS to encoder input source text
        encoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # SOS token to decoder input
        decoder_input = torch.cat(
            [
            self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype=torch.int64)
            ]
        )

        # Add EOS token to label (label is output we expect from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # encoder_input = encoder_input.to(torch.float32)
        # decoder_input = decoder_input.to(torch.float32)
        # label = label.to(torch.float32)
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,Seq_len) removed the paddings which is needed for self attention mechanism(doesnt need the padding)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,Seq_len) & (1, Seq_len, Seq_len)
            "label": label, # (Seq_len),
            "src_text": src_text,
            "tgt_text": tgt_text,

        }
# class BilingualDataset(Dataset):
#     def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
#         super().__init__()
#         self.seq_len = seq_len
#         self.ds = ds
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
#
#         # Special tokens with consistent dtype
#         self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
#         self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
#         self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
#
#     def __len__(self):
#         return len(self.ds)
#
#     def __getitem__(self, index: Any) -> Any:
#         # Extract the source and target text
#         src_target_pair = self.ds[index]
#         src_text = src_target_pair['translation'][self.src_lang]
#         tgt_text = src_target_pair['translation'][self.tgt_lang]
#
#         # Tokenize source and target text
#         enc_input_tokens = self.tokenizer_src.encode(src_text).ids
#         dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
#
#         # Calculate the number of padding tokens
#         enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # [SOS] and [EOS]
#         dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # [SOS] only
#
#         if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
#             raise ValueError("Sentence is too long for the specified sequence length.")
#
#         # Construct encoder input with [SOS], [EOS], and padding
#         encoder_input = torch.cat([
#             self.sos_token,
#             torch.tensor(enc_input_tokens, dtype=torch.int64),
#             self.eos_token,
#             self.pad_token.repeat(enc_num_padding_tokens)
#         ])
#
#         # Construct decoder input with [SOS] and padding
#         decoder_input = torch.cat([
#             self.sos_token,
#             torch.tensor(dec_input_tokens, dtype=torch.int64),
#             self.pad_token.repeat(dec_num_padding_tokens)
#         ])
#
#         # Construct label with [EOS] and padding
#         label = torch.cat([
#             torch.tensor(dec_input_tokens, dtype=torch.int64),
#             self.eos_token,
#             self.pad_token.repeat(dec_num_padding_tokens)
#         ])
#
#         # Validate tensor sizes
#         assert encoder_input.size(0) == self.seq_len
#         assert decoder_input.size(0) == self.seq_len
#         assert label.size(0) == self.seq_len
#
#         # Construct masks
#         encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
#         decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len)
#
#         return {
#             "encoder_input": encoder_input,
#             "decoder_input": decoder_input,
#             "encoder_mask": encoder_mask,  # Shape: (1, 1, seq_len)
#             "decoder_mask": decoder_mask,  # Shape: (1, seq_len, seq_len)
#             "label": label,                # Shape: (seq_len,)
#             "src_text": src_text,
#             "tgt_text": tgt_text,
#         }
#

def causal_mask(size): # removes values which are above the diagonal of the self attention map
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0













