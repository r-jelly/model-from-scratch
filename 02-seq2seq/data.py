import re
import random
from typing import Tuple, List, Dict, Iterable
from collections import Counter

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

def synthetic_reverse_dataset(
    batch_size: int,
    min_seq: int,
    max_seq: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    임의의 숫자 토큰 시퀀스를 생성하고, 이를 뒤집는 것을 목표로 하는 데이터셋 제작

    Args:
        batch_size (int): 배치 크기
        min_seq (int): 생성할 토큰 시퀀스의 최소 길이
        max_seq (int): 생성할 토큰 시퀀스의 최대 길이

    Retruns:
        source (Tensor): (batch_size, seq_len) 크기의 원본 숫자열
        target_input (Tensor): (batch_size, seq_len+1) 사이즈, BOS 토큰 + 뒤집힌 숫자열
        target_output (Tensor): (batch_size, seq_len+1) 사이즈, 뒤집힌 숫자열 + EOS 토큰
        valid_mask (Tensor): (batch_size, seq_len) 사이즈로 실제 source 위치를 표시
    """
    # 0. random number sequence 구성
    # Token IDs = PAD: 0, BOS: 1, EOS: 2, 숫자 0~9: 3~12
    num_seq_list = []
    for _ in range(batch_size):
        seq_len = random.randint(min_seq, max_seq)
        cur_source = torch.randint(low=3, high=13, size=(seq_len,))
        num_seq_list.append(cur_source)

    # 1. source 구성하기
    source = pad_sequence(num_seq_list, batch_first=True, padding_value=PAD_TOKEN) # 서로 다른 길이의 Tensor 시퀀스를 하나로

    # 2. valid_mask 제작
    valid_mask = (source != PAD_TOKEN)

    # 3. target 제작
    target_input, target_output = [], []
    for seq in num_seq_list:
        reversed_seq = torch.flip(seq, dims=(0,))
        target_input.append(torch.cat((torch.LongTensor([BOS_TOKEN]), reversed_seq)))
        target_output.append(torch.cat((reversed_seq, torch.LongTensor([EOS_TOKEN]))))

    target_input = pad_sequence(target_input, batch_first=True, padding_value=PAD_TOKEN)
    target_output = pad_sequence(target_output, batch_first=True, padding_value=PAD_TOKEN)

    return source, target_input, target_output, valid_mask


def word_tokenizer(text: str) -> List[str]:
    """
    문자열 하나를 입력받아, 단어 토큰 단위로 분해하는 함수
    다음과 같은 규칙으로 Tokenize를 진행한다:
        1) 전원 소문자화
        2) 공백 제거
        3) 문장부호는 별도의 토큰
        4) 단어 내부의 아포스트로피 및 다른 언어 Unicode 보존

    Args:
        text (str): 토크나이징을 진행할 전체 문자열
    Returns:
        tokens (List[str]): 토큰들의 리스트
    """
    # 대문자를 소문자화
    text = text.lower()

    # 정규표현식으로 단어, 문장부호 찾기
    normalized = r"[^\W_]+(?:['’][^\W_]+)*|[.,!?]"
    tokens = re.findall(normalized, text)

    return tokens


def build_vocab(corpus: Iterable[str], min_freq: int) -> List[str]:
    """
    문자열들의 Iterable 객체와 최소 빈도수를 입력받아서, 최종 Vocabulary를 생성

    Args:
        corpus (Iterable[str]): 문장들의 Iterable 객체
        min_freq (int): 토큰으로 인정하는 최소 빈도수
    Returns:
        vocab (List[str]): 조건을 만족하는 Vocab
    """
    token_freq = Counter()

    # 각 문장을 word_tokenizer로 분리
    for sentence in corpus:
        tokens = word_tokenizer(sentence)
        token_freq.update(tokens)

    # min_freq 이상인 토큰만 수집
    allow_tokens = [k for k, v in token_freq.items() if v >= min_freq]
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']

    # 최종 vocab 생성
    allow_tokens = sorted(
        allow_tokens,
        key=lambda token: (-token_freq[token], token)
    )
    vocab = special_tokens + allow_tokens
    return vocab


def encode_text(text: str, token_to_id: Dict[str, int]) -> List[int]:
    """
    문자열을 tokenizer 규칙에 맞게 token id들의 리스트로 반환하는 함수

    Args:
        text (str): 변환할 문자열
        token_to_id (Dict[str, int]): 토큰에 대한 ID 사전
    Returns:
        token_ids (List[int]): 변환된 token id 리스트
    """
    tokens = word_tokenizer(text)
    token_ids = [
        token_to_id.get(token, token_to_id['<unk>'])
        for token in tokens
    ]
    return token_ids


def translate_collate_fn(
    batch: List[Dict[str, str]],
    source_token_to_id: Dict[str, int],
    target_token_to_id: Dict[str, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Dataset의 여러 샘플을 1개의 Batch Tensor로 조립하는 함수
    - 각 문장을 Token ID로 변환 후, <bos>/<eos> 추가
    - 길이가 다른 문장들을 최장 길이에 맞춰 Padding
    - source의 실제 토큰 위치를 나타내는 masking Tensor 생성

    Args:
        batch (List): DE/EN의 문자열 Dictionary를 포함하는 리스트
        source_token_to_id (Dict): source 언어의 token_to_id 사전
        target_token_to_id (Dict): target 언어의 token_to_id 사전
    Returns:
        source (LongTensor): source 언어의 token id 리스트 + EOS
        target_input (LongTensor): BOS + target 언어의 token id 리스트
        target_output (LongTensor): target 언어의 token id 리스트 + EOS
        valid_mask (BoolTensor): source의 실제 토큰 위치를 나타내는 텐서
    """
    # 번역 Task에서는 source 문장 끝에 EOS를 붙이는 것이 관례라고 함
    # https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    source_seq = []
    target_input_seq = []
    target_output_seq = []

    for sample in batch:
        source_ids = encode_text(sample['de'], source_token_to_id)
        target_ids = encode_text(sample['en'], target_token_to_id)

        source_seq.append(Tensor(source_ids+[EOS_TOKEN]).type(torch.long))
        target_input_seq.append(Tensor([BOS_TOKEN]+target_ids).type(torch.long))
        target_output_seq.append(Tensor(target_ids+[EOS_TOKEN]).type(torch.long))

    source = pad_sequence(source_seq, batch_first=True, padding_value=PAD_TOKEN)
    target_input = pad_sequence(target_input_seq, batch_first=True, padding_value=PAD_TOKEN)
    target_output = pad_sequence(target_output_seq, batch_first=True, padding_value=PAD_TOKEN)
    valid_mask = source != PAD_TOKEN

    return source, target_input, target_output, valid_mask


if __name__ == "__main__":
    source, target_input, target_output, valid_mask = \
        synthetic_reverse_dataset(5, 8, 16)

    assert source.shape[0] == target_input.shape[0] == target_output.shape[0] == valid_mask.shape[0]
    assert source.shape == valid_mask.shape
    assert target_input.shape == target_output.shape
    assert all(target_input[:, 0] == BOS_TOKEN)

    assert word_tokenizer("A man, runs!") == ["a", "man", ",", "runs", "!"]
    assert word_tokenizer("Don't stop.") == ["don't", "stop", "."]
    assert word_tokenizer("Ein Mädchen läuft.") == ["ein", "mädchen", "läuft", "."]

    corpus = (
        sentence
        for sentence in [
            "A man runs.",
            "A woman runs.",
            "A dog sleeps.",
        ]
    )
    vocab = build_vocab(corpus, min_freq=2)
    assert vocab == ["<pad>", "<bos>", "<eos>", "<unk>", ".", "a", "runs"]

    token_to_id = {token: token_id for token_id, token in enumerate(vocab)}
    assert encode_text("A cat runs.", token_to_id) == [5, 3, 6, 4]
    assert encode_text("", token_to_id) == []

    source_token_to_id = {
        '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
        'ein': 4, 'mann': 5, 'läuft': 6, '.': 7,
    }
    target_token_to_id = {
        '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
        'a': 4, 'man': 5, 'runs': 6, '.': 7,
    }
    translation_batch = [
        {'de': 'Ein Mann.', 'en': 'A man.'},
        {'de': 'Ein Mann läuft.', 'en': 'A man runs.'},
    ]
    source, target_input, target_output, valid_mask = translate_collate_fn(
        translation_batch,
        source_token_to_id,
        target_token_to_id,
    )

    assert source.dtype == target_input.dtype == target_output.dtype == torch.long
    assert valid_mask.dtype == torch.bool
    assert source.tolist() == [[4, 5, 7, 2, 0], [4, 5, 6, 7, 2]]
    assert target_input.tolist() == [[1, 4, 5, 7, 0], [1, 4, 5, 6, 7]]
    assert target_output.tolist() == [[4, 5, 7, 2, 0], [4, 5, 6, 7, 2]]
    assert valid_mask.tolist() == [[True, True, True, True, False], [True] * 5]
