import random
from typing import Tuple

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


if __name__ == "__main__":
    source, target_input, target_output, valid_mask = \
        synthetic_reverse_dataset(5, 8, 16)

    assert source.shape[0] == target_input.shape[0] == target_output.shape[0] == valid_mask.shape[0]
    assert source.shape == valid_mask.shape
    assert target_input.shape == target_output.shape
    assert all(target_input[:, 0] == BOS_TOKEN)