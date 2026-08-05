from typing import List, Callable
import re


def char_tokenizer(text: str) -> List[str]:
    """
    문자 기반으로 토크나이징을 수행하는 함수
    
    Args:
        text (str): 토크나이징을 수행하고자 하는 텍스트
    Returns:
        List[str]: 문자 기반으로 쪼개진 토큰들이 담긴 리스트
    """
    return list(text)


def word_tokenizer(text: str) -> List[str]:
    """
    단어 기반으로 토크나이징을 수행하는 함수

    Args:
        text (str): 토크나이징을 수행하고자 하는 텍스트
    Returns:
        List[str]: 단어 기반으로 쪼개진 토큰들이 담긴 리스트
    """
    return re.findall(r"[a-zA-Z']+|[.,!?]|\s+", text)


def get_vocab(corpus: List[str], tokenize_fn: Callable) -> List[str]:
    """
    입력된 전체 corpus에 대해서 정렬된 vocabulary를 생성하는 함수

    Args:
        corpus (List[str]): vocabulary를 구성하고자 하는 텍스트 집합
        tokenize_fn (Callable): 토크나이징을 수행할 수 있는 함수
    Returns:
        List[str]: sorted unique vocabulary
    """
    vocab = []
    for text in corpus:
        tokenized_text = tokenize_fn(text)
        vocab += tokenized_text
    return sorted(set(vocab))