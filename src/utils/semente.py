"""Fixação de sementes aleatórias para reprodutibilidade."""

import os
import random

import numpy as np
import torch


def definir_semente(semente: int = 42, deterministico: bool = False) -> None:
    """Fixa as sementes de `random`, `numpy` e `torch`.

    Com `deterministico=True` o cuDNN roda em modo determinístico: os resultados
    passam a ser bit-a-bit reproduzíveis, ao custo de perder alguns por cento de
    velocidade nas convoluções.
    """
    os.environ["PYTHONHASHSEED"] = str(semente)
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)

    if deterministico:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True deixa o cuDNN escolher o algoritmo mais rápido para o
        # nosso tamanho fixo de entrada — vale bastante em 1.700+ lotes/época.
        torch.backends.cudnn.benchmark = True
