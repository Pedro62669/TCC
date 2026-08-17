"""Construção do modelo por transfer learning."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

# Arquiteturas disponíveis: construtor + pesos pré-treinados na ImageNet.
ARQUITETURAS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def criar_modelo(
    arquitetura: str = "resnet18",
    num_classes: int = 2,
    pretreinado: bool = True,
    congelar_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Cria uma ResNet com a camada final adaptada para `num_classes`.

    A ideia do transfer learning: as camadas convolucionais já sabem extrair
    bordas, texturas e formas a partir do ImageNet. Trocamos só a camada
    totalmente conectada final, que no ImageNet decide entre 1000 categorias, por
    uma que decide entre saudável e leucemia.

    `congelar_backbone=True` treina *apenas* essa camada nova (mais rápido, menos
    propenso a overfitting em datasets pequenos). Com 81 mil imagens vale mais
    ajustar a rede inteira, então o padrão é `False`.
    """
    if arquitetura not in ARQUITETURAS:
        raise ValueError(
            f"Arquitetura {arquitetura!r} desconhecida. Opções: {sorted(ARQUITETURAS)}"
        )

    construtor, pesos = ARQUITETURAS[arquitetura]
    modelo = construtor(weights=pesos if pretreinado else None)

    if congelar_backbone:
        for parametro in modelo.parameters():
            parametro.requires_grad = False

    # Substitui o classificador. Os parâmetros novos já nascem com
    # requires_grad=True, então continuam treináveis mesmo com o backbone travado.
    num_features = modelo.fc.in_features
    modelo.fc = (
        nn.Sequential(nn.Dropout(dropout), nn.Linear(num_features, num_classes))
        if dropout > 0
        else nn.Linear(num_features, num_classes)
    )
    return modelo


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    """Devolve (total de parâmetros, parâmetros treináveis)."""
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis
