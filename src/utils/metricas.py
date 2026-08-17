"""Métricas de avaliação e gráficos.

Acurácia sozinha não diz nada num dataset 25/75: um modelo que responde
"leucemia" para tudo já acerta 75%. As métricas aqui olham a classe positiva
(leucemia) e a negativa separadamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Fora do Jupyter (ex.: `python -m src.treino`) não há janela para desenhar, e o
# backend padrão travaria o script. Dentro do notebook mantemos o backend inline
# para que as figuras apareçam abaixo da célula.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def calcular_metricas(
    y_verdadeiro: np.ndarray,
    y_predito: np.ndarray,
    y_probabilidade: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula o conjunto de métricas usado no TCC.

    `y_probabilidade` é a probabilidade da classe positiva (leucemia); se
    fornecida, entra o AUC-ROC, que mede a qualidade do ranking independente do
    limiar de decisão.
    """
    y_verdadeiro = np.asarray(y_verdadeiro)
    y_predito = np.asarray(y_predito)

    metricas = {
        "acuracia": float(accuracy_score(y_verdadeiro, y_predito)),
        "acuracia_balanceada": float(balanced_accuracy_score(y_verdadeiro, y_predito)),
        "precisao": float(precision_score(y_verdadeiro, y_predito, zero_division=0)),
        # Recall da classe leucemia = sensibilidade. É a métrica que mais importa
        # clinicamente: mede quantos doentes o modelo deixa passar.
        "recall": float(recall_score(y_verdadeiro, y_predito, zero_division=0)),
        "f1": float(f1_score(y_verdadeiro, y_predito, zero_division=0)),
    }

    tn, fp, fn, tp = confusion_matrix(y_verdadeiro, y_predito, labels=[0, 1]).ravel()
    metricas["especificidade"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metricas["verdadeiros_negativos"] = int(tn)
    metricas["falsos_positivos"] = int(fp)
    metricas["falsos_negativos"] = int(fn)
    metricas["verdadeiros_positivos"] = int(tp)

    if y_probabilidade is not None and len(np.unique(y_verdadeiro)) == 2:
        metricas["auc_roc"] = float(roc_auc_score(y_verdadeiro, np.asarray(y_probabilidade)))

    return metricas


def formatar_metricas(metricas: dict[str, float], titulo: str = "Métricas") -> str:
    """Formata as métricas como bloco de texto legível no terminal."""
    rotulos = [
        ("acuracia", "Acurácia"),
        ("acuracia_balanceada", "Acurácia balanceada"),
        ("precisao", "Precisão (leucemia)"),
        ("recall", "Recall / sensibilidade"),
        ("especificidade", "Especificidade"),
        ("f1", "F1-score"),
        ("auc_roc", "AUC-ROC"),
    ]
    linhas = [titulo, "=" * max(len(titulo), 42)]
    linhas += [
        f"{nome:<26}{metricas[chave]:.4f}" for chave, nome in rotulos if chave in metricas
    ]
    if "verdadeiros_positivos" in metricas:
        linhas.append("")
        linhas.append(
            f"VP={metricas['verdadeiros_positivos']:,}  "
            f"VN={metricas['verdadeiros_negativos']:,}  "
            f"FP={metricas['falsos_positivos']:,}  "
            f"FN={metricas['falsos_negativos']:,}"
        )
    return "\n".join(linhas)


# ==========================================
# Gráficos
# ==========================================
def plotar_matriz_confusao(
    y_verdadeiro: np.ndarray,
    y_predito: np.ndarray,
    caminho_saida: Path | str | None = None,
    classes: tuple[str, ...] = config.CLASSES,
    normalizar: bool = False,
):
    """Matriz de confusão com contagens absolutas (ou percentuais por linha)."""
    matriz = confusion_matrix(y_verdadeiro, y_predito, labels=range(len(classes)))
    exibicao = matriz.astype(float)
    if normalizar:
        exibicao = exibicao / exibicao.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    imagem = ax.imshow(exibicao, cmap="Blues")
    fig.colorbar(imagem, ax=ax)

    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predito pelo modelo",
        ylabel="Classe verdadeira",
        title="Matriz de confusão" + (" (normalizada)" if normalizar else ""),
    )

    limite = exibicao.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            texto = f"{exibicao[i, j]:.1%}" if normalizar else f"{int(matriz[i, j]):,}"
            ax.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                color="white" if exibicao[i, j] > limite else "black",
                fontsize=11,
            )

    fig.tight_layout()
    if caminho_saida:
        fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    return fig


def plotar_curvas_treino(historico: dict[str, list[float]], caminho_saida: Path | str | None = None):
    """Curvas de perda e acurácia ao longo das épocas.

    Distância crescente entre a curva de treino e a de validação é o sinal
    visual de overfitting.
    """
    epocas = range(1, len(historico["perda_treino"]) + 1)
    fig, (ax_perda, ax_acuracia) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_perda.plot(epocas, historico["perda_treino"], "o-", label="Treino")
    ax_perda.plot(epocas, historico["perda_validacao"], "s-", label="Validação")
    ax_perda.set(xlabel="Época", ylabel="Perda (loss)", title="Perda por época")
    ax_perda.legend()
    ax_perda.grid(alpha=0.3)

    ax_acuracia.plot(epocas, historico["acuracia_treino"], "o-", label="Treino")
    ax_acuracia.plot(epocas, historico["acuracia_validacao"], "s-", label="Validação")
    if "acuracia_balanceada_validacao" in historico:
        ax_acuracia.plot(
            epocas,
            historico["acuracia_balanceada_validacao"],
            "^--",
            label="Val. balanceada",
        )
    ax_acuracia.set(xlabel="Época", ylabel="Acurácia", title="Acurácia por época")
    ax_acuracia.legend()
    ax_acuracia.grid(alpha=0.3)

    fig.tight_layout()
    if caminho_saida:
        fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    return fig


def plotar_curva_roc(
    y_verdadeiro: np.ndarray,
    y_probabilidade: np.ndarray,
    caminho_saida: Path | str | None = None,
):
    """Curva ROC da classe leucemia."""
    fpr, tpr, _ = roc_curve(y_verdadeiro, y_probabilidade)
    auc = roc_auc_score(y_verdadeiro, y_probabilidade)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ResNet (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC = 0.5)")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Taxa de falsos positivos (1 - especificidade)",
        ylabel="Taxa de verdadeiros positivos (sensibilidade)",
        title="Curva ROC — detecção de leucemia",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if caminho_saida:
        fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    return fig
