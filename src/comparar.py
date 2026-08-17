"""Comparação entre estratégias de divisão dos dados.

Gera a tabela e a figura que quantificam o efeito do vazamento de dados
(*data leakage*): quanto a divisão aleatória infla cada métrica em relação à
divisão por paciente.

Uso (a partir da raiz do repositório):

    python -m src.comparar
    python -m src.comparar --arquivos outputs/a.json outputs/b.json \\
                           --rotulos "Experimento A" "Experimento B"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from src import config

# Métricas comparadas, na ordem em que aparecem na tabela.
METRICAS = [
    ("acuracia", "Acurácia"),
    ("acuracia_balanceada", "Acurácia balanceada"),
    ("precisao", "Precisão (leucemia)"),
    ("recall", "Recall / sensibilidade"),
    ("especificidade", "Especificidade"),
    ("f1", "F1-score"),
    ("auc_roc", "AUC-ROC"),
]

# Paleta validada com scripts/validate_palette.js (rampa ordinal de 1 matiz,
# 2 tons, modo claro): todos os checks passam.
COR_REFERENCIA = "#86b6ef"  # tom claro — a divisão com vazamento
COR_DESTAQUE = "#1c5cab"  # tom escuro — a divisão correta
COR_CONECTOR = "#c3c2b7"
COR_SUPERFICIE = "#fcfcfb"
COR_TINTA = "#0b0b0b"
COR_TINTA_SEC = "#52514e"
COR_MUTED = "#898781"
COR_GRADE = "#e1e0d9"


def _formatar(valor: float) -> str:
    """Formata no padrão brasileiro (vírgula decimal)."""
    return f"{valor:.4f}".replace(".", ",")


def _formatar_com_sinal(valor: float) -> str:
    """Como `_formatar`, mas com sinal explícito e o menos tipográfico."""
    return f"{'+' if valor >= 0 else '−'}{_formatar(abs(valor))}"


def carregar_metricas(caminho: Path) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de métricas não encontrado: {caminho}\n"
            "Rode o treino correspondente antes, por exemplo:\n"
            "  python -m src.treino --estrategia aleatorio --nome comparacao_leakage"
        )
    return json.loads(caminho.read_text(encoding="utf-8"))


def montar_tabela(dados: list[dict], rotulos: list[str]) -> list[tuple[str, float, float, float]]:
    """Devolve [(nome da métrica, valor A, valor B, diferença B - A)]."""
    linhas = []
    for chave, nome in METRICAS:
        if any(chave not in d for d in dados):
            continue
        a, b = dados[0][chave], dados[1][chave]
        linhas.append((nome, a, b, b - a))
    return linhas


def imprimir_tabela(linhas, rotulos: list[str]) -> str:
    """Monta a tabela em Markdown, pronta para colar no texto do TCC."""
    largura = max(len(n) for n, *_ in linhas)
    larg_dif = len("Diferença")
    cabecalho = f"| {'Métrica':<{largura}} | {rotulos[0]} | {rotulos[1]} | Diferença |"
    separador = (
        f"|{'-' * (largura + 2)}|{'-' * (len(rotulos[0]) + 2)}|"
        f"{'-' * (len(rotulos[1]) + 2)}|{'-' * (larg_dif + 2)}|"
    )

    corpo = [
        f"| {nome:<{largura}} | {_formatar(a):>{len(rotulos[0])}} | "
        f"{_formatar(b):>{len(rotulos[1])}} | {_formatar_com_sinal(d):>{larg_dif}} |"
        for nome, a, b, d in linhas
    ]
    return "\n".join([cabecalho, separador, *corpo])


def plotar_comparacao(linhas, rotulos: list[str], caminho_saida: Path):
    """Gráfico dumbbell: uma linha por métrica, um ponto por estratégia.

    Dumbbell é a forma indicada para "antes → depois por item": o que o leitor
    precisa ler é o **tamanho da lacuna** entre as duas condições. Como pontos
    não codificam magnitude por área, o eixo pode ser recortado na faixa onde os
    dados vivem sem enganar ninguém — o que importa aqui, já que todas as
    métricas ficam espremidas perto de 1,0.
    """
    # Maior diferença no topo: a métrica mais inflada é a que conta a história.
    ordenadas = sorted(linhas, key=lambda linha: abs(linha[3]))
    nomes = [linha[0] for linha in ordenadas]
    valores_a = [linha[1] for linha in ordenadas]
    valores_b = [linha[2] for linha in ordenadas]
    diferencas = [linha[3] for linha in ordenadas]
    posicoes = range(len(ordenadas))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor(COR_SUPERFICIE)
    ax.set_facecolor(COR_SUPERFICIE)

    # Conector: é ele que carrega a mensagem (o tamanho da lacuna).
    for y, (a, b) in enumerate(zip(valores_a, valores_b, strict=True)):
        ax.plot([a, b], [y, y], color=COR_CONECTOR, linewidth=2, zorder=1, solid_capstyle="round")

    # Anel de 2px na cor da superfície evita que os pontos se fundam quando a
    # lacuna é pequena.
    ax.scatter(valores_a, list(posicoes), s=90, color=COR_REFERENCIA, zorder=3,
               edgecolors=COR_SUPERFICIE, linewidths=2, label=rotulos[0])
    ax.scatter(valores_b, list(posicoes), s=90, color=COR_DESTAQUE, zorder=3,
               edgecolors=COR_SUPERFICIE, linewidths=2, label=rotulos[1])

    minimo = min(min(valores_a), min(valores_b))
    maximo = max(max(valores_a), max(valores_b))
    folga = max((maximo - minimo) * 0.22, 0.004)
    # Teto fixo em 1,0: estas métricas são limitadas por definição, e um eixo que
    # passasse de 1 sugeriria uma região que não existe.
    ax.set_xlim(minimo - folga, min(1.0, maximo + folga))

    # Eixo em português (vírgula decimal).
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda valor, _: f"{valor:.2f}".replace(".", ","))
    )

    # Rótulo direto apenas da diferença — um número por linha, que é o dado que o
    # gráfico existe para mostrar. Fica fora da área de plotagem (x em fração dos
    # eixos, y em coordenada de dado), então não distorce a escala. Os valores
    # exatos de cada ponto ficam na tabela.
    fora = blended_transform_factory(ax.transAxes, ax.transData)
    for y, d in enumerate(diferencas):
        ax.text(1.13, y, _formatar_com_sinal(d), transform=fora,
                va="center", ha="right", fontsize=9, color=COR_TINTA_SEC)

    ax.set_yticks(list(posicoes))
    ax.set_yticklabels(nomes, fontsize=10, color=COR_TINTA)
    ax.set_ylim(-0.7, len(ordenadas) - 0.3)
    ax.set_xlabel("Valor da métrica no conjunto de teste", fontsize=10, color=COR_TINTA_SEC)
    ax.tick_params(axis="x", colors=COR_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)

    ax.set_title("Efeito do vazamento de dados sobre as métricas",
                 fontsize=13, color=COR_TINTA, pad=32, loc="left")
    ax.text(0, 1.06, "Nível célula — mesma arquitetura, mesmos hiperparâmetros; muda só a divisão dos dados",
            transform=ax.transAxes, fontsize=9.5, color=COR_TINTA_SEC, va="bottom")
    ax.text(1.13, 1.06, "Diferença", transform=ax.transAxes, fontsize=9,
            color=COR_MUTED, va="bottom", ha="right")

    # Grade hairline sólida e recessiva; sem moldura.
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ax.spines.values():
        lado.set_visible(False)

    legenda = ax.legend(loc="lower right", frameon=False, fontsize=9.5, ncol=2,
                        bbox_to_anchor=(1.0, -0.3))
    for texto in legenda.get_texts():
        texto.set_color(COR_TINTA_SEC)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150, bbox_inches="tight", facecolor=COR_SUPERFICIE)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compara duas execuções de treino.")
    parser.add_argument(
        "--arquivos",
        nargs=2,
        type=Path,
        default=[
            config.DIR_SAIDAS / "comparacao_leakage_metricas_teste.json",
            config.DIR_SAIDAS / "resnet18_paciente_metricas_teste.json",
        ],
        help="dois JSONs de métricas; o segundo é tratado como a referência correta",
    )
    parser.add_argument(
        "--rotulos", nargs=2, default=["Divisão aleatória", "Divisão por paciente"]
    )
    parser.add_argument("--saida", type=Path, default=config.DIR_SAIDAS / "comparacao_divisoes")
    args = parser.parse_args(argv)

    dados = [carregar_metricas(caminho) for caminho in args.arquivos]
    linhas = montar_tabela(dados, args.rotulos)
    tabela = imprimir_tabela(linhas, args.rotulos)

    print("\nCOMPARAÇÃO ENTRE ESTRATÉGIAS DE DIVISÃO — nível célula, conjunto de teste\n")
    print(tabela)

    inflacao = [d for *_, d in linhas]
    print(
        f"\n{args.rotulos[1]} é mais baixa em {sum(1 for d in inflacao if d < 0)} "
        f"de {len(inflacao)} métricas. "
        f"Diferença média: {_formatar_com_sinal(sum(inflacao) / len(inflacao))}"
    )
    print(
        "\nATENÇÃO: com divisão aleatória, a métrica por paciente não faz sentido — os\n"
        "mesmos 189 pacientes aparecem no treino e no teste, então não existe paciente\n"
        "novo para agregar. A comparação honesta é a de nível célula, acima."
    )

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_md = args.saida.with_suffix(".md")
    caminho_png = args.saida.with_suffix(".png")

    caminho_md.write_text(
        "# Comparação entre estratégias de divisão\n\n"
        "Nível célula, conjunto de teste. Mesma arquitetura (ResNet18), mesmos "
        "hiperparâmetros; muda apenas a divisão dos dados.\n\n"
        f"{tabela}\n",
        encoding="utf-8",
    )
    plotar_comparacao(linhas, args.rotulos, caminho_png)

    print(f"\nTabela salva em {caminho_md}")
    print(f"Figura salva em {caminho_png}")


if __name__ == "__main__":
    main()
