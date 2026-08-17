"""Avaliação de um checkpoint treinado no conjunto de teste.

Uso (a partir da raiz do repositório):

    python -m src.avaliar --checkpoint outputs/resnet18_paciente_melhor.pth

Gera métricas, matriz de confusão, curva ROC e — o que mais interessa
clinicamente — o diagnóstico agregado por paciente.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src import config
from src.dados import criar_dataloaders
from src.modelos import criar_modelo
from src.treino import avaliar, obter_dispositivo
from src.utils import (
    calcular_metricas,
    definir_semente,
    formatar_metricas,
    plotar_curva_roc,
    plotar_matriz_confusao,
)


def agregar_por_paciente(
    amostras,
    y_verdadeiro: np.ndarray,
    y_probabilidade: np.ndarray,
    limiar: float = 0.5,
) -> tuple[dict[str, dict], dict[str, float]]:
    """Converte predições por célula em um diagnóstico por paciente.

    Na prática clínica não se classifica uma célula isolada: o hematologista
    olha um esfregaço inteiro. Aqui a decisão do paciente é a **probabilidade
    média** de leucemia entre todas as suas células.

    Devolve (detalhe por paciente, métricas no nível do paciente).
    """
    por_paciente: dict[str, dict] = defaultdict(lambda: {"probabilidades": [], "rotulo": None})

    for amostra, verdadeiro, prob in zip(amostras, y_verdadeiro, y_probabilidade, strict=True):
        registro = por_paciente[amostra.paciente]
        registro["probabilidades"].append(float(prob))
        registro["rotulo"] = int(verdadeiro)

    detalhe = {}
    for paciente, registro in sorted(por_paciente.items()):
        media = float(np.mean(registro["probabilidades"]))
        detalhe[paciente] = {
            "rotulo_verdadeiro": registro["rotulo"],
            "classe_verdadeira": config.CLASSES[registro["rotulo"]],
            "probabilidade_media": media,
            "predito": int(media >= limiar),
            "classe_predita": config.CLASSES[int(media >= limiar)],
            "num_celulas": len(registro["probabilidades"]),
            "fracao_celulas_leucemia": float(
                np.mean(np.array(registro["probabilidades"]) >= limiar)
            ),
        }

    verdadeiros = np.array([d["rotulo_verdadeiro"] for d in detalhe.values()])
    preditos = np.array([d["predito"] for d in detalhe.values()])
    probabilidades = np.array([d["probabilidade_media"] for d in detalhe.values()])
    metricas = calcular_metricas(verdadeiros, preditos, probabilidades)

    return detalhe, metricas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Avalia um checkpoint no conjunto de teste.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lote", type=int, default=config.TAMANHO_LOTE)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--limiar", type=float, default=0.5)
    parser.add_argument(
        "--conjunto", default="teste", choices=["treino", "validacao", "teste"]
    )
    parser.add_argument("--sem-amp", action="store_true")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")

    dispositivo = obter_dispositivo()
    checkpoint = torch.load(args.checkpoint, map_location=dispositivo, weights_only=False)

    # Reconstrói exatamente a mesma divisão usada no treino — se estes valores
    # divergirem, o "teste" pode conter imagens que o modelo já viu.
    arquitetura = checkpoint.get("arquitetura", "resnet18")
    estrategia = checkpoint.get("estrategia_divisao", "paciente")
    tamanho_imagem = checkpoint.get("tamanho_imagem", config.TAMANHO_IMAGEM)
    semente = checkpoint.get("semente", config.SEED)

    print(
        f"Checkpoint: {args.checkpoint.name} | época {checkpoint.get('epoca', '?')} | "
        f"{arquitetura} | divisão '{estrategia}' | imagem {tamanho_imagem}px | semente {semente}"
    )

    definir_semente(semente)
    loaders, divisoes = criar_dataloaders(
        tamanho_lote=args.lote,
        tamanho_imagem=tamanho_imagem,
        semente=semente,
        num_workers=args.workers,
        estrategia=estrategia,
        aumento_dados=False,
        # Sem drop_last: a agregação por paciente casa as predições 1-a-1 com
        # `divisoes[conjunto]`, então nenhum lote pode ser descartado.
        descartar_ultimo=False,
    )

    modelo = criar_modelo(arquitetura=arquitetura, num_classes=len(config.CLASSES))
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo = modelo.to(dispositivo)

    _, y_verdadeiro, y_predito, y_prob = avaliar(
        modelo,
        loaders[args.conjunto],
        None,
        dispositivo,
        usar_amp=not args.sem_amp,
        descricao=f"Avaliando ({args.conjunto})",
    )

    # --- Nível da célula ---
    metricas_celula = calcular_metricas(y_verdadeiro, y_predito, y_prob)
    print("\n" + formatar_metricas(metricas_celula, f"Nível CÉLULA — conjunto de {args.conjunto}"))

    # --- Nível do paciente ---
    detalhe_pacientes, metricas_paciente = agregar_por_paciente(
        divisoes[args.conjunto], y_verdadeiro, y_prob, limiar=args.limiar
    )
    print(
        "\n"
        + formatar_metricas(
            metricas_paciente,
            f"Nível PACIENTE — {len(detalhe_pacientes)} pacientes no conjunto de {args.conjunto}",
        )
    )

    errados = [p for p, d in detalhe_pacientes.items() if d["predito"] != d["rotulo_verdadeiro"]]
    if errados:
        print(f"\nPacientes classificados errado ({len(errados)}):")
        for paciente in errados:
            d = detalhe_pacientes[paciente]
            print(
                f"  {paciente}: verdadeiro={d['classe_verdadeira']:<10} "
                f"predito={d['classe_predita']:<10} "
                f"prob. média={d['probabilidade_media']:.3f} ({d['num_celulas']} células)"
            )
    else:
        print("\nTodos os pacientes do conjunto foram classificados corretamente.")

    # --- Artefatos ---
    dir_saidas = Path(config.DIR_SAIDAS)
    dir_saidas.mkdir(parents=True, exist_ok=True)
    prefixo = args.checkpoint.stem.replace("_melhor", "")

    plotar_matriz_confusao(
        y_verdadeiro, y_predito, dir_saidas / f"{prefixo}_matriz_confusao_{args.conjunto}.png"
    )
    plotar_matriz_confusao(
        y_verdadeiro,
        y_predito,
        dir_saidas / f"{prefixo}_matriz_confusao_{args.conjunto}_normalizada.png",
        normalizar=True,
    )
    plotar_curva_roc(y_verdadeiro, y_prob, dir_saidas / f"{prefixo}_curva_roc_{args.conjunto}.png")

    (dir_saidas / f"{prefixo}_relatorio_{args.conjunto}.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "conjunto": args.conjunto,
                "estrategia_divisao": estrategia,
                "limiar": args.limiar,
                "metricas_celula": metricas_celula,
                "metricas_paciente": metricas_paciente,
                "pacientes": detalhe_pacientes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nRelatório e gráficos salvos em {dir_saidas}")


if __name__ == "__main__":
    main()
