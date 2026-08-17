"""Treinamento da ResNet por transfer learning.

Uso como script (a partir da raiz do repositório):

    python -m src.treino                          # padrões do config.py
    python -m src.treino --epocas 20 --lote 64
    python -m src.treino --estrategia aleatorio --nome comparacao_leakage
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dados import calcular_pesos_classes, criar_dataloaders, resumir_divisao
from src.modelos import contar_parametros, criar_modelo
from src.utils import (
    calcular_metricas,
    definir_semente,
    formatar_metricas,
    plotar_curvas_treino,
    plotar_matriz_confusao,
)


def obter_dispositivo() -> torch.device:
    """Escolhe GPU se houver, senão CPU, e informa o que foi escolhido."""
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detectada: {torch.cuda.get_device_name(0)} ({vram:.2f} GB de VRAM)")
    else:
        dispositivo = torch.device("cpu")
        print("ATENÇÃO: nenhuma GPU detectada — o treino na CPU será muito lento.")
    return dispositivo


# ==========================================
# Uma época
# ==========================================
def treinar_uma_epoca(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
    escalador: torch.amp.GradScaler | None = None,
    descricao: str = "Treino",
) -> tuple[float, float]:
    """Roda uma época de treino. Devolve (perda média, acurácia)."""
    modelo.train()
    perda_total, acertos, total = 0.0, 0, 0

    barra = tqdm(loader, desc=descricao, leave=False)
    for imagens, rotulos in barra:
        imagens = imagens.to(dispositivo, non_blocking=True)
        rotulos = rotulos.to(dispositivo, non_blocking=True)

        otimizador.zero_grad(set_to_none=True)

        # Precisão mista: parte das operações roda em float16, o que praticamente
        # dobra a velocidade em GPUs com tensor cores e reduz o uso de VRAM.
        with torch.amp.autocast("cuda", enabled=escalador is not None):
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)

        if escalador is not None:
            escalador.scale(perda).backward()
            escalador.step(otimizador)
            escalador.update()
        else:
            perda.backward()
            otimizador.step()

        perda_total += perda.item() * rotulos.size(0)
        acertos += (saidas.argmax(1) == rotulos).sum().item()
        total += rotulos.size(0)
        barra.set_postfix(perda=f"{perda_total / total:.4f}", acc=f"{acertos / total:.4f}")

    return perda_total / total, acertos / total


@torch.no_grad()
def avaliar(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module | None,
    dispositivo: torch.device,
    usar_amp: bool = True,
    descricao: str = "Avaliação",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Roda o modelo sem atualizar pesos.

    Devolve (perda média, rótulos verdadeiros, preditos, prob. de leucemia).
    """
    modelo.eval()
    perda_total, total = 0.0, 0
    verdadeiros, preditos, probabilidades = [], [], []

    for imagens, rotulos in tqdm(loader, desc=descricao, leave=False):
        imagens = imagens.to(dispositivo, non_blocking=True)
        rotulos = rotulos.to(dispositivo, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=usar_amp and dispositivo.type == "cuda"):
            saidas = modelo(imagens)
            if criterio is not None:
                perda_total += criterio(saidas, rotulos).item() * rotulos.size(0)

        # float32 antes do softmax: sob autocast as saídas vêm em float16 e o
        # softmax perderia precisão nas probabilidades.
        prob = torch.softmax(saidas.float(), dim=1)[:, config.CLASSE_POSITIVA]

        verdadeiros.append(rotulos.cpu().numpy())
        preditos.append(saidas.argmax(1).cpu().numpy())
        probabilidades.append(prob.cpu().numpy())
        total += rotulos.size(0)

    return (
        perda_total / total if criterio is not None else float("nan"),
        np.concatenate(verdadeiros),
        np.concatenate(preditos),
        np.concatenate(probabilidades),
    )


# ==========================================
# Loop completo
# ==========================================
def treinar(
    modelo: nn.Module,
    loaders: dict[str, DataLoader],
    dispositivo: torch.device,
    epocas: int = config.EPOCAS,
    taxa_aprendizado: float = config.TAXA_APRENDIZADO,
    decaimento_peso: float = config.DECAIMENTO_PESO,
    pesos_classes: torch.Tensor | None = None,
    paciencia: int = config.PACIENCIA,
    usar_amp: bool = True,
    dir_saidas: Path | str = config.DIR_SAIDAS,
    nome: str = "resnet18",
    metadados: dict | None = None,
) -> tuple[dict[str, list[float]], Path]:
    """Treina o modelo e salva o melhor checkpoint.

    O "melhor" é escolhido pela **acurácia balanceada** na validação, não pela
    acurácia simples: com 25/75 de desbalanceamento, a acurácia simples premiaria
    um modelo que ignora a classe minoritária.

    Devolve (histórico, caminho do melhor checkpoint).
    """
    dir_saidas = Path(dir_saidas)
    dir_saidas.mkdir(parents=True, exist_ok=True)
    caminho_checkpoint = dir_saidas / f"{nome}_melhor.pth"

    modelo = modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss(
        weight=pesos_classes.to(dispositivo) if pesos_classes is not None else None
    )
    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=taxa_aprendizado, weight_decay=decaimento_peso
    )
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=2
    )
    escalador = (
        torch.amp.GradScaler("cuda") if usar_amp and dispositivo.type == "cuda" else None
    )

    historico: dict[str, list[float]] = {
        "perda_treino": [],
        "acuracia_treino": [],
        "perda_validacao": [],
        "acuracia_validacao": [],
        "acuracia_balanceada_validacao": [],
        "f1_validacao": [],
        "auc_validacao": [],
        "taxa_aprendizado": [],
    }

    melhor_metrica = -1.0
    melhor_epoca = 0
    epocas_sem_melhora = 0
    inicio = time.time()

    print(f"\nIniciando treinamento — {epocas} épocas, {len(loaders['treino'])} lotes por época.")
    print(f"Precisão mista (AMP): {'ativada' if escalador else 'desativada'}\n")

    for epoca in range(1, epocas + 1):
        perda_treino, acuracia_treino = treinar_uma_epoca(
            modelo,
            loaders["treino"],
            criterio,
            otimizador,
            dispositivo,
            escalador,
            descricao=f"Época {epoca}/{epocas} [treino]",
        )

        perda_val, y_verdadeiro, y_predito, y_prob = avaliar(
            modelo,
            loaders["validacao"],
            criterio,
            dispositivo,
            usar_amp=usar_amp,
            descricao=f"Época {epoca}/{epocas} [validação]",
        )
        metricas_val = calcular_metricas(y_verdadeiro, y_predito, y_prob)

        historico["perda_treino"].append(perda_treino)
        historico["acuracia_treino"].append(acuracia_treino)
        historico["perda_validacao"].append(perda_val)
        historico["acuracia_validacao"].append(metricas_val["acuracia"])
        historico["acuracia_balanceada_validacao"].append(metricas_val["acuracia_balanceada"])
        historico["f1_validacao"].append(metricas_val["f1"])
        historico["auc_validacao"].append(metricas_val.get("auc_roc", float("nan")))
        historico["taxa_aprendizado"].append(otimizador.param_groups[0]["lr"])

        metrica_alvo = metricas_val["acuracia_balanceada"]
        agendador.step(metrica_alvo)

        print(
            f"Época {epoca:>2}/{epocas} | "
            f"treino: perda {perda_treino:.4f}, acc {acuracia_treino:.4f} | "
            f"val: perda {perda_val:.4f}, acc {metricas_val['acuracia']:.4f}, "
            f"acc_bal {metrica_alvo:.4f}, F1 {metricas_val['f1']:.4f}, "
            f"AUC {metricas_val.get('auc_roc', float('nan')):.4f}"
        )

        if metrica_alvo > melhor_metrica:
            melhor_metrica, melhor_epoca, epocas_sem_melhora = metrica_alvo, epoca, 0
            torch.save(
                {
                    "epoca": epoca,
                    "estado_modelo": modelo.state_dict(),
                    "metricas_validacao": metricas_val,
                    "classes": list(config.CLASSES),
                    **(metadados or {}),
                },
                caminho_checkpoint,
            )
            print(f"   -> melhor até agora; checkpoint salvo em {caminho_checkpoint.name}")
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                print(
                    f"\nEarly stopping: {paciencia} épocas sem melhora "
                    f"(melhor foi a época {melhor_epoca})."
                )
                break

    duracao = time.time() - inicio
    print(
        f"\nTreino concluído em {duracao / 60:.1f} min. "
        f"Melhor acurácia balanceada de validação: {melhor_metrica:.4f} (época {melhor_epoca})."
    )

    dir_saidas.mkdir(parents=True, exist_ok=True)
    (dir_saidas / f"{nome}_historico.json").write_text(
        json.dumps(
            {
                "historico": historico,
                "melhor_epoca": melhor_epoca,
                "melhor_acuracia_balanceada": melhor_metrica,
                "duracao_segundos": duracao,
                **(metadados or {}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plotar_curvas_treino(historico, dir_saidas / f"{nome}_curvas_treino.png")

    return historico, caminho_checkpoint


# ==========================================
# CLI
# ==========================================
def analisar_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina a ResNet para classificação de AML.")
    parser.add_argument("--epocas", type=int, default=config.EPOCAS)
    parser.add_argument("--lote", type=int, default=config.TAMANHO_LOTE)
    parser.add_argument("--lr", type=float, default=config.TAXA_APRENDIZADO)
    parser.add_argument("--decaimento-peso", type=float, default=config.DECAIMENTO_PESO)
    parser.add_argument("--tamanho-imagem", type=int, default=config.TAMANHO_IMAGEM)
    parser.add_argument("--arquitetura", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--congelar-backbone", action="store_true", help="treina só a camada final")
    parser.add_argument(
        "--estrategia",
        default="paciente",
        choices=["paciente", "aleatorio"],
        help="divisão dos dados; 'aleatorio' só para demonstrar o vazamento",
    )
    parser.add_argument("--sem-pesos-classe", action="store_true", help="desliga os pesos de classe")
    parser.add_argument("--sem-aumento", action="store_true", help="desliga o aumento de dados")
    parser.add_argument("--sem-amp", action="store_true", help="desliga a precisão mista")
    parser.add_argument("--paciencia", type=int, default=config.PACIENCIA)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--semente", type=int, default=config.SEED)
    parser.add_argument("--nome", default=None, help="prefixo dos arquivos em outputs/")
    parser.add_argument(
        "--max-amostras",
        type=int,
        default=None,
        help="sorteia no máximo N imagens por conjunto (teste rápido do pipeline)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = analisar_argumentos(argv)
    nome = args.nome or f"{args.arquitetura}_{args.estrategia}"

    definir_semente(args.semente)
    dispositivo = obter_dispositivo()

    print("\nMontando os DataLoaders...")
    loaders, divisoes = criar_dataloaders(
        tamanho_lote=args.lote,
        tamanho_imagem=args.tamanho_imagem,
        semente=args.semente,
        num_workers=args.workers,
        estrategia=args.estrategia,
        aumento_dados=not args.sem_aumento,
        max_amostras=args.max_amostras,
    )
    print(resumir_divisao(divisoes))

    if args.max_amostras:
        print(f"\nMODO TESTE: no máximo {args.max_amostras} imagens por conjunto.")

    modelo = criar_modelo(
        arquitetura=args.arquitetura,
        num_classes=len(config.CLASSES),
        congelar_backbone=args.congelar_backbone,
        dropout=args.dropout,
    )
    total, treinaveis = contar_parametros(modelo)
    print(f"\nModelo: {args.arquitetura} | {total:,} parâmetros ({treinaveis:,} treináveis)")

    pesos = None if args.sem_pesos_classe else calcular_pesos_classes(divisoes["treino"])
    if pesos is not None:
        print(f"Pesos de classe: {config.CLASSES[0]}={pesos[0]:.3f}, {config.CLASSES[1]}={pesos[1]:.3f}")

    metadados = {
        "arquitetura": args.arquitetura,
        "estrategia_divisao": args.estrategia,
        "tamanho_imagem": args.tamanho_imagem,
        "tamanho_lote": args.lote,
        "taxa_aprendizado": args.lr,
        "semente": args.semente,
        "aumento_dados": not args.sem_aumento,
        "pesos_classe": pesos.tolist() if pesos is not None else None,
    }

    historico, caminho_checkpoint = treinar(
        modelo,
        loaders,
        dispositivo,
        epocas=args.epocas,
        taxa_aprendizado=args.lr,
        decaimento_peso=args.decaimento_peso,
        pesos_classes=pesos,
        paciencia=args.paciencia,
        usar_amp=not args.sem_amp,
        nome=nome,
        metadados=metadados,
    )

    # Avaliação final no conjunto de teste, com os melhores pesos recarregados.
    print("\nAvaliando o melhor checkpoint no conjunto de TESTE...")
    checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(checkpoint["estado_modelo"])
    _, y_verdadeiro, y_predito, y_prob = avaliar(
        modelo, loaders["teste"], None, dispositivo, usar_amp=not args.sem_amp, descricao="Teste"
    )
    metricas_teste = calcular_metricas(y_verdadeiro, y_predito, y_prob)
    print("\n" + formatar_metricas(metricas_teste, "Resultado no conjunto de TESTE"))

    dir_saidas = Path(config.DIR_SAIDAS)
    (dir_saidas / f"{nome}_metricas_teste.json").write_text(
        json.dumps({**metricas_teste, **metadados}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plotar_matriz_confusao(y_verdadeiro, y_predito, dir_saidas / f"{nome}_matriz_confusao.png")
    print(f"\nArtefatos salvos em {dir_saidas}")


if __name__ == "__main__":
    # O guard é obrigatório no Windows: com num_workers > 0 o PyTorch cria os
    # processos por spawn, e cada um reimporta este arquivo.
    main()
