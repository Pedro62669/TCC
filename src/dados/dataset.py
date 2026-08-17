"""Pipeline de dados com divisão por paciente.

Por que não usar `ImageFolder` + `random_split`?
------------------------------------------------
Cada paciente contribui com centenas de células da mesma lâmina — mesma
coloração, mesmo microscópio, mesmo dia. Se as imagens de um paciente caírem ao
mesmo tempo no treino e no teste, o modelo pode acertar simplesmente por
reconhecer o paciente, e não a doença. Isso é vazamento de dados
(*data leakage*) e infla as métricas a ponto de torná-las inúteis.

A divisão aqui é feita no nível do **paciente**: os 189 pacientes são
distribuídos entre treino/validação/teste, e todas as imagens de um paciente vão
junto com ele. É o procedimento padrão em imagem médica.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


@dataclass(frozen=True)
class Amostra:
    """Uma imagem de célula: onde está, de quem é e a que classe pertence."""

    caminho: Path
    rotulo: int
    paciente: str


# ==========================================
# Leitura do dataset
# ==========================================
def _extrair_paciente(nome_arquivo: str) -> str:
    """`AQK_image_0.tif` -> `AQK`.

    O prefixo é criado pelo notebook 03 a partir da pasta de origem do paciente.
    """
    return nome_arquivo.split("_")[0]


def listar_amostras(
    diretorio: Path | str | None = None,
    classes: Sequence[str] = config.CLASSES,
) -> list[Amostra]:
    """Varre `dataset_binario/` e devolve a lista de amostras.

    Levanta erro se alguma pasta de classe faltar ou estiver vazia — melhor
    falhar aqui do que treinar num dataset pela metade sem perceber.
    """
    diretorio = Path(diretorio or config.DIR_DADOS)
    if not diretorio.is_dir():
        raise FileNotFoundError(
            f"Dataset não encontrado em {diretorio}. "
            "Rode o notebook 03_organizar_dataset.ipynb primeiro."
        )

    amostras: list[Amostra] = []
    for rotulo, classe in enumerate(classes):
        pasta = diretorio / classe
        if not pasta.is_dir():
            raise FileNotFoundError(f"Pasta da classe '{classe}' não existe em {diretorio}.")

        encontradas = [
            Amostra(caminho=arquivo, rotulo=rotulo, paciente=_extrair_paciente(arquivo.name))
            for arquivo in sorted(pasta.iterdir())
            if arquivo.suffix.lower() in config.EXTENSOES_IMAGEM
        ]
        if not encontradas:
            raise RuntimeError(f"Nenhuma imagem encontrada em {pasta}.")
        amostras.extend(encontradas)

    return amostras


# ==========================================
# Divisões
# ==========================================
def dividir_por_paciente(
    amostras: Iterable[Amostra],
    proporcoes: tuple[float, float, float] = config.PROPORCOES,
    semente: int = config.SEED,
) -> dict[str, list[Amostra]]:
    """Divide as amostras garantindo que nenhum paciente cruze conjuntos.

    A divisão é estratificada: cada classe é distribuída separadamente, então
    treino, validação e teste mantêm mais ou menos a mesma proporção
    saudáveis/leucemia do dataset completo.

    Como os pacientes têm de 99 a 500 imagens cada, as proporções finais de
    *imagens* ficam próximas — mas não exatamente iguais — às pedidas. Isso é
    inerente a dividir no nível do paciente e é o preço correto a pagar.
    """
    if not 0.999 < sum(proporcoes) < 1.001:
        raise ValueError(f"As proporções devem somar 1.0, recebido {sum(proporcoes)}.")

    amostras = list(amostras)
    prop_treino, prop_val, _ = proporcoes
    divisoes: dict[str, list[Amostra]] = {"treino": [], "validacao": [], "teste": []}

    # Agrupa por classe -> paciente -> imagens
    por_classe: dict[int, dict[str, list[Amostra]]] = defaultdict(lambda: defaultdict(list))
    for amostra in amostras:
        por_classe[amostra.rotulo][amostra.paciente].append(amostra)

    rng = random.Random(semente)

    for rotulo in sorted(por_classe):
        pacientes = sorted(por_classe[rotulo])  # sorted() torna o embaralhamento reprodutível
        rng.shuffle(pacientes)

        total_classe = sum(len(por_classe[rotulo][p]) for p in pacientes)
        alvo_treino = prop_treino * total_classe
        alvo_val = (prop_treino + prop_val) * total_classe

        acumulado = 0
        for paciente in pacientes:
            imagens = por_classe[rotulo][paciente]
            # O paciente inteiro vai para o conjunto onde seu *início* cai.
            if acumulado < alvo_treino:
                destino = "treino"
            elif acumulado < alvo_val:
                destino = "validacao"
            else:
                destino = "teste"
            divisoes[destino].extend(imagens)
            acumulado += len(imagens)

    for nome, conjunto in divisoes.items():
        if not conjunto:
            raise RuntimeError(
                f"A divisão '{nome}' ficou vazia. Ajuste as proporções "
                f"(há apenas {len(set(a.paciente for a in amostras))} pacientes)."
            )

    return divisoes


def dividir_aleatorio(
    amostras: Iterable[Amostra],
    proporcoes: tuple[float, float, float] = config.PROPORCOES,
    semente: int = config.SEED,
) -> dict[str, list[Amostra]]:
    """Divisão aleatória por imagem, ignorando o paciente.

    Existe só para efeito de comparação no TCC: treinar com esta divisão e com
    `dividir_por_paciente` mostra numericamente o quanto o vazamento de dados
    infla as métricas. **Não use para reportar o resultado final.**
    """
    amostras = list(amostras)
    rng = random.Random(semente)
    rng.shuffle(amostras)

    total = len(amostras)
    n_treino = int(proporcoes[0] * total)
    n_val = int(proporcoes[1] * total)

    return {
        "treino": amostras[:n_treino],
        "validacao": amostras[n_treino : n_treino + n_val],
        "teste": amostras[n_treino + n_val :],
    }


def resumir_divisao(divisoes: dict[str, list[Amostra]]) -> str:
    """Tabela legível com imagens, pacientes e balanceamento por conjunto."""
    total_imagens = sum(len(v) for v in divisoes.values())
    linhas = [
        f"{'Conjunto':<12}{'Imagens':>9}{'%':>7}{'Pacientes':>11}{'Saudáveis':>11}{'Leucemia':>10}",
        "-" * 60,
    ]
    for nome in ("treino", "validacao", "teste"):
        conjunto = divisoes[nome]
        contagem = Counter(a.rotulo for a in conjunto)
        pacientes = len({a.paciente for a in conjunto})
        linhas.append(
            f"{nome:<12}{len(conjunto):>9,}{100 * len(conjunto) / total_imagens:>6.1f}%"
            f"{pacientes:>11}{contagem[0]:>11,}{contagem[1]:>10,}"
        )
    linhas.append("-" * 60)
    linhas.append(f"{'TOTAL':<12}{total_imagens:>9,}")

    # Checagem explícita: nenhum paciente pode aparecer em dois conjuntos.
    conjuntos_pacientes = {n: {a.paciente for a in v} for n, v in divisoes.items()}
    vazamento = (
        (conjuntos_pacientes["treino"] & conjuntos_pacientes["validacao"])
        | (conjuntos_pacientes["treino"] & conjuntos_pacientes["teste"])
        | (conjuntos_pacientes["validacao"] & conjuntos_pacientes["teste"])
    )
    linhas.append(
        "OK: nenhum paciente compartilhado entre os conjuntos."
        if not vazamento
        else f"ATENÇÃO: pacientes em mais de um conjunto: {sorted(vazamento)}"
    )
    return "\n".join(linhas)


# ==========================================
# Dataset e transformações
# ==========================================
class DatasetCelulas(Dataset):
    """Dataset PyTorch sobre uma lista de `Amostra`."""

    def __init__(self, amostras: Sequence[Amostra], transformacao=None):
        self.amostras = list(amostras)
        self.transformacao = transformacao

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        amostra = self.amostras[indice]
        # convert("RGB") é obrigatório: os .tif do MLL abrem em modos variados e
        # a ResNet espera 3 canais.
        imagem = Image.open(amostra.caminho).convert("RGB")
        if self.transformacao is not None:
            imagem = self.transformacao(imagem)
        return imagem, amostra.rotulo


def criar_transformacoes(treino: bool, tamanho: int = config.TAMANHO_IMAGEM):
    """Transformações de imagem.

    No treino aplicamos aumento de dados (*data augmentation*). Espelhamentos e
    rotações são seguros aqui porque uma célula não tem orientação canônica — a
    posição dela na lâmina é arbitrária. O jitter de cor/brilho simula variação
    de coloração e iluminação entre lâminas, que é a maior fonte de diferença
    entre pacientes.
    """
    normalizacao = transforms.Normalize(mean=config.MEDIA_IMAGENET, std=config.DESVIO_IMAGENET)

    if not treino:
        return transforms.Compose(
            [
                transforms.Resize((tamanho, tamanho)),
                transforms.ToTensor(),
                normalizacao,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((tamanho, tamanho)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalizacao,
        ]
    )


def calcular_pesos_classes(amostras: Sequence[Amostra], num_classes: int = 2) -> torch.Tensor:
    """Pesos inversamente proporcionais à frequência de cada classe.

    O dataset é ~25% saudáveis / ~75% leucemia. Sem correção, o modelo aprende
    que chutar "leucemia" já garante 75% de acurácia. Estes pesos entram na
    `CrossEntropyLoss` e fazem cada erro na classe minoritária custar mais.
    """
    contagem = Counter(a.rotulo for a in amostras)
    total = sum(contagem.values())
    pesos = [total / (num_classes * max(contagem[i], 1)) for i in range(num_classes)]
    return torch.tensor(pesos, dtype=torch.float32)


# ==========================================
# DataLoaders
# ==========================================
def criar_dataloaders(
    diretorio: Path | str | None = None,
    tamanho_lote: int = config.TAMANHO_LOTE,
    tamanho_imagem: int = config.TAMANHO_IMAGEM,
    proporcoes: tuple[float, float, float] = config.PROPORCOES,
    semente: int = config.SEED,
    num_workers: int = config.NUM_WORKERS,
    estrategia: str = "paciente",
    aumento_dados: bool = True,
    descartar_ultimo: bool = True,
    max_amostras: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, list[Amostra]]]:
    """Monta os três DataLoaders e devolve também as divisões brutas.

    `estrategia`: "paciente" (correto) ou "aleatorio" (só para comparação).

    `descartar_ultimo` (drop_last) descarta o último lote incompleto do treino —
    lotes minúsculos deixam a BatchNorm instável. Passe `False` quando precisar
    que as predições casem 1-a-1 com `divisoes["treino"]`.

    `max_amostras` sorteia no máximo N imagens por conjunto; serve para validar o
    pipeline em segundos sem varrer as 81 mil imagens.
    """
    if estrategia not in {"paciente", "aleatorio"}:
        raise ValueError(f"estratégia inválida: {estrategia!r} (use 'paciente' ou 'aleatorio')")

    amostras = listar_amostras(diretorio)
    divisor = dividir_por_paciente if estrategia == "paciente" else dividir_aleatorio
    divisoes = divisor(amostras, proporcoes=proporcoes, semente=semente)

    if max_amostras is not None:
        # Sorteio aleatório (e não os N primeiros): pegar os primeiros traria só
        # os pacientes do começo da lista, e portanto uma única classe.
        rng = random.Random(semente)
        divisoes = {
            nome: rng.sample(conjunto, min(len(conjunto), max_amostras))
            for nome, conjunto in divisoes.items()
        }

    transf_treino = criar_transformacoes(treino=aumento_dados, tamanho=tamanho_imagem)
    transf_aval = criar_transformacoes(treino=False, tamanho=tamanho_imagem)

    datasets = {
        "treino": DatasetCelulas(divisoes["treino"], transf_treino),
        "validacao": DatasetCelulas(divisoes["validacao"], transf_aval),
        "teste": DatasetCelulas(divisoes["teste"], transf_aval),
    }

    comum = {
        "batch_size": tamanho_lote,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        **({"prefetch_factor": 4} if num_workers > 0 else {}),
    }

    # `persistent_workers` só no treino. No Windows cada worker é um processo
    # novo que reimporta o torch inteiro (~1 GB de commit cada); manter os três
    # loaders com workers vivos ao mesmo tempo estoura o arquivo de paginação
    # (OSError WinError 1455). Os de validação/teste sobem e descem por época.
    persistente = {"persistent_workers": True} if num_workers > 0 else {}

    loaders = {
        "treino": DataLoader(
            datasets["treino"],
            shuffle=True,
            drop_last=descartar_ultimo,
            **comum,
            **persistente,
        ),
        "validacao": DataLoader(datasets["validacao"], shuffle=False, **comum),
        "teste": DataLoader(datasets["teste"], shuffle=False, **comum),
    }
    return loaders, divisoes
