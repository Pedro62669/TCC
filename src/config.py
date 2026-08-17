"""Configurações centrais do projeto.

Todos os caminhos são derivados da raiz do repositório, então os módulos
funcionam igual sendo chamados de `src/`, de `notebooks/` ou da raiz.
"""

from pathlib import Path

# ==========================================
# Caminhos
# ==========================================
RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "data" / "processed" / "dataset_binario"
DIR_SAIDAS = RAIZ / "outputs"

# ==========================================
# Classes
# ==========================================
# A ordem define o índice numérico: Saudaveis = 0, Leucemia = 1.
# "Leucemia" é a classe positiva — é dela que saem recall/precisão/F1.
CLASSES = ("Saudaveis", "Leucemia")
CLASSE_POSITIVA = 1

# ==========================================
# Dados
# ==========================================
SEED = 42
TAMANHO_IMAGEM = 224  # ResNet pré-treinada na ImageNet espera 224x224
TAMANHO_LOTE = 32
PROPORCOES = (0.70, 0.15, 0.15)  # treino / validação / teste
EXTENSOES_IMAGEM = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Estatísticas da ImageNet — necessárias porque usamos pesos pré-treinados nela
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

# ==========================================
# Treinamento
# ==========================================
EPOCAS = 15
TAXA_APRENDIZADO = 1e-4
DECAIMENTO_PESO = 1e-4
PACIENCIA = 5  # épocas sem melhora antes do early stopping
NUM_WORKERS = 4
