# TCC — Classificação de Leucemia Mieloide Aguda (AML) com Deep Learning

Projeto de Trabalho de Conclusão de Curso (TCC) para detecção de **Leucemia Mieloide Aguda (AML)** a partir de imagens de citomorfologia de células sanguíneas, utilizando **Visão Computacional** e **PyTorch**.

O objetivo é treinar um modelo de deep learning capaz de distinguir imagens de células **saudáveis** de células com **leucemia**, usando o dataset público **AML-Cytomorphology_MLL_Helmholtz**.

---

## Visão geral do pipeline

O desenvolvimento segue um fluxo em etapas, implementado em notebooks Jupyter:

```
Dataset TCIA (raw)
       │
       ▼
[01] Análise exploratória ──► entender desequilíbrio de classes
       │
       ▼
[03] Organizar dataset ─────► binário: Saudaveis / Leucemia
       │
       ▼
[04] Pipeline PyTorch ──────► transformações, splits, DataLoaders
       │
       ▼
[05] Treinamento (ResNet18) ► transfer learning na GPU
       │
       ▼
   outputs/ (modelos salvos — pendente)
```

> **Nota:** O notebook `02` não existe no repositório. Provavelmente correspondia à etapa de download e extração do dataset a partir do TCIA.

---

## Dataset

### Fonte

- **Nome:** AML-Cytomorphology_MLL_Helmholtz
- **Origem:** Munich Leukemia Laboratory (MLL), via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/)
- **DOI:** [10.7937/6PPE-4020](https://doi.org/10.7937/6PPE-4020)
- **Total:** 81.214 imagens de células individuais (`.tif`), 189 pacientes, anos 2009–2020
- **Microscopia:** 40x, imersão em óleo, 144×144 pixels por célula

### Estrutura original (`data/raw/`)

O dataset original possui 5 categorias genéticas (WHO 2022):

| Pasta | Descrição |
|-------|-----------|
| `control/` | Doadores saudáveis (controles) |
| `PML_RARA/` | APL com fusão PML::RARA |
| `NPM1/` | AML com mutação NPM1 |
| `CBFB_MYH11/` | AML com fusão CBFB::MYH11 |
| `RUNX1_RUNX1T1/` | AML com fusão RUNX1::RUNX1T1 |

Cada pasta contém subpastas por paciente (ex.: `AEC/`, `AQK/`) com dezenas a centenas de imagens `.tif`.

### Dataset processado (`data/processed/dataset_binario/`)

O notebook `03_organizar_dataset.ipynb` agrupa as 4 subpastas de AML em uma única classe binária:

| Classe | Origem | Imagens |
|--------|--------|---------|
| `Saudaveis/` | `control/` | 20.305 |
| `Leucemia/` | `RUNX1_RUNX1T1`, `CBFB_MYH11`, `PML_RARA`, `NPM1` | 60.909 |
| **Total** | | **81.214** |

Arquivos renomeados com prefixo do paciente para evitar colisões (ex.: `AEC_image_0.tif`).

### Desequilíbrio de classes

Aproximadamente **25% saudáveis** vs **75% leucemia**. Esse desequilíbrio é explorado visualmente no notebook `01` e deve ser considerado na avaliação do modelo (métricas além da acurácia, como recall, F1 e matriz de confusão).

---

## Estrutura do projeto

```
tcc-leucemia-ia/
├── data/
│   ├── raw/                          # Dataset original (5 categorias)
│   │   ├── control/
│   │   ├── PML_RARA/
│   │   ├── NPM1/
│   │   ├── CBFB_MYH11/
│   │   └── RUNX1_RUNX1T1/
│   └── processed/
│       └── dataset_binario/          # Dataset binário para treino
│           ├── Saudaveis/
│           └── Leucemia/
├── notebooks/
│   ├── 01_analise_dados.ipynb        # Análise exploratória
│   ├── 03_organizar_dataset.ipynb    # Preparação do dataset binário
│   ├── 04_pipeline_pytorch.ipynb     # Pipeline de dados PyTorch
│   └── 05_treinamento_modelo.ipynb   # Treinamento com ResNet18
├── src/
│   ├── models/                       # (vazio — reservado para código modular)
│   └── utils/                        # (vazio — reservado para utilitários)
├── outputs/                          # (vazio — destino de modelos treinados)
└── README.md
```

---

## Notebooks — detalhamento

### 01 — Análise de dados (`01_analise_dados.ipynb`)

**Objetivo:** Validar o ambiente Python e visualizar o desafio de desequilíbrio de classes.

- Instala `matplotlib`, `pandas`, `numpy`
- Gera gráfico de pizza simulando a proporção saudáveis vs AML
- Confirma que o ambiente gráfico funciona no Cursor/Jupyter

### 03 — Organizar dataset (`03_organizar_dataset.ipynb`)

**Objetivo:** Converter o dataset multi-classe em classificação binária.

- Percorre recursivamente `data/raw/`
- Classifica imagens (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`) por pasta de origem
- Copia para `data/processed/dataset_binario/Saudaveis/` ou `Leucemia/`
- Resultado: **81.214 imagens** copiadas em ~2 minutos

### 04 — Pipeline PyTorch (`04_pipeline_pytorch.ipynb`)

**Objetivo:** Montar o pipeline de dados para alimentar a GPU.

| Parâmetro | Valor |
|-----------|-------|
| Tamanho da imagem | 224 × 224 px |
| Normalização | ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) |
| Batch size | 32 |
| Split treino/val/teste | 70% / 15% / 15% |
| Seed | 42 (reprodutibilidade) |
| DataLoaders | `train_loader`, `val_loader`, `test_loader` |

Com 81.214 imagens e batch de 32, o treino gera ~**1.777 lotes** por época.

### 05 — Treinamento do modelo (`05_treinamento_modelo.ipynb`)

**Objetivo:** Treinar uma **ResNet18** com transfer learning na GPU.

**Configuração detectada na execução:**
- GPU: NVIDIA GeForce RTX 2070 SUPER (8,59 GB VRAM)
- Modelo base: ResNet18 pré-treinada (ImageNet), baixada automaticamente
- Framework: PyTorch + torchvision

**Status atual:** O notebook está **incompleto**. O código-fonte foi truncado na linha `transformacoes = transforms.` e a execução falhou com:

```
NameError: name 'train_loader' is not defined
```

Para concluir o treinamento, é necessário completar o notebook incorporando o pipeline de dados do notebook `04` (criação dos DataLoaders) antes do loop de treino.

---

## Pré-requisitos

### Hardware

- **Recomendado:** GPU NVIDIA com CUDA (testado com RTX 2070 SUPER, 8 GB VRAM)
- **Alternativa:** CPU (funciona, porém treinamento muito mais lento)

### Software

- Python 3.12+
- Jupyter Notebook ou JupyterLab
- CUDA (se usar GPU)

### Dependências Python

Instaladas inline nos notebooks via `%pip install`. Pacotes utilizados:

```
torch
torchvision
matplotlib
pandas
numpy
tqdm
pillow
```

> **Sugestão:** Criar um `requirements.txt` para facilitar a reprodução do ambiente.

---

## Como executar

### 1. Obter o dataset

Baixe o **AML-Cytomorphology_MLL_Helmholtz** pelo [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) (requer registro). Extraia o conteúdo em:

```
data/raw/
├── control/
├── PML_RARA/
├── NPM1/
├── CBFB_MYH11/
└── RUNX1_RUNX1T1/
```

### 2. Instalar dependências

```bash
pip install torch torchvision matplotlib pandas numpy tqdm pillow jupyter
```

Para GPU NVIDIA, instale o PyTorch com suporte CUDA conforme [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Executar os notebooks em ordem

Abra a pasta `notebooks/` no Jupyter e execute na sequência:

1. `01_analise_dados.ipynb` — validação do ambiente
2. `03_organizar_dataset.ipynb` — gera o dataset binário (~2 min)
3. `04_pipeline_pytorch.ipynb` — valida o pipeline de dados
4. `05_treinamento_modelo.ipynb` — treina o modelo (requer conclusão do código)

---

## Arquitetura do modelo (planejada)

```
Entrada: imagem 224×224×3 (tensor normalizado)
    │
    ▼
ResNet18 (pré-treinada ImageNet)
    │
    ▼
Camada fully-connected adaptada (2 classes)
    │
    ▼
Saída: Saudável (0) ou Leucemia (1)
```

Abordagem de **transfer learning**: reutiliza pesos aprendidos no ImageNet e ajusta a camada final para a tarefa binária de classificação.

---

## Histórico de desenvolvimento

| Etapa | Status | Descrição |
|-------|--------|-----------|
| Setup do ambiente Python | Concluído | Bibliotecas instaladas e validadas no Cursor |
| Download do dataset TCIA | Concluído | 81.214 imagens em `data/raw/` |
| Análise exploratória | Concluído | Visualização do desequilíbrio de classes |
| Preparação dataset binário | Concluído | 20.305 saudáveis + 60.909 leucemia |
| Pipeline PyTorch | Concluído | DataLoaders com split 70/15/15 |
| Treinamento ResNet18 | **Pendente** | Notebook incompleto, sem modelo salvo |
| Modularização (`src/`) | **Pendente** | Pastas criadas, sem código |
| Avaliação e métricas | **Pendente** | Matriz de confusão, F1, recall |
| Exportação do modelo | **Pendente** | Pasta `outputs/` vazia |

---

## Próximos passos sugeridos

1. **Completar o notebook `05`** — integrar o pipeline do notebook `04` e finalizar o loop de treino com ResNet18
2. **Salvar checkpoints** em `outputs/` (melhor modelo por acurácia de validação)
3. **Avaliar no conjunto de teste** — matriz de confusão, precision, recall, F1
4. **Tratar desequilíbio** — pesos de classe, oversampling ou métricas adequadas
5. **Modularizar código** — mover lógica para `src/models/` e `src/utils/`
6. **Criar `requirements.txt`** — fixar versões das dependências
7. **Inicializar git** — controle de versão (dataset excluído via `.gitignore`)

---

## Referências

- Hehr, M., Sadafi, A., Matek, C., et al. (2023). *A morphological dataset of white blood cells from patients with four different genetic AML entities and non-malignant controls (AML-Cytomorphology_MLL_Helmholtz)*. The Cancer Imaging Archive. https://doi.org/10.7937/6PPE-4020
- [PyTorch — Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [torchvision — ResNet](https://pytorch.org/vision/stable/models/resnet.html)

---

## Licença e uso

Este projeto é acadêmico (TCC). O dataset AML-Cytomorphology possui termos de uso próprios do TCIA — consulte a [página oficial](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) antes de redistribuir os dados.
