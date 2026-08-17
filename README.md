# TCC — Classificação de Leucemia Mieloide Aguda (AML) com Deep Learning

Detecção de **Leucemia Mieloide Aguda** a partir de imagens de citomorfologia de células sanguíneas, usando visão computacional e PyTorch.

O modelo classifica células individuais como **saudáveis** ou **leucêmicas** e agrega essas predições num diagnóstico por **paciente**, que é o que interessa clinicamente. O dataset é o **AML-Cytomorphology_MLL_Helmholtz**: 81.214 imagens de 189 pacientes.

---

## Resultados

ResNet18 com transfer learning, divisão por paciente, 15 épocas (melhor na 12ª por early stopping).

### Nível célula — conjunto de teste (12.039 imagens)

| Métrica | Valor |
|---|---:|
| Acurácia | 0,9546 |
| Acurácia balanceada | 0,9580 |
| Precisão (leucemia) | 0,9878 |
| Recall / sensibilidade | 0,9512 |
| Especificidade | 0,9648 |
| F1-score | 0,9692 |
| AUC-ROC | 0,9923 |

Matriz de confusão: VP = 8.585 · VN = 2.908 · FP = 106 · FN = 440

### Nível paciente — 30 pacientes de teste

Agregando as centenas de células de cada paciente numa decisão única, **todas as métricas chegam a 1,0**: os 30 pacientes (10 saudáveis, 20 com leucemia) foram classificados corretamente, sem nenhum erro.

O ganho não é acidental. Um paciente traz de 99 a 500 células; erros isolados em células individuais são diluídos pela média das probabilidades, e o sinal de doença — presente na maioria das células — prevalece.

> Ressalva honesta para a defesa: 100% sobre **30 pacientes** é um resultado forte, mas a amostra é pequena. Um intervalo de confiança exigiria validação cruzada por paciente (ver [Próximos passos](#próximos-passos)).

---

## O achado metodológico: vazamento de dados

Este é o ponto central do trabalho, e não estava previsto no plano inicial.

Cada paciente contribui com centenas de células **da mesma lâmina** — mesma coloração, mesmo microscópio, mesmo dia. A abordagem intuitiva (`ImageFolder` + `random_split`, dividindo por imagem) espalha células de um mesmo paciente entre treino e teste. O modelo então pode acertar por **reconhecer o paciente**, não a doença. Isso é *data leakage*, e infla as métricas.

Para medir o efeito, os dois cenários foram treinados com hiperparâmetros idênticos, mudando **apenas** a estratégia de divisão:

| Métrica | Divisão aleatória | Divisão por paciente | Diferença |
|---|---:|---:|---:|
| Acurácia | 0,9696 | 0,9546 | −0,0150 |
| Acurácia balanceada | 0,9665 | 0,9580 | −0,0085 |
| Precisão (leucemia) | 0,9863 | 0,9878 | +0,0015 |
| Recall / sensibilidade | 0,9728 | 0,9512 | −0,0216 |
| Especificidade | 0,9601 | 0,9648 | +0,0047 |
| F1-score | 0,9795 | 0,9692 | −0,0103 |
| AUC-ROC | 0,9955 | 0,9923 | −0,0032 |

A divisão aleatória parece melhor, mas o ganho é ilusório: vem de reconhecer pacientes já vistos. **Os números reportados neste trabalho são os da divisão por paciente**, mesmo sendo mais baixos.

Reproduzir: `python -m src.comparar` — gera `outputs/comparacao_divisoes.md` e a figura correspondente.

---

## Dataset

- **Nome:** AML-Cytomorphology_MLL_Helmholtz
- **Origem:** Munich Leukemia Laboratory (MLL), via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/)
- **DOI:** [10.7937/6PPE-4020](https://doi.org/10.7937/6PPE-4020)
- **Conteúdo:** 81.214 imagens `.tif` de células individuais, 189 pacientes, 2009–2020
- **Microscopia:** 40x com imersão em óleo, 144×144 px por célula

### Estrutura original (`data/raw/`)

| Pasta | Descrição | Pacientes | Imagens |
|---|---|---:|---:|
| `control/` | Doadores saudáveis | 60 | 20.305 |
| `CBFB_MYH11/` | AML com fusão CBFB::MYH11 | 37 | 17.212 |
| `NPM1/` | AML com mutação NPM1 | 36 | 17.710 |
| `RUNX1_RUNX1T1/` | AML com fusão RUNX1::RUNX1T1 | 32 | 14.403 |
| `PML_RARA/` | APL com fusão PML::RARA | 24 | 11.584 |
| **Total** | | **189** | **81.214** |

### Dataset binário (`data/processed/dataset_binario/`)

O notebook `03` agrupa as 4 entidades de AML numa única classe:

| Classe | Imagens | Proporção |
|---|---:|---:|
| `Saudaveis/` | 20.305 | 25,0% |
| `Leucemia/` | 60.909 | 75,0% |

Os arquivos recebem o prefixo do paciente (`AQK_image_0.tif`) — é desse prefixo que a divisão por paciente extrai o identificador.

**O desbalanceamento 25/75 importa:** um modelo que respondesse "leucemia" para tudo já teria 75% de acurácia. Por isso o treino usa pesos de classe inversamente proporcionais à frequência (`[1,98, 0,67]`) e o melhor checkpoint é escolhido pela **acurácia balanceada**, não pela acurácia simples.

---

## Estrutura do projeto

```
tcc-leucemia-ia/
├── data/
│   ├── raw/                        # dataset original do TCIA (5 categorias)
│   └── processed/dataset_binario/  # Saudaveis/ e Leucemia/
├── notebooks/
│   ├── 01_analise_dados.ipynb      # validação do ambiente
│   ├── 03_organizar_dataset.ipynb  # monta o dataset binário
│   ├── 04_pipeline_pytorch.ipynb   # abordagem inicial (com vazamento — ver nota no notebook)
│   └── 05_treinamento_modelo.ipynb # treino interativo sobre os módulos de src/
├── src/
│   ├── config.py                   # caminhos, classes, hiperparâmetros, semente
│   ├── dados/dataset.py            # divisão por paciente, transformações, DataLoaders
│   ├── modelos/resnet.py           # ResNet18/34/50 com a camada final adaptada
│   ├── treino.py                   # loop de treino (CLI)
│   ├── avaliar.py                  # avaliação + agregação por paciente (CLI)
│   ├── comparar.py                 # comparação entre estratégias de divisão (CLI)
│   └── utils/
│       ├── metricas.py             # métricas e gráficos
│       └── semente.py              # reprodutibilidade
├── tests/                          # testes da divisão por paciente
├── outputs/                        # métricas, figuras e checkpoints
└── requirements.txt
```

Os módulos vivem em `src/` e os notebooks apenas os chamam. A lógica existe num lugar só, então o que roda no notebook e o que roda pela linha de comando são a mesma coisa.

---

## Como executar

### 1. Dataset

Baixe o **AML-Cytomorphology_MLL_Helmholtz** pelo [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) (requer registro) e extraia em `data/raw/`, preservando as 5 subpastas. Depois rode `notebooks/03_organizar_dataset.ipynb` para gerar o dataset binário (~2 min).

### 2. Dependências

```bash
pip install -r requirements.txt
```

O `requirements.txt` fixa a build CUDA 11.8 do PyTorch. Sem ela o pip instala a versão de CPU e o treino passa de ~35 min para várias horas.

### 3. Treinar

```bash
# Treino padrão: divisão por paciente, 15 épocas
python -m src.treino

# Verificação rápida do pipeline (~1 min, não serve para reportar resultado)
python -m src.treino --epocas 2 --max-amostras 800 --nome smoke_test

# Reproduzir o experimento de vazamento
python -m src.treino --estrategia aleatorio --nome comparacao_leakage
```

Principais opções: `--epocas --lote --lr --arquitetura {resnet18,resnet34,resnet50} --estrategia {paciente,aleatorio} --dropout --congelar-backbone --paciencia --workers --semente --nome --max-amostras`. Use `--help` para a lista completa.

### 4. Avaliar

```bash
python -m src.avaliar --checkpoint outputs/resnet18_paciente_melhor.pth
```

Gera métricas por célula **e** por paciente, matrizes de confusão, curva ROC e a lista de pacientes classificados errado.

### 5. Comparar as estratégias

```bash
python -m src.comparar
```

### 6. Testes

```bash
python -m pytest tests/
```

Os testes verificam o contrato que sustenta a metodologia: nenhum paciente aparece em mais de um conjunto. Rodam em segundos com dados sintéticos, sem precisar do dataset em disco.

---

## Detalhes de implementação

### Divisão por paciente

Os 189 pacientes são distribuídos entre treino/validação/teste (70/15/15), e **todas** as imagens de um paciente acompanham ele. A divisão é estratificada por classe, então os três conjuntos preservam a proporção 25/75. Como os pacientes têm de 99 a 500 imagens, as proporções finais de imagens ficam próximas — mas não exatamente iguais — às pedidas: é o preço correto por dividir no nível certo.

### Aumento de dados

Espelhamento horizontal e vertical, rotação de até 20°, e jitter de brilho/contraste/saturação/matiz. Rotações e espelhamentos são seguros porque uma célula não tem orientação canônica — sua posição na lâmina é arbitrária. O jitter de cor simula variação de coloração e iluminação entre lâminas, que é a maior fonte de diferença entre pacientes. Validação e teste recebem apenas redimensionamento e normalização.

### Treinamento

| Parâmetro | Valor |
|---|---|
| Arquitetura | ResNet18 pré-treinada (ImageNet), camada `fc` adaptada para 2 classes |
| Otimizador | AdamW, lr 1e-4, weight decay 1e-4 |
| Agendador | `ReduceLROnPlateau` (fator 0,5, paciência 2) |
| Perda | `CrossEntropyLoss` com pesos de classe |
| Precisão mista | AMP ativada (`torch.amp`) |
| Lote / imagem | 32 / 224×224 |
| Early stopping | paciência de 5 épocas |
| Critério do melhor | acurácia balanceada de validação |
| Semente | 42 |

### Agregação por paciente

`src/avaliar.py` calcula a probabilidade média de leucemia entre todas as células de um paciente e compara com o limiar (padrão 0,5). O relatório traz, por paciente, a probabilidade média, o número de células e a fração de células classificadas como leucêmicas.

---

## Ambiente de referência

- Windows 11 Pro, Python 3.12
- GPU NVIDIA RTX 2070 SUPER (8 GB VRAM), CUDA 11.8
- Tempo de treino: ~35 min por experimento de 15 épocas

Funciona em CPU, mas o treino fica ordens de grandeza mais lento.

> No Windows, o `num_workers > 0` faz o PyTorch criar processos por *spawn*, e cada um reimporta o torch inteiro. Dentro do Jupyter isso costuma travar ou estourar o arquivo de paginação, então o notebook 05 usa `num_workers = 0`; pela linha de comando, `--workers 4` funciona bem.

---

## Próximos passos

1. **Validação cruzada k-fold por paciente** — o resultado de 100% no nível paciente vem de 30 pacientes; k-fold daria intervalos de confiança.
2. **Repetir com várias sementes** — todos os resultados atuais vêm de `seed=42`; sem repetições não dá para separar efeito real de ruído, o que é especialmente relevante na comparação de vazamento.
3. **Grad-CAM** — mapas de atenção para mostrar que o modelo olha morfologia celular e não artefato de coloração.
4. **Comparar arquiteturas** — `criar_modelo()` já suporta resnet34 e resnet50.
5. **Classificação multiclasse** — as 4 entidades genéticas estão preservadas em `data/raw/`.

---

## Versionamento

`data/` e os checkpoints `.pth` ficam fora do git: o dataset vem do TCIA e os modelos (~45 MB cada) são reproduzíveis a partir do código e da semente. Já as métricas, figuras e tabelas de `outputs/` **são versionadas** — custam ~35 min de GPU cada e são os resultados que vão para a monografia.

---

## Referências

- Hehr, M., Sadafi, A., Matek, C., et al. (2023). *A morphological dataset of white blood cells from patients with four different genetic AML entities and non-malignant controls (AML-Cytomorphology_MLL_Helmholtz)*. The Cancer Imaging Archive. https://doi.org/10.7937/6PPE-4020
- [PyTorch — Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [torchvision — ResNet](https://pytorch.org/vision/stable/models/resnet.html)

## Licença e uso

Projeto acadêmico (TCC). O dataset AML-Cytomorphology possui termos de uso próprios do TCIA — consulte a [página oficial](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) antes de redistribuir os dados.
