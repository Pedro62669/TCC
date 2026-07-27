# Dataset

Baixe o **AML-Cytomorphology_MLL_Helmholtz** e extraia aqui:

- [TCIA — AML-Cytomorphology](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/)
- DOI: https://doi.org/10.7937/6PPE-4020

Estrutura esperada em `raw/`:

```
data/raw/
├── control/
├── PML_RARA/
├── NPM1/
├── CBFB_MYH11/
└── RUNX1_RUNX1T1/
```

Depois execute o notebook `03_organizar_dataset.ipynb` para gerar `processed/dataset_binario/`.
