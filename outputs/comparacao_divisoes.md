# Comparação entre estratégias de divisão

Nível célula, conjunto de teste. Mesma arquitetura (ResNet18), mesmos hiperparâmetros; muda apenas a divisão dos dados.

| Métrica                | Divisão aleatória | Divisão por paciente | Diferença |
|------------------------|-------------------|----------------------|-----------|
| Acurácia               |            0,9696 |               0,9546 |   −0,0150 |
| Acurácia balanceada    |            0,9665 |               0,9580 |   −0,0085 |
| Precisão (leucemia)    |            0,9863 |               0,9878 |   +0,0015 |
| Recall / sensibilidade |            0,9728 |               0,9512 |   −0,0216 |
| Especificidade         |            0,9601 |               0,9648 |   +0,0047 |
| F1-score               |            0,9795 |               0,9692 |   −0,0103 |
| AUC-ROC                |            0,9955 |               0,9923 |   −0,0032 |
