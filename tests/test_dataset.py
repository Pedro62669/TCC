"""Testes da divisão de dados.

O que se testa aqui é a afirmação central da metodologia do TCC: a divisão é
feita no nível do paciente, de modo que nenhuma célula de um mesmo paciente
apareça em dois conjuntos. Se esse contrato quebrar, todas as métricas
reportadas passam a estar infladas por vazamento de dados — e nada no
treinamento acusaria o problema.

As amostras são sintéticas de propósito: os testes rodam em milissegundos e não
dependem das 81 mil imagens estarem presentes no disco.
"""

from pathlib import Path

import pytest

from src.dados.dataset import (
    Amostra,
    _extrair_paciente,
    calcular_pesos_classes,
    dividir_aleatorio,
    dividir_por_paciente,
)


def construir_amostras(
    pacientes_por_classe: int = 20,
    imagens_por_paciente: int = 50,
) -> list[Amostra]:
    """Gera amostras sintéticas imitando a nomenclatura real do dataset."""
    amostras = []
    for rotulo in (0, 1):
        for indice_paciente in range(pacientes_por_classe):
            paciente = f"P{rotulo}{indice_paciente:03d}"
            for indice_imagem in range(imagens_por_paciente):
                amostras.append(
                    Amostra(
                        caminho=Path(f"{paciente}_image_{indice_imagem}.tif"),
                        rotulo=rotulo,
                        paciente=paciente,
                    )
                )
    return amostras


def pacientes_de(conjunto: list[Amostra]) -> set[str]:
    return {amostra.paciente for amostra in conjunto}


# ==========================================
# Extração do identificador do paciente
# ==========================================
@pytest.mark.parametrize(
    "nome_arquivo, esperado",
    [
        ("AQK_image_0.tif", "AQK"),
        ("AEC_image_1234.tif", "AEC"),
        ("ABC_image_7.tiff", "ABC"),
    ],
)
def test_extrair_paciente_le_o_prefixo(nome_arquivo, esperado):
    assert _extrair_paciente(nome_arquivo) == esperado


# ==========================================
# O contrato que sustenta a metodologia
# ==========================================
def test_nenhum_paciente_cruza_conjuntos():
    """Nenhum paciente pode aparecer em mais de um conjunto."""
    divisoes = dividir_por_paciente(construir_amostras())

    treino = pacientes_de(divisoes["treino"])
    validacao = pacientes_de(divisoes["validacao"])
    teste = pacientes_de(divisoes["teste"])

    assert not treino & validacao, "pacientes compartilhados entre treino e validação"
    assert not treino & teste, "pacientes compartilhados entre treino e teste"
    assert not validacao & teste, "pacientes compartilhados entre validação e teste"


def test_divisao_preserva_todas_as_amostras():
    """Nenhuma imagem pode ser perdida nem duplicada na divisão."""
    amostras = construir_amostras()
    divisoes = dividir_por_paciente(amostras)

    total = sum(len(conjunto) for conjunto in divisoes.values())
    assert total == len(amostras)

    caminhos = [a.caminho for conjunto in divisoes.values() for a in conjunto]
    assert len(set(caminhos)) == len(amostras), "há amostras duplicadas entre conjuntos"


def test_divisao_e_reprodutivel_com_a_mesma_semente():
    amostras = construir_amostras()
    primeira = dividir_por_paciente(amostras, semente=42)
    segunda = dividir_por_paciente(amostras, semente=42)

    for nome in ("treino", "validacao", "teste"):
        assert pacientes_de(primeira[nome]) == pacientes_de(segunda[nome])


def test_sementes_diferentes_produzem_divisoes_diferentes():
    amostras = construir_amostras()
    uma = dividir_por_paciente(amostras, semente=42)
    outra = dividir_por_paciente(amostras, semente=7)

    assert pacientes_de(uma["teste"]) != pacientes_de(outra["teste"])


def test_ambas_as_classes_aparecem_em_todos_os_conjuntos():
    """A divisão é estratificada: nenhum conjunto pode ficar com uma classe só."""
    divisoes = dividir_por_paciente(construir_amostras())

    for nome, conjunto in divisoes.items():
        rotulos = {amostra.rotulo for amostra in conjunto}
        assert rotulos == {0, 1}, f"o conjunto '{nome}' não tem as duas classes"


def test_proporcoes_invalidas_sao_rejeitadas():
    with pytest.raises(ValueError, match="devem somar"):
        dividir_por_paciente(construir_amostras(), proporcoes=(0.5, 0.3, 0.5))


# ==========================================
# O contraste que justifica a metodologia
# ==========================================
def test_divisao_aleatoria_vaza_pacientes():
    """Demonstra o problema que `dividir_por_paciente` resolve.

    Este teste afirma o comportamento *indesejado* de propósito: com divisão por
    imagem, o mesmo paciente cai em treino e teste ao mesmo tempo. É a diferença
    que o experimento de comparação do TCC quantifica.
    """
    divisoes = dividir_aleatorio(construir_amostras())

    compartilhados = pacientes_de(divisoes["treino"]) & pacientes_de(divisoes["teste"])
    assert compartilhados, (
        "esperava-se vazamento na divisão aleatória; se este teste falhar, "
        "a comparação de vazamento do TCC perde o contraste"
    )


# ==========================================
# Pesos de classe
# ==========================================
def test_pesos_de_classe_compensam_o_desbalanceamento():
    """A classe minoritária tem de receber peso maior."""
    amostras = [Amostra(Path(f"A_image_{i}.tif"), 0, "A") for i in range(25)]
    amostras += [Amostra(Path(f"B_image_{i}.tif"), 1, "B") for i in range(75)]

    pesos = calcular_pesos_classes(amostras)

    assert pesos[0] > pesos[1], "a classe minoritária deveria pesar mais"
    # 100 / (2 * 25) = 2,0 para a minoritária; 100 / (2 * 75) ≈ 0,667 para a outra.
    assert pesos[0] == pytest.approx(2.0)
    assert pesos[1] == pytest.approx(2.0 / 3.0)
