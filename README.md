# Classificação Binária para Triagem Pré-Manobra de Partidas Diretas de Motores em Sistemas Industriais de Média Tensão

Repositório contendo os scripts em Python 3 (`scikit-learn`, `pandas`, `numpy`) utilizados para seleção de atributos por Informação Mútua, treinamento, validação cruzada, avaliação de limiares e geração de figuras, bem como os conjuntos de dados simulados no OpenDSS (`IC_dados_OpenDSS_2026-11-15`).

Trabalho submetido à disciplina de pós-graduação **EEE901 — Introdução à Inteligência Computacional** (PPGEE/UFMG) como requisito parcial para aprovação.

## Autores
* Cilene Cristina Nunes Faria
* Yuri Henriques Batista

## Execução
Instale as dependências e execute o script principal a partir da raiz do repositório:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python modelo_ml_pre_manobra.py
```
### Créditos da Base de Dados
Os scripts para geração da base de dados utilizada são de autoria de Silva (2026) [1] e foram disponibilizados por ele.

[1] L. F. Silva, “Investigação da sobretensão de manobra causada pela reignição de arco em disjuntores a vácuo,” Dissertação de Mestrado, Programa de Pós-Graduação em Engenharia Elétrica, CEFET-MG, Belo Horizonte, Brasil, 2026.
