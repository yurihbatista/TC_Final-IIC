# Código gerador da base OpenDSS do trabalho de IC

Esta pasta regera as bases de `../dados/` e as curvas de `../curvas_de_aprendizagem/`. Ela é uma extração mínima do repositório do estudo SEPOC: só os módulos de que o gerador depende, as duas plantas, a configuração, o histórico do NSGA-II e os testes do gerador.

## 1. Conteúdo

| Caminho | O que é |
|---|---|
| `src/ic/dataset.py` | Gerador das bases: amostragem, simulação e rótulo |
| `src/ic/learning_curve.py` | Curva de aprendizagem, viés e validação externa |
| `src/scenario_core.py`, `src/powerflow_runner.py`, `src/opendss_interface.py`, `src/dss_builder.py`, `src/motor_model.py`, `src/protection_logic.py`, `src/config_loader.py` | Núcleo: monta o `.dss`, roda os 4 instantâneos no OpenDSS e calcula tensões, carregamentos e Scc |
| `data/` | Planta P1, a do artigo: barras, cargas, motores e `system_config.yaml` |
| `data_plant2/` | Planta P2, a complexa |
| `experiments/ic/config.yaml` | Configuração da geração: faixas, rótulo e curva. Igual a `../config_geracao_ic.yaml` |
| `results_sepoc/history_all.csv` | Histórico das avaliações do estudo SEPOC (NSGA-II, NSGA-III, UNSGA-III e busca aleatória). Fonte das bases `p1_nsga2_history_s3` e `p1_history_Random_s4` |
| `tests/test_ic.py`, `tests/conftest.py` | Testes do gerador |
| `conferir_reproducao.py` | Compara uma base regerada com a do pacote |
| `requirements_ic.txt` | Versões usadas na geração |

## 2. Instalação

Ambiente de origem: Windows 11 e Python 3.13.5. O `py-dss-interface` traz o motor do OpenDSS (`OpenDSSDirect.dll`) e roda no Windows. Em outros sistemas, não foi testado.

```bash
python -m pip install -r requirements_ic.txt
```

Rode todos os comandos **dentro desta pasta** (`codigo/`).

**Caminho com acento ou espaço.** O `py-dss-interface` só aceita caminhos ASCII. Se esta pasta estiver num caminho com acento, o código compila o OpenDSS numa cópia em `%TEMP%\nsga2vcb_dss\`. Para usar outra pasta, defina a variável de ambiente `NSGA2VCB_DSS_ROOT` com um caminho ASCII. Os arquivos de trabalho vão para `work/ic/`, criada na primeira execução.

## 3. Testes

```bash
python -m pytest -q tests/test_ic.py
```

## 4. Regerar as bases

Estes são os comandos que geraram cada base do pacote (campo `cmdline` dos `.meta.json`), com a saída em `saida/`. A semente fixa o plano de amostragem; `--workers` só muda a divisão em processos.

```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 1500 --seed 1 --out saida/p1_stratified_k_n1500_s1.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy bernoulli --n 1500 --seed 2 --out saida/p1_bernoulli_n1500_s2.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 6000 --seed 103 --workers 4 --plants P1 --out saida/p1_stratified_k_n6000_s103.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 1000 --seed 101 --workers 4 --out saida/test_p1_stratified_k_n1000_s101.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy bernoulli --n 1000 --seed 102 --workers 4 --out saida/test_p1_bernoulli_n1000_s102.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy nsga2_history --seed 3 --workers 4 --out saida/p1_nsga2_history_s3.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy nsga2_history --history-algorithms Random --seed 4 --workers 4 --out saida/p1_history_Random_s4.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 1500 --seed 201 --workers 4 --plants P2 --out saida/p2_stratified_k_n1500_s201.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 6000 --seed 203 --workers 6 --plants P2 --out saida/p2_stratified_k_n6000_s203.csv
```
```bash
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 1000 --seed 202 --workers 4 --plants P2 --out saida/test_p2_stratified_k_n1000_s202.csv
```

**Conferir contra a base do pacote:**

```bash
python conferir_reproducao.py saida/p1_stratified_k_n1500_s1.csv ../dados/P1_planta_do_artigo/p1_stratified_k_n1500_s1.csv
```

O resultado esperado é «REPRODUZIU». O script exige `bits` e `y` idênticos e, em cada coluna numérica, |dif| ≤ 1e-6 + 1e-5·|valor|.

O fluxo de potência converge com tolerância 1e-6 (`src/dss_builder.py`) e parte do estado anterior do motor OpenDSS («a quente»). Com vários processos, a ordem das amostras em cada processo muda, e o Pprev pode diferir em alguns milionésimos relativos.

Resultado medido ao regerar a partir deste pacote, numa pasta com acento no caminho (2026-11-15):
- `p1_stratified_k_n1500_s1` e `test_p2_stratified_k_n1000_s202`: diferença 0 em todas as colunas;
- `p1_nsga2_history_s3`, com 4 processos: `bits` e `y` idênticos; o Pprev diferiu em até 0,013 kW (2,3e-6 relativo) em 9 das 4126 linhas, e a Vmin em 3e-14.

**Retomada.** Cada base é gerada em blocos de 250 amostras, gravados em `<nome>.blocks/`. Se a execução for interrompida, repita o mesmo comando com `--resume`, e só os blocos que faltam serão calculados.

**Travamento conhecido.** Com vários processos (`--workers`), um processo já travou ao carregar a `OpenDSSDirect.dll` (deadlock na inicialização da DLL, dentro de `LoadLibrary`). O sintoma é ficar parado sem uso de CPU. Interrompa com Ctrl-C, encerre os processos `python` que sobrarem e repita o comando com `--resume`.

## 5. Gerar outras bases

- **Mais amostras ou outra semente:** mude `--n` e `--seed`. Não há teto para n.
- **Outra estratégia:**
  - `--strategy exhaustive` enumera todos os 2^19 = 524 288 vetores liga/desliga da P1 numa condição fixa, definida no bloco `exhaustive` do yaml. Na máquina de origem, com 8 processos, a extrapolação de uma enumeração parcial dá cerca de 1 h. É uma estimativa, não uma medição da enumeração completa.
  - `--strategy bernoulli` e `--strategy stratified_k` sorteiam o liga/desliga.
- **Outras faixas, rótulo ou proporção N-1:** edite uma cópia do `experiments/ic/config.yaml` e passe-a em `--config`. Os parâmetros marcados `[A DECIDIR pelo grupo]` são os de desenho experimental.
- **Validação do motor de simulação:** `python -m src.ic.dataset --config experiments/ic/config.yaml --validate 40 --seed 7 --out saida/validacao --plants P1` compara o caminho rápido (comandos `Edit` no circuito já compilado) com a recompilação completa.

## 6. Refazer uma curva de aprendizagem

Exemplo: a curva `learning_curve_P1` do pacote.

```bash
python -m src.ic.learning_curve --config experiments/ic/config.yaml --data ../dados/P1_planta_do_artigo/p1_stratified_k_n1500_s1.csv ../dados/P1_planta_do_artigo/p1_bernoulli_n1500_s2.csv --test-data ../dados/P1_planta_do_artigo/test_p1_stratified_k_n1000_s101.csv ../dados/P1_planta_do_artigo/test_p1_bernoulli_n1000_s102.csv --external ../dados/P1_cenarios_do_estudo_SEPOC/p1_nsga2_history_s3.csv --compare ../dados/P1_cenarios_do_estudo_SEPOC/p1_history_Random_s4.csv --out saida/learning_curve_P1
```

Outra opção é a referência independente, usada nas pastas `*_V1_refindep`:

```bash
python -m src.ic.learning_curve --config experiments/ic/config.yaml --data ../dados/P2_planta_complexa/p2_stratified_k_n1500_s201.csv --test-data ../dados/P2_planta_complexa/test_p2_stratified_k_n1000_s202.csv --reference independent --reference-data ../dados/P2_planta_complexa/p2_stratified_k_n6000_s203.csv --out saida/learning_curve_P2_V1_refindep
```

As opções `--n-max-cap` e `--max-pool-frac` geram as variantes `cap1500` e `frac05`. Veja `python -m src.ic.learning_curve --help`.

## 7. Limitações

- **Só o necessário para o IC.** A otimização NSGA-II/NSGA-III do estudo SEPOC e o experimento do PAE ficaram de fora.
- **Testes.** Os do gerador estão aqui; os do núcleo ficaram no repositório de origem.
- **Planta 2.** É ilustrativa, e seus parâmetros (`data_plant2/system_config.yaml`) são propostas marcadas `[A DECIDIR pelo grupo]`.
- **Sistemas operacionais.** Testado só no Windows 11 com as versões de `requirements_ic.txt`.
