# Base OpenDSS para o trabalho de IC: amostragem, tamanho da base e viés

Escrito em 2026-11-14, por volta das 18h45 (hora do `date` da máquina). Implementa a seção 5 de `docs/ESPEC_BASE_CASOS_2026-11-14.md`.

Todo número abaixo foi medido nesta sessão. A fonte de cada um está entre colchetes: um log em `work/logs/`, um `.meta.json` ou um CSV em `experiments/ic/demo/`.

Marcas usadas no texto:
- **MEDIDO**: saiu de execução real.
- **INFERIDO**: é conta ou interpretação feita sobre números medidos.
- **DECLARADO**: é o que um documento diz, sem conferência nesta sessão.

As escolhas de desenho experimental estão em `experiments/ic/config.yaml`, marcadas com `# [A DECIDIR pelo grupo]` (política PAE). Os valores ali são padrões para a mecânica funcionar, não são recomendação.

> **Ressalva de tempo.** A máquina (12 CPUs) estava compartilhada com outros agentes rodando ao mesmo tempo. Por isso os tempos por amostra variam entre execuções:
> - de 7,3 ms, medido com `time.perf_counter` em 100 cenários às 18:26 [`work/logs/ic_prof_eval.log`; o cProfile do mesmo log é de outros 200 cenários]. A carga da máquina nesse momento não foi registrada: o `experiments/pae/README.md` declara cerca de 53 % de CPU às 18:26, também sem log;
> - a 21,6 ms em execução serial com a máquina carregada [`p1_stratified_k_n1500_s1.meta.json`].
>
> As extrapolações abaixo dão as duas pontas.

## 0. Como usar

```bash
# dataset (n sem teto; retomável por blocos; --workers N processos)
python -m src.ic.dataset --config experiments/ic/config.yaml --strategy stratified_k --n 1500 --seed 1 --out experiments/ic/demo/p1_stratified_k_n1500_s1.csv
python -m src.ic.dataset ... --strategy bernoulli --n 1500 --seed 2 --out ...
python -m src.ic.dataset ... --strategy nsga2_history --seed 3 --workers 4 --out experiments/ic/demo/p1_nsga2_history_s3.csv
python -m src.ic.dataset ... --strategy exhaustive [--n 4096] --workers 8 --block-size 1024 --out ... [--resume]
python -m src.ic.dataset ... --strategy stratified_k --plants P2 --n 6000 --seed 203 --workers 6 --out ...
# validação Edit x recompilação e Scc analítico x faultstudy
python -m src.ic.dataset --config experiments/ic/config.yaml --validate 40 --seed 7 --out work/ic/validate_P1 --plants P1
# curva de aprendizagem + viés + validação externa
python -m src.ic.learning_curve --config experiments/ic/config.yaml \
  --data experiments/ic/demo/p1_stratified_k_n1500_s1.csv experiments/ic/demo/p1_bernoulli_n1500_s2.csv \
  --test-data experiments/ic/demo/test_p1_stratified_k_n1000_s101.csv experiments/ic/demo/test_p1_bernoulli_n1000_s102.csv \
  --external experiments/ic/demo/p1_nsga2_history_s3.csv --compare experiments/ic/demo/p1_history_Random_s4.csv \
  --out experiments/ic/demo/learning_curve_P1
# comparar plantas com o mesmo n_max e referencia independente (correcao IC-V1)
python -m src.ic.learning_curve --config experiments/ic/config.yaml \
  --data experiments/ic/demo/p2_stratified_k_n1500_s201.csv --test-data experiments/ic/demo/test_p2_stratified_k_n1000_s202.csv \
  --reference independent --reference-data experiments/ic/demo/p2_stratified_k_n6000_s203.csv \
  --out experiments/ic/demo/learning_curve_P2_V1_refindep
python -m src.ic.learning_curve ... --data experiments/ic/demo/p2_stratified_k_n6000_s203.csv --n-max-cap 1500 --out ...
```

**Saída.** O CSV traz as colunas da especificação, nesta ordem:
- as 7 features: `Pprev_kW`, `Scc_MVA`, `Pn_kW`, `Ip_In`, `cosphi_p`, `eta`, `C_N1`;
- `bits` e `n_on`;
- os alvos contínuos: `v_min_inrush`, `v_max_post`, `tr_load_nom`, `tr_load_af`;
- o rótulo `y`;
- a proveniência: `strategy`, `seed`, `sample_id`.

Depois vêm as colunas extras: `plant`, `target_motor_id`, `contingency`, `isc3_kA`, `static_load_mult`, `pf_run`, `converged`, `v_min_label`, `tr_load_inrush_af`, `y_sepoc`, `f3`, `scc_method`, `eval_ms` e, só na importação do NSGA-II, as colunas `hist_*` (lista completada em 2026-11-15; conferida contra os CSVs). Cada CSV vem com um `<nome>.meta.json`, que guarda a configuração, as versões, os tempos, o balanço de classes e o plano. Os blocos ficam em `<nome>.blocks/` e são a unidade de retomada do `--resume`.

### Definições adotadas (a do grupo de IC pode ser outra)

- **`Pprev_kW`**: potência ativa total que sai da fonte no instantâneo PRE-ENERG, com o alvo desligado. **Inclui as perdas.**
  - Com P1, N-1 e todas as máquinas ligadas, dá 13 292,7 kW (MEDIDO) [`work/ic/validate_P1/validate_summary.json`].
  - O artigo de IC reporta 12,61 MW para o «caso base crítico» (DECLARADO, seção IV-C).
  - A diferença pode vir de outra definição: soma de potência mecânica, sem perdas, ou outra base. **Isso precisa ser alinhado com o grupo.**
- **`Scc_MVA`**: Scc trifásico na barra do alvo, calculado como o Thevenin da rede, sem contribuição de cargas.
  - Na P1 é analítico, do `scenario_core.scc_mva_analytic`.
  - Quando o analítico não se aplica (P2, que tem gerador e interligação), roda um faultstudy no próprio engine, logo após recompilar o master. A definição é a mesma de `scenario_core.scc_mva_faultstudy`.
- **`tr_load_nom` / `tr_load_af`**: o maior valor, entre os trafos em serviço, de max(S em PRE-ENERG, PRE-DESL, POS-DESL) dividido pelo kVA AN (nominal) ou pelo kVA AF.
- **`y`**: padrão do artigo de IC (seção II-C), configurável no bloco `label:`. Vale 1 quando valem as três condições:
  - Vmin > 0,85 pu;
  - `v_max_post` ≤ 1,08 pu;
  - `tr_load_nom` ≤ 1,0, com convergência do fluxo.
- **`y_sepoc`**: viabilidade da formulação restrita (g1, g2 e g3 no AF), registrada para comparação.

### Validações (MEDIDO)

**Edit × recompilação na P1** [`work/logs/ic_validate_P1.log`]. Foram 40 amostras LHS mais 3 casos extremos, avaliadas pelo caminho Edit em ordem embaralhada:

| Grandeza | Diferença máxima, Edit menos recompilação completa |
|---|---|
| Vmin | 3,0e-13 pu |
| v_max_post | 0 |
| `tr_load_nom` | 4,4e-16 |
| Pprev | 0,039 kW (início a quente do fluxo, tolerância 1e-6) |

- Rótulos e convergência: 0 divergências.
- Custo por amostra: Edit 22 ms; reconstrução do .dss mais compilação, 137 ms.

**Scc na P1.**
- Analítico × faultstudy do OpenDSS (subprocesso): diferença máxima de 2,6e-6 MVA nas 43 amostras.
- **P1 em N-1 com isc3 = 15 kA: 74,205 MVA** (analítico 74,20524; faultstudy 74,20524). O artigo de IC diz 74,20 MVA.

**P2** [`work/logs/ic_validate_P2b.log`], 11 amostras:
- fluxos idênticos: diferença de 0 em Vmin, Pprev e carregamentos;
- Scc no engine × faultstudy em subprocesso: diferença máxima de 0,0012 MVA;
- P2 em N-1: 77,348 MVA.

Antes da correção, o faultstudy no engine rodava depois dos fluxos e dava até 1,18 MVA de diferença, porque o resultado mudava com o estado do gerador. O 1,18 MVA está em `work/logs/ic_validate_P2.log` (1,1782 MVA; 78,474 no engine contra 77,348 no subprocesso); a dependência do gerador está em `work/logs/ic_check_p2_scc.log` (77,348 MVA com G e 70,555 MVA sem G). Por isso o caminho atual recompila antes do faultstudy.

**Importação do NSGA-II.** Vmin recalculado × Vmin do `history_all.csv`: diferença máxima de 5,9e-13 em 4126 cenários. `y_sepoc` coincide com `feasible` do histórico em 100 %.

**Testes.** `python -m pytest -q tests/test_ic.py`: 10 passaram, em cerca de 7 a 11 s. Depois da correção IC-V1: 12 passaram em 12,75 s [`work/logs/iccorr_pytest_ic.log`]; as mutações da regra nova estão em `work/logs/iccorr_mutation_*.log`.

Teste sob mutação [`work/logs/ic_mutation_*.log`]: cada uma das mutações abaixo faz o teste falhar.

| Mutação | Erro produzido em Vmin |
|---|---|
| Sem o Edit do isc3 | 9,3e-3 pu |
| Sem o Edit dos trafos | 0,10 pu |
| Sem o Edit da carga do alvo | 0,034 pu |
| Rótulo com ≥ em vez de > | o teste de rótulo falha |

### Demo (`experiments/ic/demo/`)

As bases da P1 do demo usam N-0/N-1 em 50/50 por LHS, isc3 entre 10 e 20 kA, alvo por LHS (média ±2σ da Tabela II), carga entre 0,8 e 1,2 e nenhuma não convergência.

| Arquivo | n | y=1 / y=0 | Tempo médio por amostra (`eval_ms`) |
|---|---|---|---|
| `p1_stratified_k_n1500_s1` | 1500 | 784 / 716 | 21,6 ms (1 processo) |
| `p1_bernoulli_n1500_s2` | 1500 | 762 / 738 | 12,6 ms (1 processo) |
| `p1_nsga2_history_s3`: todos os 4126 bits únicos do NSGA2 em 8000 avaliações, **sem corte** | 4126 | 1216 / 2910 | 39,8 ms (4 processos) |
| `p1_history_Random_s4`: algoritmo Random do mesmo histórico, como referência de viés | 7941 | 46 / 7895 | 24,6 ms (4 processos) |
| teste fixo: `test_p1_stratified_k_n1000_s101` + `test_p1_bernoulli_n1000_s102` | 2000 | 1024 / 976 | — |
| P2: `p2_stratified_k_n1500_s201` (regerada em 2026-11-15) | 1500 | 910 / 590 | 42,4 ms (4 processos); antes da regeração, 48,4 ms |
| P2: `p2_stratified_k_n6000_s203` (regerada em 2026-11-15) | 6000 | 3579 / 2421 | 52,4 ms (6 processos); antes da regeração, 58,4 ms |
| P2, teste fixo: `test_p2_stratified_k_n1000_s202` (regerada em 2026-11-15) | 1000 | 599 / 401 | 47,3 ms (4 processos); antes da regeração, 48,2 ms |
| `p1_stratified_k_n6000_s103` (pool independente de referência, correção IC-V1) | 6000 | 3121 / 2879 | 20,3 ms (4 processos) |

Estatística do `stratified_k` P1 comparada à Tabela II do artigo de IC:

| Variável | Esta base (média / desvio) | Artigo de IC |
|---|---|---|
| Scc | 97,9 / 25,0 MVA | 98,63 / 25,33 |
| Pprev | 8441 / 3017 kW | 8901,5 / 2273 |
| Pn | 995,9 / 179,5 kW | 995,86 / 155,41 |

---

## (1) O que é a «taxa de amostragem» neste problema

A base é **quase estática**. Cada cenário vira 4 fluxos de potência instantâneos:
- PRE-ENERG;
- INRUSH, com o rotor bloqueado modelado como impedância;
- PRE-DESL;
- POS-DESL.

**Não há passo de tempo.** Logo, «taxa de amostragem» aqui só pode significar a **densidade de amostragem do espaço de cenários**: quantos pontos (condição + vetor liga/desliga) se avaliam por região desse espaço. Isso inclui quantos caem perto da fronteira Vmin = 0,85. Aumentar a «taxa» é aumentar n, ou concentrar amostras onde o classificador erra.

A pergunta «quanto basta?» se responde pela curva de aprendizagem da seção (3), não por um teorema de amostragem.

Se o professor se referia a **amostragem temporal**, isto é, à forma de onda do afundamento durante a partida, este modelo não a tem. Seria preciso simulação dinâmica, por exemplo:
- o modo `dynamics` do OpenDSS, com máquina de indução (`IndMach012`) e passo de integração;
- ou um programa EMT.

Nesse caso a taxa seria o passo de integração ou de registro (ms). O custo por cenário passaria de cerca de 10 ms para segundos, e o rótulo passaria a depender do tempo de aceleração. É outro estudo (INFERIDO).

## (2) Como usar os cenários do NSGA-II

**O que eles são** [`p1_nsga2_history_s3.meta.json`]:
- os 4126 vetores únicos que o NSGA2 visitou nas 8000 avaliações do `history_all.csv`;
- **só variam o liga/desliga**: a condição é fixa (P1, N-1, isc3 = 15 kA, alvo de 1250 kW, carga 1,0). Logo, Scc e as features do motor são constantes, com Scc = 74,205 MVA em todos.

**Viés medido** [`learning_curve_P1/bias_by_strategy.csv`, δ = 0,01 pu]:

| Conjunto | Fração com \|Vmin − 0,85\| < 0,01 | Pprev: média e quantis p05–p95 | Fração viável | Liga/desliga |
|---|---|---|---|---|
| NSGA2 | **65,6 %** (2707 de 4126) | 6050 kW, de 4459 a 9072 | 29,5 % | condição fixa |
| Random, mesma condição fixa | 4,1 % | 8423 kW, de 6285 a 10 565 | 0,6 % | uniforme |
| `stratified_k` | 10,4 % | 8441 kW, de 3802 a 13 059 | 52,3 % | — |
| `bernoulli` | 15,2 % | 8387 kW, de 6073 a 10 552 | 50,8 % | — |

O otimizador concentra amostras na fronteira e em Pprev baixo. A Figura `learning_curve_P1/bias_pprev_vmin.png` mostra isso.

**Recomendação de uso** (mecânica; a decisão é do grupo):
1. **Validação externa**, como o artigo de IC já faz. Ela mede a qualidade junto à fronteira.
   - Treinado em `stratified_k` (n = 1500) e avaliado nos 4126 do NSGA2: acurácia 0,940, recall da classe 0 = 1,000, AUC 0,999.
   - No teste fixo, o mesmo modelo dá acurácia 0,986 [`learning_curve_P1/external_validation.csv`].
   - A queda de acurácia vem de casos viáveis junto à fronteira classificados como críticos (INFERIDO a partir de recall0 = 1).
2. **Enriquecimento de fronteira** do treino, por exemplo active learning ou sobreamostragem perto de Vmin = 0,85. **Nunca misturar ao conjunto de teste**: o teste ficaria artificialmente difícil, ou fácil, e não representaria a operação.
3. **Cuidado com vazamento.** Os mesmos bits (e as mesmas features) não podem aparecer no treino e no teste. A rotina conta vetores de features idênticos entre o conjunto externo e treino/teste: 0 e 0 nesta demo, porque as condições do demo são contínuas por LHS.
   - Se o grupo gerar treino na condição fixa do artigo, deduplicar por `bits` e condição antes de particionar.
   - Particionar **antes** de qualquer seleção de atributos ou padronização; o pipeline já faz isso.
4. O histórico não representa N-0, outras Scc nem outros motores. **Sozinho, não serve como base de treino.**

## (3) Como a curva de aprendizagem define o n necessário

**Procedimento** (`src/ic/learning_curve.py`):
- o conjunto de teste é **fixo** e independente: 2000 cenários P1 e 1000 cenários P2;
- n de treino percorre uma grade geométrica: 100, 200, 400, ... até o disponível;
- para cada n, 20 reamostragens estratificadas por y;
- o modelo é um pipeline de padronização mais regressão logística L2, com C = 1,0 como no artigo de IC.

**Regra objetiva.** n suficiente é o menor n da grade a partir do qual, para todo n maior, valem as duas condições:
- |média(n) − média(n_max)| ≤ ε, com ε = 0,005;
- a largura do intervalo de 95 % das reamostragens é ≤ 0,02.

Os dois limiares são configuráveis, e a escolha deles é do grupo. Se só o próprio n_max satisfaz a regra, ela devolve «não estabelecido».

> **Correção IC-V1 (2026-11-14, por volta das 19h05).** Na regra acima, a referência é a média no maior n, que é **um único ajuste no pool inteiro**: 1 reamostragem, IC de largura 0 [`src/ic/learning_curve.py`, `reps = 1 if n >= len(pool)`]. Consequências:
> - **o n\* é relativo ao n_max** (o tamanho da base), não é uma propriedade da planta. O resultado agora declara isso (`relative_to_n_max`, `n_max`, `reference_kind`, `frac_of_pool_at_n_star`);
> - perto de N, as subamostras sem reposição se sobrepõem (fração esperada de pontos comuns entre duas subamostras = n/N; coluna `frac_of_pool` no `curve_summary.csv`), o que encolhe a largura do IC e favorece a regra (INFERIDO).
>
> Mecânica nova, com padrões `# [A DECIDIR pelo grupo]` em `RULE_DEFAULTS` de `src/ic/learning_curve.py` e sobreponíveis pela CLI. Não foram para o `config.yaml` porque `dataset.config_digest` inclui o yaml inteiro e chaves novas invalidariam o `--resume` das bases já geradas.
>
> | Parâmetro (CLI) | Padrão | O que faz |
> |---|---|---|
> | `reference` (`--reference`) | `n_max` (antigo) | `independent`: referência = média de ajustes num pool independente da mesma estratégia (`--reference-data`); o próprio n_max entra no teste do platô |
> | `reference_n`, `reference_reps` (`--reference-n`, `--reference-reps`) | `None`, 20 | tamanho e número de ajustes da referência independente (`None` = 1 ajuste no pool de referência inteiro) |
> | `n_max_cap` (`--n-max-cap`) | sem teto | teto comum do pool (subamostra estratificada de semente fixa); **para comparar plantas, usar o mesmo n_max** |
> | `max_pool_frac` (`--max-pool-frac`) | 1,0 (antigo) | só entram na regra os n com n/N ≤ fração |
>
> Um ponto com uma só reamostragem nunca vira n\*. Com os padrões, a regra reproduz os n\* gravados antes da correção: 15 de 15 [`work/logs/iccorr_compat_check.log`].

**Resultados P1** [`learning_curve_P1/learning_curve_result.json`]:

| Treino | Acurácia (platô) | Recall classe 0 | AUC |
|---|---|---|---|
| `stratified_k` | n* = 800 (0,986) | n* = 800 (0,992) | n* = 100 (0,9995) |
| `bernoulli` | **não estabelecido até 1500** (0,960) | **não estabelecido** (0,956) | n* = 800 |
| pool (3000) | 800 | 1600 | 100 |

Com o mesmo n, a amostragem `bernoulli` aprende pior. Ela quase não visita Pprev extremos: p05–p95 entre 6073 e 10 552 kW, contra 3802 a 13 059 kW do `stratified_k`. Por isso extrapola mal nessas regiões do teste. **A estratégia importa tanto quanto n.**

**Caso complexo (P2)**, com barramento seccionado, gerador e BT [`learning_curve_P2_n6000/learning_curve_result.json`], usando treino `stratified_k` até 6000:

| Métrica | n* | Valor |
|---|---|---|
| Acurácia | 1600 | 0,954 |
| AUC | 200 | 0,993 |
| Recall da classe 0 | **não estabelecido até 6000** | sobe de 0,909 (n=800) para 0,921 (n=3200) e 0,925 (n=6000); IC de 0,023 em n=3200 |

**Comparação com a P1 (corrigida em IC-V1, 2026-11-14 por volta das 19h05).** A versão anterior deste README dizia que a P2 precisa de mais amostras, com acurácia n\* = 1600 contra 800 na P1. **Essa frase está superada**: ela comparava a P2 com n_max = 6000 e a P1 com n_max = 1500. Medido com o mesmo n_max (comandos e logs abaixo; teste fixo P1 de 2000 e P2 de 1000, como antes):

| Comparação | Referência do platô | P1: n\* da acurácia | P2: n\* da acurácia |
|---|---|---|---|
| n_max = 1500 (pools s1 e s201) | n_max (1 ajuste) | 800 | 800 |
| n_max = 1500 | independente: 1 ajuste num pool de 6000 (s103 e s203) | 800 (ref. 0,987) | 800 (ref. 0,954) |
| n_max = 1500, subamostra do pool P2 de 6000 (`--n-max-cap 1500`) | n_max | — | 400 |
| n_max = 6000 (s103 e s203) | n_max | 800 | 1600 |
| n_max = 6000, `--max-pool-frac 0.5` | n_max | 800 | 1600 |

Fontes: `learning_curve_P1`, `learning_curve_P2`, `learning_curve_P1_V1_refindep`, `learning_curve_P2_V1_refindep`, `learning_curve_P2_n6000_V1_cap1500`, `learning_curve_P1_n6000_V1`, `learning_curve_P2_n6000`, `learning_curve_P1_n6000_V1_frac05` e `learning_curve_P2_n6000_V1_frac05` em `experiments/ic/demo/`; resumo em `work/logs/iccorr_summary.log`. O pool P1 de 6000 (`p1_stratified_k_n6000_s103.csv`, 3121 / 2879, 0 não convergências, 20,3 ms por amostra em 4 processos) foi gerado nesta correção [`work/logs/iccorr_ds_p1_n6000_s103.log`].

O que a tabela mostra:
- com bases iguais de 1500, a acurácia dá n\* = 800 nas duas plantas. Com 6000, dá 800 na P1 e 1600 na P2;
- logo, **o n\* da acurácia não separa as plantas de forma robusta**: depende do n_max e da base sorteada. Duas bases P2 de 1500 dão 400 e 800 (MEDIDO);
- o recall0 também depende do n_max: na P1, n\* = 800 com 1500 e 3200 com 6000. Na P2, não se estabelece em nenhum dos casos;
- **o que difere de forma consistente é o platô**. Na referência de 6000, a acurácia fica em 0,954 na P2 e 0,987 na P1, e o recall0 em 0,925 e 0,993. Com 1500: 0,951 × 0,986 e 0,915 × 0,992.

Interpretação (INFERIDO): a fronteira na P2 não é linear nas 7 features. O afundamento depende da seção, do gerador e da barra de BT, que as 7 features não descrevem. Para o caso complexo, «taxa maior» ajuda até um ponto. Depois disso o ganho vem de:
- features por seção;
- um modelo não linear;
- amostragem dirigida à fronteira.

A curva mostra onde acaba o ganho por n. O procedimento é o mesmo para qualquer planta: gerar com `--plants P2 --n N` e rodar a curva com teste fixo.

## (4) Sugestão do dono: alternar liga/desliga (enumeração exaustiva)

**O que a enumeração dá.** Com a condição fixada (planta, contingência, isc3, alvo, carga), a estratégia `exhaustive` avalia **todos** os 2^n vetores liga/desliga:
- P1: 2^19 = 524 288;
- P2: 2^21 = 2 097 152.

Isso fornece a verdade completa daquela condição: a fronteira de viabilidade exata, o conjunto viável inteiro e a frente de Pareto exata do problema do SEPOC na condição do artigo. Serve de referência para medir o viés do NSGA-II e o erro de qualquer amostra.

**O que ela não dá.** Uma única condição fixa tem Scc, contingência e motor constantes. Para o classificador de IC, que precisa variar essas features, a base exaustiva **não substitui** a amostragem do espaço contínuo. Seria preciso repetir 2^19 vezes por condição.

**Custo medido e extrapolado (P1)** [`work/logs/ic_exh_w1.log`, `work/logs/ic_exh_w8.log`, `work/logs/ic_prof_eval.log`]. A implementação é por blocos, retomável (`--resume`) e paralela (`--workers`).

| Configuração | Medido | Extrapolação para 2^19 |
|---|---|---|
| 1 processo, máquina carregada | 4096 índices em 88,4 s = 46,3 cenários/s | **≈ 3,1 h** |
| 8 processos, máquina carregada | 16 384 índices em 97,1 s = 168,7 cenários/s | **≈ 52 min** |
| 1 processo, carga da máquina não registrada | 7,3 ms por cenário (`time.perf_counter`, 100 cenários) | ≈ 64 min |

- **P2, 1 processo** [`work/logs/ic_exh_p2_w1.log`]: 2048 índices em 30,9 s = 66,4 cenários/s. Para 2^21 isso dá ≈ 8,8 h. Com 8 processos, a mesma razão da P1 (×3,6) daria cerca de 2,4 h (INFERIDO).
- **Espaço em disco**: 275 bytes por linha de CSV (MEDIDO em `work/ic/exh/p1_exhaustive_first16384_w8.csv`), o que dá cerca de 144 MB por enumeração da P1, mais os blocos (INFERIDO).
- **Onde está o custo**: o tempo por cenário é dominado pelo `iterrows` do pandas dentro de `powerflow_runner.collect_snapshot` (núcleo, não editado aqui) [`work/logs/ic_prof_eval.log`].
- **Corte declarado**: as enumerações medidas são **parciais**. Cobrem os primeiros índices, com os bits mais significativos, os motores pequenos, em 0, e **não** representam a distribuição do espaço completo. Não as use como amostra.

---

## Limitações e o que ficou de fora

- A P2 (`data_plant2/`) é do agente planta2, e seus valores são `[A DECIDIR pelo grupo]`. Os números da P2 valem para a versão do arquivo lida em 2026-11-14 por volta das 18h30. Com mais precisão (MEDIDO pelas datas de modificação): as bases P2 do demo foram gravadas entre 18:34:15 (s201) e 18:43:06 (s203), e o `data_plant2/system_config.yaml` mudou às 18:34:29 (correção P2-V1, que retirou o bloco `source:` ignorado pelo código), entre a s201 e as outras. Segundo os logs do agente planta2 (`work/logs/plant2_P2V1_antes.log` e `plant2_P2V1_depois.log`), o master do caso base, o Scc (77,348 MVA) e o Vmin com todas desligadas ficaram iguais antes e depois. A validação da P2 (`ic_validate_P2b.log`, 18:24:58) é anterior a essa mudança. **Atualização de 2026-11-15:** as 3 bases P2 do demo foram regeradas entre 13:00 e 13:14 de 15/11, todas depois dessa mudança e da correção do `normamps`. Os horários de 14/11 citados acima valem para as bases antigas, que estão em `_quarentena/2026-11-15_ic_P2_f3_TIE400A/`. Na comparação, só mudaram a coluna `f3` e o tempo `eval_ms`. A `s203` foi concluída com `--resume`: 23 de 24 blocos vêm da primeira execução, e o `.meta.json` registra só a retomada.
- Na P2, `Pprev_kW` (vem de `circuit.total_power`) é a **soma das duas fontes** e **exclui o gerador local** [`work/logs/ic_check_p2_pprev.log`], com todas as máquinas ligadas:

  | Contingência | total_power | Vsource.source | SRC_B | Gerador G |
  |---|---|---|---|---|
  | N-0 | 11 954,7 kW | 6114,2 kW | 5840,5 kW | 1500,0 kW |
  | N-1 | 12 029,3 kW | 12 029,3 kW | 0 | 1500,0 kW |

  **Correção de 2026-11-15:** a coluna do gerador G dizia 1000,0 kW. O valor estava errado: o `work/ic/check_p2_pprev.py` somava `powers[0:n:2]` com `n = len(powers)//2`. Isso cobre o terminal 1 inteiro do Vsource, que tem 2 terminais, mas só 2 das 3 fases do Generator, que tem 1 terminal e 4 condutores. Medido com a soma correta, G entrega 1500,0 kW em N-0 e em N-1 (MEDIDO, `work/ic/check_p2_gen_power.py`, log `work/logs/ic_check_p2_gen_power.log`). As colunas das fontes não mudam.

  Logo, na P2 o Pprev não é a carga total: é a carga mais as perdas, menos os 1500 kW de G. Precisa de definição do grupo.
- **[SUPERADO em 2026-11-15, por volta das 13h15: as 3 bases P2 foram regeradas com o `normamps` corrigido. Só mudaram a coluna `f3` e o tempo de execução `eval_ms`; todas as demais colunas, inclusive `y` e as 7 features, saíram idênticas (MEDIDO, `work/logs/ic_regen_compare_*.log`, que ignora `eval_ms` de propósito). Detalhes em `experiments/ic/demo/NOTA_f3_P2_desatualizado.txt`; as bases antigas estão em `_quarentena/2026-11-15_ic_P2_f3_TIE400A/`.]** Texto original: coluna `f3` das bases da P2 desatualizada (2026-11-14, por volta das 19h45). As bases P2 do demo (`p2_stratified_k_n1500_s201`, `p2_stratified_k_n6000_s203` e `test_p2_stratified_k_n1000_s202`) foram geradas quando o disjuntor TIE_AB saía no `.dss` sem `normamps`, e o OpenDSS assumia 400 A. Corrigido em `src/dss_builder.py`, o limite passou a ser `ampacity_A = 2000`. A sobrecarga artificial da interligação entrava **só** na contagem f3. Logo, **apenas a coluna auxiliar `f3`** dessas bases está desatualizada. O `normamps` não entra no fluxo de potência, então as 7 features, `y`, `y_sepoc`, `v_min_inrush`, `v_max_post` e os carregamentos dos trafos não mudam (INFERIDO do código; a viabilidade e o f4 do caso base ficaram iguais antes e depois, MEDIDO em `work/logs/ff_check_tie.log`). As curvas de aprendizagem usam `y` e as features, então continuam válidas. Para ter o `f3` correto, regerar as bases P2. As bases da P1 não mudam, porque a P1 não tem linhas.
- O modo `target.mode: existing` (alvo sorteado entre motores existentes, com recompilação por alvo) foi implementado, mas **não** foi exercitado no demo nem nos testes.
- `test_smoke.py` tem 2 falhas preexistentes, por `UnicodeEncodeError` de caminho com acento. Elas não vêm deste módulo: já constavam de `work/logs/pytest_all_F1.log`.
- Os tempos dependem da carga da máquina, compartilhada com outros agentes; veja a ressalva no topo.
