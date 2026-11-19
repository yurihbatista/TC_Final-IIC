# Base OpenDSS para o trabalho de Inteligência Computacional

Pacote de dados montado em 2026-11-15. São cenários de partida de um motor grande numa planta industrial, simulados no OpenDSS e rotulados com o critério do artigo de IC (Classe 1 = viável, Classe 0 = crítico), com as mesmas 7 variáveis de entrada.

Marcas de procedência: **MEDIDO** = calculado a partir dos arquivos deste pacote ou de execução registrada; **INFERIDO** = interpretação ou conta sobre números medidos; **DECLARADO** = vem de um documento, sem conferência.

## 1. Conteúdo

| Pasta / arquivo | O que é |
|---|---|
| `dados/P1_planta_do_artigo/` | Planta do artigo SEPOC/IC (1 barramento de 4,16 kV, 2 trafos de 7,5/9 MVA, 19 máquinas além do alvo). Bases de treino e de teste. |
| `dados/P1_cenarios_do_estudo_SEPOC/` | Cenários liga/desliga visitados no estudo SEPOC: NSGA-II e, como referência, busca aleatória. Condição fixa: N-1, Scc = 74,2 MVA, motor de 1250 kW. |
| `dados/P2_planta_complexa/` | Segunda planta: duas fontes, barramento seccionado (MV_A–MV_B), nível de 0,48 kV e gerador local; 21 máquinas além do alvo. |
| `*.meta.json` (um por base) | Comando, semente, configuração efetiva, versões, tempos e balanço de classes de cada base. |
| `resumo_bases.csv` | Uma linha por base: n, classes, fração N-1 e mínimo, máximo e média de cada feature (MEDIDO). |
| `dicionario_de_dados.csv` | Descrição, papel e unidade de **todas** as colunas das bases em `dados/`. As colunas de `resumo_bases.csv`, `maquinas_bits_*.csv` e `curvas_de_aprendizagem/` não estão nele. A coluna `papel` indica quais **não** podem ser entrada do modelo. |
| `maquinas_bits_P1.csv`, `maquinas_bits_P2.csv` | Qual máquina corresponde a cada posição da coluna `bits`. |
| `curvas_de_aprendizagem/` | Curvas de aprendizagem, viés das estratégias e validação externa (seção 5). |
| `config_geracao_ic.yaml` | Configuração base da geração (faixas, rótulo, curva). É igual a `codigo/experiments/ic/config.yaml`. As bases da P2 foram geradas com `--plants P2` e a `p1_history_Random_s4` com `--history-algorithms Random`. Essas sobreposições estão em `cmdline` e `config` de cada `.meta.json`. |
| `codigo/` | **Código gerador**: regera todas as bases e curvas deste pacote. Guia em `codigo/LEIA-ME_CODIGO.md`, com instalação, testes, os 10 comandos de geração e a conferência da reprodução. |
| `referencia/README_amostragem_base_completa.md` | Documento técnico completo da base. Os caminhos `src/...`, `data/...` e `experiments/ic/...` citados nele estão em `codigo/`. Os de `work/logs/...` e `_quarentena/...` são do repositório de origem e **não** estão neste pacote. |
| `SHA256SUMS.txt` | Hash de cada arquivo, para conferir a integridade (seção 7). |

### Bases (MEDIDO, `resumo_bases.csv`)

| Arquivo | Uso sugerido | n | Classe 1 / Classe 0 | N-1 |
|---|---|---|---|---|
| `p1_stratified_k_n1500_s1` | treino | 1500 | 784 / 716 | 50 % |
| `p1_bernoulli_n1500_s2` | treino | 1500 | 762 / 738 | 50 % |
| `p1_stratified_k_n6000_s103` | treino / referência | 6000 | 3121 / 2879 | 50 % |
| `test_p1_stratified_k_n1000_s101` | teste fixo P1 (metade 1) | 1000 | 519 / 481 | 50 % |
| `test_p1_bernoulli_n1000_s102` | teste fixo P1 (metade 2) | 1000 | 505 / 495 | 50 % |
| `p1_nsga2_history_s3` | validação externa / viés | 4126 | 1216 / 2910 | 100 % |
| `p1_history_Random_s4` | comparação de viés | 7941 | 46 / 7895 | 100 % |
| `p2_stratified_k_n1500_s201` | treino P2 | 1500 | 910 / 590 | 50 % |
| `p2_stratified_k_n6000_s203` | treino / referência P2 | 6000 | 3579 / 2421 | 50 % |
| `test_p2_stratified_k_n1000_s202` | teste fixo P2 | 1000 | 599 / 401 | 50 % |

Todas as 31 567 amostras convergiram, e não há valor faltante nas features nem no rótulo (MEDIDO, colunas `nao_convergidos` e `nan_em_features_ou_y` de `resumo_bases.csv`).

**Teste fixo usado nas curvas** (MEDIDO, campo `test` dos `learning_curve_result.json`):
- P1: a **união** dos dois arquivos `test_p1_*`, com n = 2000 (1024 / 976);
- P2: só `test_p2_stratified_k_n1000_s202`, com n = 1000.

## 2. Como os dados foram gerados

- **Modelo.** Fluxo de potência quase estático no OpenDSS (py-dss-interface 2.3.0). Cada cenário é avaliado em 4 instantâneos:
  1. PRE-ENERG: motor alvo desligado;
  2. INRUSH: motor alvo em partida, com rotor bloqueado modelado como impedância;
  3. PRE-DESL: motor alvo em regime;
  4. POS-DESL: motor alvo desligado depois de operar.

  Não há série temporal (seção 5).
- **O que varia em cada amostra** (faixas em `config_geracao_ic.yaml`):
  - contingência N-0 ou N-1, em metade das amostras cada;
  - corrente de curto da fonte principal entre 10 e 20 kA, o que faz o Scc variar (na P2 a SRC_B fica fixa em 10 kA);
  - multiplicador das cargas estáticas entre 0,8 e 1,2;
  - parâmetros do motor alvo (Pn, Ip/In, cos φp, η) por hipercubo latino, na faixa média ± 2 desvios da Tabela II do artigo de IC;
  - estado liga/desliga das demais máquinas.
- **Estratégias do liga/desliga** (coluna `strategy`):
  - `stratified_k`: sorteia quantas máquinas ficam ligadas (k uniforme de 0 a n) e depois quais;
  - `bernoulli`: cada máquina ligada com probabilidade 0,5;
  - `nsga2_history`: os vetores do histórico do estudo SEPOC.
- **Rótulo** (coluna `y`). Vale 1 quando: o fluxo convergiu, E a tensão mínima na partida é > 0,85 pu, E a tensão máxima depois do desligamento é ≤ 1,08 pu, E o carregamento dos trafos em serviço é ≤ 1,0 da potência **nominal** (AN). Senão, vale 0.
- **Como o rótulo se comporta nestas bases** (MEDIDO):
  - a condição de sobretensão **nunca** fica ativa: `v_max_post` ≤ 1,000 pu nas 31 567 amostras;
  - na `p1_stratified_k_n1500_s1`, dos 716 críticos:
    - 231 têm Vmin > 0,85 pu e são críticos só pelo carregamento;
    - 464 violam a tensão e o carregamento;
    - 21 violam só a tensão;
  - na mesma base, as 434 amostras N-1 com Pprev > 7,5 MW são todas críticas: o trafo que sobra já está acima do nominal antes da partida.

  Isso ajuda a explicar o peso de Pprev e C_N1 nos modelos (INFERIDO).

## 3. Correspondência com o artigo de IC e diferenças

| Artigo de IC | Coluna |
|---|---|
| Pprev (kW) | `Pprev_kW` |
| Scc (MVA) | `Scc_MVA` |
| Pn (kW) | `Pn_kW` |
| Ip/In | `Ip_In` |
| cos φp | `cosphi_p` |
| η | `eta` |
| C_N-1 | `C_N1` |
| Classe (1 viável / 0 crítico) | `y` |

**Diferenças que o grupo precisa conhecer:**
1. **Pprev e o caso crítico.** Aqui, Pprev é a potência ativa que sai da(s) fonte(s) no instantâneo PRE-ENERG, **incluindo as perdas**. Na P2, é líquida da geração do gerador local (1500 kW).
   - No caso crítico (P1, N-1, todas as máquinas ligadas), esta base dá Pprev = 13 292,7 kW e Vmin = 0,755 pu (MEDIDO na base de origem).
   - O artigo de IC cita 12,61 MW e 0,73 pu (DECLARADO no artigo).
   - Como a Vmin também difere, a diferença não se explica só pela definição de Pprev: o modelo de simulação também difere (INFERIDO). Convém alinhar antes de comparar números com o artigo.
2. **Balanço de classes e distribuição.**
   - As bases `stratified_k`/`bernoulli` têm cerca de 50 % de cada classe na P1 e cerca de 40 % de críticos na P2. A base do artigo tem 70,38 % de críticos (DECLARADO no artigo).
   - Aqui, os críticos em N-0 são poucos: 131 de 750 (17,5 %) na `s1` e 481 de 3000 (16,0 %) na `s103` (MEDIDO).
   - **Para imitar o desenho do artigo** (800 = 563 críticos + 237 viáveis, partição 80/20 estratificada): subamostre da `p1_stratified_k_n6000_s103` por classe **e** por `C_N1`, e declare isso.
   - Subamostrar só por classe eleva a média de `C_N1` para cerca de 0,64. É o valor esperado de uma subamostra 563/237 da `s103`: 2398 dos 2879 críticos e 602 dos 3121 viáveis estão em N-1, o que dá 0,643. Uma subamostra aleatória deu 0,655 (MEDIDO). Na Tabela II do artigo, a média é 0,491 (DECLARADO).
   - As distribuições não são as mesmas, então não espere reproduzir exatamente os números do artigo.
3. **Faixa do Scc.** Na P1 vai de 67,2 a 134,5 MVA; na P2, de 74,0 a 151,9 MVA (MEDIDO, `resumo_bases.csv`).
   - O Scc da P2 inclui a contribuição do gerador local. Em N-1, fica cerca de 6,8 MVA acima do da P1 na mesma faixa de corrente de curto: 74,04–85,04 contra 67,25–78,25 MVA (MEDIDO).
   - Nas bases do estudo SEPOC, o Scc é fixo em 74,21 MVA.
4. **Colunas que NÃO podem ser entradas do modelo.** O artigo de IC usa só as 7 features. As colunas abaixo são **resultado da simulação**; usá-las como entrada vaza o rótulo. A coluna `papel` do `dicionario_de_dados.csv` marca todas elas com «NAO usar como entrada».
   - `v_min_inrush`, `v_min_label`, `v_max_post`;
   - `tr_load_nom`, `tr_load_af`, `tr_load_inrush_af`;
   - `y_sepoc`, `f3`, `converged`, `eval_ms`;
   - nas bases do estudo SEPOC, também `hist_v_min_inrush` e `hist_feasible`, que são cópias de `v_min_inrush` e de `y`: `hist_feasible` = `y` em 100 % das linhas (MEDIDO).

   `bits`, `n_on` e as colunas de condição (`isc3_kA`, `static_load_mult`, `contingency`) não são features do artigo. Se o grupo decidir usá-las, deve declarar isso. `plant`, `target_motor_id` e `pf_run` são constantes dentro de cada base. `strategy`, `seed`, `sample_id`, `scc_method`, `hist_algorithm` e `hist_seed` são só proveniência.

## 4. Cuidados de uso

- **Leia `bits` como texto.** Tem zeros à esquerda (pandas: `dtype={"bits": str}`). A posição k corresponde à linha `bit = k` de `maquinas_bits_<planta>.csv`.
- **Excel.** Os CSVs usam vírgula como separador e ponto como decimal. No Excel em português, o duplo clique põe a linha inteira na coluna A. Use Dados > Obter Dados > De Texto/CSV, com delimitador «Vírgula», origem «65001: Unicode (UTF-8)» e a coluna `bits` como Texto.
- **Teste fixo e independente.** Use os arquivos `test_*` só para avaliar, nunca para treinar, ajustar hiperparâmetros ou selecionar atributos.
- **Cenários do estudo SEPOC.**
  - Servem para validação externa e para medir viés. **Sozinhos não servem de treino**: todos têm a mesma condição (N-1, Scc fixo, motor de 1250 kW), e só o liga/desliga varia.
  - Na validação externa, o artigo de IC reporta só 2 cenários-chave do NSGA-II: o caso base crítico e a solução Top-1 (DECLARADO no artigo, seções II-D e IV-C). O caso base com todas as máquinas ligadas **não está** nestes arquivos, e a Top-1 não está marcada (MEDIDO: 0 linhas com todas ligadas nos dois arquivos).
  - Os dois arquivos têm 52 vetores liga/desliga em comum, ou seja, o mesmo cenário (MEDIDO). O Pprev difere até 0,032 kW por ruído numérico, e por isso a comparação exata de features não acusa a repetição. Não treine num e valide no outro sem tirar esses 52.
- **`p1_history_Random_s4`.** A coluna `strategy` diz `nsga2_history`, mas a origem é a busca aleatória do estudo SEPOC (coluna `hist_algorithm` = Random).
- **Sobreposição.** Nenhum vetor de features se repete entre a base externa do NSGA-II e o treino/teste da P1 (MEDIDO, `curvas_de_aprendizagem/learning_curve_P1/external_validation.csv`, colunas `overlap_*` = 0).

## 5. «Taxa de amostragem» e tamanho da base

**O que «taxa» significa aqui.** O modelo é quase estático: não há passo de tempo. A «taxa de amostragem» é, portanto, a **densidade de amostras no espaço de cenários**: quantos cenários e em que regiões. Se a pergunta for sobre amostragem no tempo (a forma de onda da partida), ela exige simulação dinâmica (modo `dynamics` do OpenDSS ou EMT). É outro estudo (INFERIDO).

**Como decidir n.** Use uma curva de aprendizagem com conjunto de teste fixo (P1: os dois `test_p1_*` juntos; P2: `test_p2_*`):
- regressão logística L2 com C = 1,0 e padronização;
- 20 reamostragens estratificadas para cada n;
- grade de n: 100, 200, 400, ...;
- **regra do platô:** o n suficiente é o menor n tal que, para ele e para todos os n maiores da grade, |média(n) − referência| ≤ 0,005 e a largura do intervalo de 95 % (percentis das 20 reamostragens) ≤ 0,02.

**Atenção ao reúso do teste.** Escolher n olhando o teste fixo é uma decisão tomada com o teste. Se o grupo fizer isso, deve declarar, ou reportar o desempenho final num conjunto que não entrou na curva (por exemplo, amostras da `p1_stratified_k_n6000_s103` não usadas no treino).

**Nomes das pastas:**
- `V1`: versão do código que acrescenta a referência independente, o teto do pool e o corte por fração, e que nunca aceita como n suficiente um ponto com uma só reamostragem. Com referência em n_max, dá o mesmo n suficiente que a versão antiga das pastas sem `V1` (MEDIDO na base de origem: 0 diferenças ao recalcular as pastas antigas com o código novo);
- `refindep`: referência independente, isto é, 1 ajuste num pool separado de 6000 da mesma estratégia;
- `cap1500`: pool de 6000 subamostrado para 1500;
- `frac05`: só entram na regra os n de até 50 % do pool;
- `pooled` (nas figuras e CSVs): união de `stratified_k` e `bernoulli`, com 3000 amostras.

Resultados (MEDIDO, `curvas_de_aprendizagem/<pasta>/learning_curve_result.json`). **Entre parênteses está o valor de referência do platô**, não o desempenho medido no n suficiente.

| Pasta | Caso | Acurácia: n suficiente (referência) | Recall da Classe 0: n suficiente (referência) |
|---|---|---|---|
| `learning_curve_P1` | P1, `stratified_k`, n_max = 1500 | 800 (0,9855) | 800 (0,9918) |
| `learning_curve_P1` | P1, `bernoulli`, n_max = 1500 | não estabelecido (0,9595) | não estabelecido (0,9559) |
| `learning_curve_P1_V1_refindep` | P1, `stratified_k`, n_max = 1500, referência independente | 800 (0,987) | não estabelecido (0,9928) |
| `learning_curve_P2_V1_refindep` | P2, `stratified_k`, n_max = 1500, referência independente | 800 (0,954) | não estabelecido (0,9252) |
| `learning_curve_P2` | P2, `stratified_k`, n_max = 1500 | 800 (0,951) | não estabelecido (0,9152) |
| `learning_curve_P2_n6000_V1_cap1500` | P2, pool de 6000 subamostrado para 1500 | 400 (0,945) | não estabelecido (0,9152) |
| `learning_curve_P1_n6000_V1` | P1, `stratified_k`, n_max = 6000 | 800 (0,987) | 3200 (0,9928) |
| `learning_curve_P2_n6000` | P2, `stratified_k`, n_max = 6000 | 1600 (0,954) | não estabelecido (0,9252) |
| `learning_curve_P1_n6000_V1_frac05` | P1, `stratified_k`, n_max = 6000, só n ≤ 50 % do pool | 800 (0,987) | não estabelecido (0,9928) |
| `learning_curve_P2_n6000_V1_frac05` | P2, `stratified_k`, n_max = 6000, só n ≤ 50 % do pool | 1600 (0,954) | não estabelecido (0,9252) |

Ressalvas sobre a tabela (MEDIDO nos JSON e CSVs das pastas):
- **O 800 da P2 com referência independente está no limite da regra.** Em n = 800, a acurácia média é 0,94905, contra a referência de 0,954, uma diferença de 0,00495 para ε = 0,005.
- **O 800 da P1 com n_max = 6000 também está perto do limite.** Em n = 800, a acurácia média é 0,9822, contra 0,987, uma diferença de 0,0048. A comparação «P1 800 × P2 1600» com n_max = 6000 depende dessa folga de 0,0002.
- **Os valores 800 (n_max = 1500) e 3200 (recall da P1 com n_max = 6000) estão em n/N ≈ 0,53.** Nessa região, as reamostragens se sobrepõem e o intervalo encolhe. Com a variante `frac05` (`learning_curve_P1_n6000_V1_frac05`), o recall da P1 com n_max = 6000 passa a não estabelecido. Na P2 (`learning_curve_P2_n6000_V1_frac05`), a acurácia continua em 1600.
- **AUC**, que é a métrica primária do `config_geracao_ic.yaml`: o n suficiente fica entre 100 e 400 em todas as curvas `stratified_k` (P1 e P2) e na `pooled`, que só existe na P1. Na `bernoulli`, é 800.

**Leitura (INFERIDO a partir da tabela):**
- O n suficiente depende do tamanho da base usada como referência (n_max) e da regra. Por isso, só compare plantas com o mesmo n_max e a mesma regra.
- Com n_max = 1500 e referência independente, a acurácia estabiliza em 800 nas duas plantas (na P2, no limite da regra). Com n_max = 6000 e a mesma regra (`*_n6000_V1_frac05`), a P1 estabiliza em 800 (também perto do limite) e a P2 em 1600. A pasta `learning_curve_P2_n6000`, da versão antiga, dá o mesmo 1600.
- **Teto da planta complexa.** Com treino de 6000 e teste só `stratified_k`, a acurácia é 0,954 na P2. Na P1, o mesmo desenho dá 0,977 no `test_p1_stratified_k_n1000_s101` e 0,987 no teste combinado (MEDIDO). O recall da Classe 0 é 0,925 na P2, contra cerca de 0,99 na P1.
- Na P2, mais amostras ajudam pouco a partir de certo ponto. O ganho deve vir de variáveis por seção do barramento, de um modelo não linear ou de amostragem dirigida à fronteira.
- **A estratégia importa tanto quanto n.** Com o mesmo n = 1500, a `bernoulli` não estabiliza em acurácia nem em recall da Classe 0; a AUC estabiliza em 800. Ela quase não visita valores extremos de Pprev: 8 amostras abaixo de 5 MW e 5 acima de 12 MW, contra 255 e 221 na `p1_stratified_k_n1500_s1` (MEDIDO).

**Viés dos cenários do NSGA-II** (MEDIDO, `learning_curve_P1/bias_by_strategy.csv`): fração de cenários com |Vmin − 0,85| < 0,01 pu. A métrica considera só a fronteira de tensão; a de carregamento (`tr_load_nom` = 1,0) não entra nela.

| Base | Fração perto da fronteira |
|---|---|
| NSGA-II (`p1_nsga2_history_s3`) | 65,6 % (2707 de 4126) |
| busca aleatória, mesma condição (`p1_history_Random_s4`) | 4,1 % (323 de 7941) |
| `stratified_k` (`p1_stratified_k_n1500_s1`) | 10,4 % (156 de 1500) |
| `bernoulli` (`p1_bernoulli_n1500_s2`) | 15,2 % (228 de 1500) |

**Validação externa** (MEDIDO, `learning_curve_P1/external_validation.csv` e reconferido): um modelo treinado em `p1_stratified_k_n1500_s1` deu, nos 4126 cenários do NSGA-II, acurácia 0,940 e recall da Classe 0 = 1,000.
- Nesses cenários só Pprev varia; as outras 6 features são constantes.
- Por isso, a AUC de 0,999 é a mesma de Pprev sozinho, para qualquer modelo com peso negativo em Pprev, e é idêntica nos 3 modelos do arquivo. Ela não mede a qualidade do modelo.
- A acurácia mede só onde fica o limiar de Pprev que o modelo aprendeu.

O mesmo modelo, no teste fixo da P1 (os dois `test_p1_*` juntos, n = 2000), deu:

| Teste | Acurácia | Recall da Classe 0 | AUC |
|---|---|---|---|
| os dois `test_p1_*` juntos (n = 2000) | 0,9855 | 0,9918 | 0,9995 |
| só `test_p1_stratified_k_n1000_s101` | 0,976 | 0,985 | — |
| só `test_p1_bernoulli_n1000_s102` | 0,995 | 0,998 | — |

## 6. Limitações

- As escolhas de desenho (faixas, proporção N-1, limiares do rótulo, regra do platô) são **padrões propostos** e estão marcadas `[A DECIDIR pelo grupo]` em `config_geracao_ic.yaml`. Não são recomendação.
- **Planta 2.** É um exemplo ilustrativo, não uma planta real; seus parâmetros também são propostas.
  - O gerador local, nos casos medidos, operou no limite de reativos e se comportou como injeção de potência constante, mesmo durante o afundamento (MEDIDO na base de origem). Isso é otimista para a tensão na partida.
  - Em N-1, `v_max_post` vale exatamente 1,0 pu em todas as linhas: é a barra da fonte isolada que domina o máximo.
- **Regeração das bases da P2 em 2026-11-15.** As bases da P2 foram regeradas depois da correção do limite de corrente do disjuntor de interligação (400 A → 2000 A).
  - Só mudaram a coluna auxiliar `f3` e o tempo `eval_ms`. `y`, as 7 features e as demais grandezas elétricas ficaram idênticas (MEDIDO na base de origem).
  - Por isso, as curvas da P2, calculadas antes, continuam válidas.
  - Os `.meta.json` dessas 3 bases registram o local de geração como `work/ic_regen_20261115/`.
  - A `p2_stratified_k_n6000_s203` foi concluída com `--resume`, depois de um travamento do OpenDSS num processo. 23 dos 24 blocos vêm da primeira execução (2026-11-15, 13:01–13:02); o `created` e o tempo total do `.meta.json` são os da retomada.
- **Bases do estudo SEPOC.** Cobrem só a P1, em N-1 e na condição do artigo.
- **Caminhos nos `.meta.json`.** São caminhos absolutos da máquina de origem (`D:\000 - UFMG - DOUTORADO\...`), mantidos como registro de proveniência.
- **Código gerador (`codigo/`).** Foi testado só no Windows 11, com Python 3.13.5 e as versões de `codigo/requirements_ic.txt`. A partir de uma cópia deste pacote numa pasta com acento no caminho, foi medido (2026-11-15):
  - os 12 testes do gerador passam;
  - `p1_stratified_k_n1500_s1` e `test_p2_stratified_k_n1000_s202` saem idênticas às do pacote;
  - `p1_nsga2_history_s3` sai com `bits` e `y` idênticos e Pprev a até 2,3e-6 relativo;
  - a curva `learning_curve_P1` sai idêntica.

  As outras 7 bases e 8 curvas não foram regeradas nesse teste.
- **Não estão neste pacote:**
  - a otimização NSGA-II/NSGA-III do estudo SEPOC e o experimento do PAE (fora do escopo do IC);
  - as pastas de blocos `.blocks`, que são a unidade de retomada da geração e repetem o conteúdo dos CSVs.

## 7. Integridade

`SHA256SUMS.txt` lista o SHA-256 de cada arquivo (exceto o próprio `SHA256SUMS.txt`), com caminho relativo à pasta do pacote e fim de linha LF.

Na pasta extraída, com Git Bash, Linux ou macOS:

```bash
sha256sum -c SHA256SUMS.txt
```

No Windows, sem Git Bash, use o Python:

```bash
python -c "import hashlib,os; r=[(p, os.path.isfile(p) and hashlib.sha256(open(p,'rb').read()).hexdigest()==h) for h,p in (l.split('  ',1) for l in open('SHA256SUMS.txt',encoding='utf-8').read().splitlines())]; print(sum(ok for _,ok in r), 'de', len(r), 'OK'); [print('FALHA', p) for p,ok in r if not ok]"
```

## 8. Uso de IA

O código de simulação, amostragem e curva de aprendizagem foi implementado com auxílio do Claude (Anthropic), assim como a montagem deste pacote e desta documentação.

Os dados são saída do OpenDSS: nenhum valor foi escrito à mão nem alterado. A escolha do modelo, a análise e as conclusões do trabalho de IC cabem ao grupo.
