"""Base OpenDSS para o trabalho de IC (ESPEC_BASE_CASOS_2026-11-14, secao 5).

Modulos:
  - dataset: gerador da base de cenarios (features pre-manobra + rotulo),
    estrategias bernoulli, stratified_k, exhaustive e nsga2_history;
    avaliacao por comandos `Edit` sobre circuito compilado, validada contra
    recompilacao (`--validate`).
  - learning_curve: curva de aprendizagem (regressao logistica L2) para
    dimensionar o tamanho amostral e comparacao de vies entre estrategias.

Toda escolha de desenho experimental fica em experiments/ic/config.yaml com
o comentario `# [A DECIDIR pelo grupo]` (politica PAE da disciplina).
"""

FEATURES = ["Pprev_kW", "Scc_MVA", "Pn_kW", "Ip_In", "cosphi_p", "eta", "C_N1"]
CONT_TARGETS = ["v_min_inrush", "v_max_post", "tr_load_nom", "tr_load_af"]
PROVENANCE = ["strategy", "seed", "sample_id"]
STRATEGIES = ("bernoulli", "stratified_k", "exhaustive", "nsga2_history")

__all__ = ["FEATURES", "CONT_TARGETS", "PROVENANCE", "STRATEGIES"]
