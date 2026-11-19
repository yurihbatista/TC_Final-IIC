"""
Classificação Binária para Triagem Pré-Manobra de Partidas Diretas de Motores
em Sistemas Industriais de Média Tensão

Autora: Cilene Cristina Nunes Faria (PPGEE - UFMG)
Coautor: Yuri Henriques Batista (DEE - UFMG)

Descrição:
    Executa a seleção de atributos por Informação Mútua, validação cruzada
    estratificada (5 dobras), treinamento da Regressão Logística Ridge, avaliação
    no conjunto independente de teste, análise de limiar conservador e validação
    externa com cenários do NSGA-II.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer
)

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================

script_dir = Path(__file__).resolve().parent

candidatos = [
    script_dir / "IC_dados_OpenDSS_2026-11-15" / "dados",
    script_dir / "IC_dados_OpenDSS_2026-11-15",
    script_dir / "dados"
]

base_dir = next(
    (p for p in candidatos if (p / "P1_planta_do_artigo").exists()),
    Path("IC_dados_OpenDSS_2026-11-15/dados")
)

train_df = pd.read_csv(
    base_dir / "P1_planta_do_artigo/p1_stratified_k_n1500_s1.csv",
    dtype={"bits": str}
)
test1_df = pd.read_csv(
    base_dir / "P1_planta_do_artigo/test_p1_stratified_k_n1000_s101.csv",
    dtype={"bits": str}
)
test2_df = pd.read_csv(
    base_dir / "P1_planta_do_artigo/test_p1_bernoulli_n1000_s102.csv",
    dtype={"bits": str}
)
nsga2_df = pd.read_csv(
    base_dir / "P1_cenarios_do_estudo_SEPOC/p1_nsga2_history_s3.csv",
    dtype={"bits": str}
)

test_df = pd.concat([test1_df, test2_df], ignore_index=True)

features = [
    "Pprev_kW",
    "Scc_MVA",
    "Pn_kW",
    "Ip_In",
    "cosphi_p",
    "eta",
    "C_N1"
]

X_train, y_train = train_df[features], train_df["y"]
X_test, y_test = test_df[features], test_df["y"]
X_nsga2, y_nsga2 = nsga2_df[features], nsga2_df["y"]

# ==========================================
# 2. INFORMAÇÃO MÚTUA
# ==========================================

discrete_mask = np.array([False, False, False, False, False, False, True])

mi = mutual_info_classif(
    X_train,
    y_train,
    discrete_features=discrete_mask,
    random_state=42
)

mi_df = pd.DataFrame({
    "Atributo": features,
    "IM (nats)": mi
}).sort_values(by="IM (nats)", ascending=False)

print("--- Informação Mútua ---")
print(mi_df.to_string(index=False))

rotulos_matematicos = {
    "Pprev_kW": r"$P_{\mathrm{prev}}$",
    "C_N1": r"$C_{N-1}$",
    "Scc_MVA": r"$S_{\mathrm{cc}}$",
    "Ip_In": r"$I_p/I_n$",
    "Pn_kW": r"$P_n$",
    "cosphi_p": r"$\cos\varphi_p$",
    "eta": r"$\eta$"
}

mi_plot_df = mi_df.copy()
mi_plot_df["Atributo"] = mi_plot_df["Atributo"].map(rotulos_matematicos)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=mi_plot_df,
    x="IM (nats)",
    y="Atributo",
    hue="Atributo",
    palette="Blues_r",
    legend=False
)
plt.title("Importância dos Atributos por Informação Mútua")
plt.xlabel("Informação Mútua (nats)")
plt.ylabel("Atributos")
plt.tight_layout()
plt.savefig(script_dir / "ranking_informacao_mutua.png", dpi=300)
plt.close()

# ==========================================
# 3. MODELO: REGRESSÃO LOGÍSTICA RIDGE
# ==========================================

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0,
        solver="lbfgs",
        tol=1e-4,
        max_iter=2000,
        random_state=42
    ))
])

# ==========================================
# 4. VALIDAÇÃO CRUZADA
# ==========================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "acc": "accuracy",
    "f1_class1": "f1",
    "recall_class0": make_scorer(recall_score, pos_label=0),
    "precision_class0": make_scorer(precision_score, pos_label=0),
    "roc_auc": "roc_auc"
}

cv_resultados = cross_validate(modelo, X_train, y_train, cv=cv, scoring=scoring)

print("\n=== Validação Cruzada (5 Dobras) ===")
print(f"Acurácia: {cv_resultados['test_acc'].mean()*100:.2f}% +/- {cv_resultados['test_acc'].std()*100:.2f}%")
print(f"F1 Classe 1: {cv_resultados['test_f1_class1'].mean():.4f}")
print(f"Recall Classe 0: {cv_resultados['test_recall_class0'].mean()*100:.2f}%")
print(f"Precisão Classe 0: {cv_resultados['test_precision_class0'].mean()*100:.2f}%")
print(f"ROC-AUC: {cv_resultados['test_roc_auc'].mean():.4f}")

# ==========================================
# 5. TREINAMENTO E TESTE INDEPENDENTE
# ==========================================

modelo.fit(X_train, y_train)

t0 = time.perf_counter()
for _ in range(100):
    y_prob = modelo.predict_proba(X_test)[:, 1]
tempo_inferencia = (((time.perf_counter() - t0) / 100) * 1e6) / len(X_test)

tau_padrao = 0.50
y_pred = (y_prob >= tau_padrao).astype(int)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_test, y_pred)
f1_class1 = f1_score(y_test, y_pred, pos_label=1)
recall_c0 = recall_score(y_test, y_pred, pos_label=0)
precision_c0 = precision_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_prob)

print("\n=== Teste Independente ===")
print(f"Acurácia: {acc*100:.2f}%")
print(f"F1 Classe 1: {f1_class1:.4f}")
print(f"Recall Classe 0: {recall_c0*100:.2f}%")
print(f"Precisão Classe 0: {precision_c0*100:.2f}%")
print(f"ROC-AUC: {auc:.4f}")
print(f"Tempo de inferência: {tempo_inferencia:.2f} us/cenário")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

classes_labels = ["Crítico (0)", "Viável (1)"]

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes_labels,
    yticklabels=classes_labels
)
plt.title("Matriz de Confusão - Teste Independente")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig(script_dir / "matriz_confusao_Reg_Logistica.png", dpi=300)
plt.close()

# ==========================================
# 6. LIMIAR OPERACIONAL
# ==========================================

print("\n=== Avaliação de Limiares ===")
for tau in [0.50, 0.90]:
    preds = (y_prob >= tau).astype(int)
    cm_tau = confusion_matrix(y_test, preds, labels=[0, 1])
    tn_tau, fp_tau, fn_tau, tp_tau = cm_tau.ravel()

    print(
        f"tau={tau:.2f} | "
        f"Acurácia={accuracy_score(y_test, preds)*100:.2f}% | "
        f"Recall C0={recall_score(y_test, preds, pos_label=0)*100:.2f}% | "
        f"Recall C1={recall_score(y_test, preds, pos_label=1)*100:.2f}% | "
        f"FP={fp_tau} | FN={fn_tau}"
    )

# ==========================================
# 7. VALIDAÇÃO EXTERNA NSGA-II
# ==========================================

tau_externo = 0.50
p_nsga2 = modelo.predict_proba(X_nsga2)[:, 1]
pred_nsga2 = (p_nsga2 >= tau_externo).astype(int)

cm_nsga2 = confusion_matrix(y_nsga2, pred_nsga2, labels=[0, 1])
tn_ext, fp_ext, fn_ext, tp_ext = cm_nsga2.ravel()

acc_ext = accuracy_score(y_nsga2, pred_nsga2)
auc_ext = roc_auc_score(y_nsga2, p_nsga2)
recall_c0_ext = recall_score(y_nsga2, pred_nsga2, pos_label=0)
precision_c0_ext = precision_score(y_nsga2, pred_nsga2, pos_label=0)

n_viaveis = (y_nsga2 == 1).sum()
taxa_aprovacao = tp_ext / n_viaveis

print("\n=== Validação Externa NSGA-II ===")
print(f"Acurácia: {acc_ext*100:.2f}%")
print(f"ROC-AUC: {auc_ext:.4f}")
print(f"Recall Classe 0: {recall_c0_ext*100:.2f}%")
print(f"Precisão Classe 0: {precision_c0_ext*100:.2f}%")
print(f"Taxa de aprovação dos viáveis: {taxa_aprovacao*100:.2f}%")
print(f"TN={tn_ext}, FP={fp_ext}, FN={fn_ext}, TP={tp_ext}")

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_nsga2,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes_labels,
    yticklabels=classes_labels
)
plt.title("Matriz de Confusão - Validação Externa NSGA-II")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig(script_dir / "validacao_externa_nsga2.png", dpi=300)
plt.close()

print("\nExecução finalizada com sucesso!")
print(f"Figuras salvas em: {script_dir}")