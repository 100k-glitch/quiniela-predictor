import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

# XGBoost opcional
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    warnings.warn("XGBoost no está instalado. El modelo XGB no será usado.")

def generar_equipos(nombres):
    return {
        nombre: {
            "ataque": random.randint(60, 90),
            "defensa": random.randint(60, 90),
            "experiencia": random.randint(1, 10),
            "estado_animo": random.randint(1, 10),
        }
        for nombre in nombres
    }

def calcular_features_partido(e1, e2, ventaja_local=1.1):
    # Fórmulas útiles para el pronóstico:
    ataque_ajustado_local = e1["ataque"] * e1["experiencia"] * e1["estado_animo"] * ventaja_local / 100
    ataque_ajustado_visit = e2["ataque"] * e2["experiencia"] * e2["estado_animo"] / 100

    diff_ataque_def_local = e1["ataque"] - e2["defensa"]
    diff_ataque_def_visit = e2["ataque"] - e1["defensa"]

    prom_ponderado_local = (e1["ataque"]*0.35 + e1["defensa"]*0.25 + e1["experiencia"]*0.2 + e1["estado_animo"]*0.2)
    prom_ponderado_visit = (e2["ataque"]*0.35 + e2["defensa"]*0.25 + e2["experiencia"]*0.2 + e2["estado_animo"]*0.2)

    momentum_local = e1["experiencia"] * e1["estado_animo"] / 10.0
    momentum_visit = e2["experiencia"] * e2["estado_animo"] / 10.0

    return {
        "ataque_local": e1["ataque"],
        "defensa_local": e1["defensa"],
        "experiencia_local": e1["experiencia"],
        "estado_animo_local": e1["estado_animo"],
        "ataque_visitante": e2["ataque"],
        "defensa_visitante": e2["defensa"],
        "experiencia_visitante": e2["experiencia"],
        "estado_animo_visitante": e2["estado_animo"],
        "ventaja_local": ventaja_local,
        "ataque_ajustado_local": ataque_ajustado_local,
        "ataque_ajustado_visit": ataque_ajustado_visit,
        "diff_ataque_def_local": diff_ataque_def_local,
        "diff_ataque_def_visit": diff_ataque_def_visit,
        "prom_ponderado_local": prom_ponderado_local,
        "prom_ponderado_visit": prom_ponderado_visit,
        "momentum_local": momentum_local,
        "momentum_visit": momentum_visit,
    }

def simular_historial(nombres_equipos, equipos, n=3500):
    historico = []
    for _ in range(n):
        local, visitante = random.sample(nombres_equipos, 2)
        e1, e2 = equipos[local], equipos[visitante]
        ventaja_local = 1.1

        # Simula variación en los atributos
        e1_sim = {k: v + random.randint(-2, 2) if isinstance(v, int) else v for k, v in e1.items()}
        e2_sim = {k: v + random.randint(-2, 2) if isinstance(v, int) else v for k, v in e2.items()}

        features = calcular_features_partido(e1_sim, e2_sim, ventaja_local)

        ataque_local = features["ataque_ajustado_local"] + features["momentum_local"]
        ataque_visit = features["ataque_ajustado_visit"] + features["momentum_visit"]

        goles_local = np.random.poisson(lam=max(0.1, ataque_local / 15))
        goles_visitante = np.random.poisson(lam=max(0.1, ataque_visit / 15))

        if goles_local > goles_visitante:
            resultado = 2  # gana local
        elif goles_local == goles_visitante:
            resultado = 1  # empate
        else:
            resultado = 0  # gana visitante

        features["resultado"] = resultado
        historico.append(features)
    return historico

def entrenar_ensamble(historico, voting_type='soft'):
    df = pd.DataFrame(historico)
    X = df.drop("resultado", axis=1)
    y = df["resultado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('rf', RandomForestClassifier(n_estimators=65, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=65, random_state=42)),
        ('lr', LogisticRegression(max_iter=500, solver="lbfgs"))
    ]
    if has_xgb:
        models.append(('xgb', XGBClassifier(n_estimators=65, use_label_encoder=False, eval_metric='mlogloss', random_state=42)))

    ensemble = VotingClassifier(estimators=models, voting=voting_type)
    ensemble.fit(X_train, y_train)

    print("\nREPORTE DE CADA MODELO:")
    for name, clf in ensemble.named_estimators_.items():
        y_pred = clf.predict(X_test)
        print(f"Modelo: {name}")
        print(classification_report(y_test, y_pred))

    print(f"Modelo: Ensemble (VotingClassifier, voting={voting_type})")
    y_pred = ensemble.predict(X_test)
    print(classification_report(y_test, y_pred))

    return ensemble, models, X.columns, X_test, y_test