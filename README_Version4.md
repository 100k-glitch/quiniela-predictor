# Quiniela Inteligente con Predicción de Machine Learning

Este proyecto es una app de escritorio (Tkinter) para hacer quinielas deportivas, que incluye un modelo de predicción basado en ensamble de algoritmos (Random Forest, Gradient Boosting, Logistic Regression y XGBoost).

## ¿Qué hace?

- Simula equipos y partidos de fútbol.
- Entrena un modelo predictivo avanzado.
- Te sugiere pronósticos inteligentes para cada partido.
- Puedes comparar y exportar tu quiniela con las predicciones de la IA.

## Instalación

```bash
pip install -r requirements.txt
```

> Si tienes problemas con `xgboost`, puedes instalarlo así:
> ```bash
> pip install xgboost
> ```

## Uso

Ejecuta la aplicación principal:

```bash
python quiniela_app.py
```

## Estructura de archivos

- `quiniela_modelos.py`: Lógica de simulación, features y entrenamiento del modelo.
- `quiniela_app.py`: Interfaz gráfica de la quiniela.
- `requirements.txt`: Dependencias.
- `README.md`: Esta documentación.

## Personalización

Puedes cambiar los nombres y cantidad de equipos en el bloque `if __name__ == "__main__"` de `quiniela_app.py` para adaptarlo a tu liga o jornada.

---

¡Listo para tus quinielas inteligentes y compartir con tus amigos!