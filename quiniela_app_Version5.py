import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from quiniela_modelos import (
    generar_equipos,
    calcular_features_partido,
    simular_historial,
    entrenar_ensamble
)

class QuinielaApp:
    def __init__(self, root, partidos, equipos, modelo_ensamble):
        self.root = root
        self.partidos = partidos  # Lista de tuplas: (local, visitante)
        self.equipos = equipos
        self.modelo_ensamble = modelo_ensamble
        self.pronosticos = [tk.StringVar(value="1") for _ in partidos]
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Quiniela Inteligente")
        frm = ttk.Frame(self.root, padding=20)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="Tus Pronósticos y Predicción Inteligente:").grid(row=0, column=0, columnspan=4)
        ttk.Label(frm, text="Partido").grid(row=1, column=0)
        ttk.Label(frm, text="Tu Pronóstico").grid(row=1, column=1)
        ttk.Label(frm, text="Predicción AI").grid(row=1, column=2)
        ttk.Label(frm, text="Probabilidades").grid(row=1, column=3)

        self.pred_labels = []
        self.prob_labels = []

        for i, (local, visitante) in enumerate(self.partidos):
            ttk.Label(frm, text=f"{local} vs {visitante}").grid(row=i+2, column=0)
            cb = ttk.Combobox(frm, textvariable=self.pronosticos[i], values=["1", "X", "2"], state="readonly", width=3)
            cb.grid(row=i+2, column=1)
            pred_label = ttk.Label(frm, text="...")
            pred_label.grid(row=i+2, column=2)
            prob_label = ttk.Label(frm, text="...")
            prob_label.grid(row=i+2, column=3)
            self.pred_labels.append(pred_label)
            self.prob_labels.append(prob_label)

        ttk.Button(frm, text="Actualizar Predicciones", command=self.actualizar_predicciones).grid(row=len(self.partidos)+2, column=0, columnspan=2, pady=10)
        ttk.Button(frm, text="Exportar Quiniela", command=self.exportar_quiniela).grid(row=len(self.partidos)+2, column=2, columnspan=2, pady=10)

        self.actualizar_predicciones()

    def predecir_partido(self, local, visitante):
        e1, e2 = self.equipos[local], self.equipos[visitante]
        features = calcular_features_partido(e1, e2)
        entrada = pd.DataFrame([features])
        textos = ["2", "X", "1"]  # Visitante, Empate, Local
        proba = self.modelo_ensamble.predict_proba(entrada)[0]
        pred = textos[np.argmax(proba)]
        prob_txt = f"1:{proba[2]*100:.1f}% X:{proba[1]*100:.1f}% 2:{proba[0]*100:.1f}%"
        return pred, prob_txt

    def actualizar_predicciones(self):
        for i, (local, visitante) in enumerate(self.partidos):
            pred, proba = self.predecir_partido(local, visitante)
            self.pred_labels[i]["text"] = pred
            self.prob_labels[i]["text"] = proba

    def exportar_quiniela(self):
        quiniela = []
        for i, (local, visitante) in enumerate(self.partidos):
            quiniela.append({
                "partido": f"{local} vs {visitante}",
                "tu_pronostico": self.pronosticos[i].get(),
                "prediccion_AI": self.pred_labels[i]["text"],
                "probabilidades": self.prob_labels[i]["text"]
            })
        df = pd.DataFrame(quiniela)
        archivo = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if archivo:
            df.to_csv(archivo, index=False)
            messagebox.showinfo("Exportación", "¡Tu quiniela fue exportada!")

if __name__ == "__main__":
    # Simulación de equipos y partidos
    nombres_equipos = ["Equipo A", "Equipo B", "Equipo C", "Equipo D", "Equipo E", "Equipo F"]
    equipos = generar_equipos(nombres_equipos)
    historico = simular_historial(nombres_equipos, equipos, n=3500)
    modelo_ensamble, _, _, _, _ = entrenar_ensamble(historico)
    # Simulación de jornada de partidos
    partidos = [
        ("Equipo A", "Equipo B"),
        ("Equipo C", "Equipo D"),
        ("Equipo E", "Equipo F"),
        ("Equipo B", "Equipo C"),
        ("Equipo D", "Equipo E"),
        ("Equipo F", "Equipo A"),
        ("Equipo C", "Equipo F"),
    ]
    root = tk.Tk()
    app = QuinielaApp(root, partidos, equipos, modelo_ensamble)
    root.mainloop()