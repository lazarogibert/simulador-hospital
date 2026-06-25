import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from translations import FARMACOS_TRANSLATION_DICT
import joblib
import shap
import dice_ml
import matplotlib.pyplot as plt
import os
import re
import warnings
import json_repair
import google.generativeai as genai
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.inspection import PartialDependenceDisplay
import lime
import lime.lime_tabular
import umap

# --- BLINDAJE ESTRUCTURAL SKLEARN ---
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel

try:
    sklearn.set_config(transform_output="pandas")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ==========================================
# CLASE SELECTORA MÉDICA (OVERRIDE PARA PICKLE/JOBLIB)
# ==========================================
class SelectorConConocimientoMedico(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, threshold='median', variables_obligatorias=None):
        self.estimator = estimator
        self.threshold = threshold
        self.variables_obligatorias = variables_obligatorias if variables_obligatorias else []
        self.selector_base = SelectFromModel(estimator=self.estimator, threshold=self.threshold)
        
    def fit(self, X, y=None):
        self.selector_base.fit(X, y)
        self.mascara_final_ = self.selector_base.get_support().copy()
        
        if hasattr(X, 'columns'):
            self.nombres_columnas_ = X.columns
        else:
            raise ValueError("El input debe ser un DataFrame. Activa set_output(transform='pandas').")
            
        for i, col in enumerate(self.nombres_columnas_):
            col_str = str(col)
            col_limpia = col_str.replace('num__', '').replace('cat__', '')
            
            if col_limpia.startswith(('LLM_', 'ING_', 'EVO_', 'Riesgo_')):
                es_intocable = any(var_obl in col_limpia for var_obl in self.variables_obligatorias)
                if es_intocable:
                    self.mascara_final_[i] = True 
            else:
                self.mascara_final_[i] = True
                
        self.variables_descartadas_ = self.nombres_columnas_[~self.mascara_final_].tolist()
        self.variables_mantenidas_ = self.nombres_columnas_[self.mascara_final_].tolist()
                
        return self
        
    def transform(self, X):
        import scipy.sparse
        if scipy.sparse.issparse(X) or hasattr(X, 'toarray'):
            X_seguro = X.toarray()
        else:
            X_seguro = X
            
        if hasattr(X_seguro, 'loc'):
            return X_seguro.loc[:, self.mascara_final_]
        else:
            return X_seguro[:, self.mascara_final_]
            
    def get_support(self):
        return self.mascara_final_

# ==========================================
# 0. FUNCIONES DE PROCESAMIENTO CLÍNICO (CIE-10)
# ==========================================

def normalizar_cie10(codigo):
    if pd.isna(codigo): return pd.NA
    codigo = str(codigo).strip().upper().replace(".", "").replace(" ", "")
    m = re.match(r'^([A-Z])(\d{2})', codigo)
    if not m: return pd.NA
    return f"{m.group(1)}{int(m.group(2)):02d}"

def mapear_cie10_macro(cod):
    if pd.isna(cod):
        return "Desconocido"

    letra = cod[0]
    num = int(cod[1:3])

    if letra == "A":
        if 15 <= num <= 19: return "Tuberculosis"
        elif 30 <= num <= 30: return "Lepra"
        elif 50 <= num <= 53: return "Sífilis"
        else: return "Otras infecciosas (A)"

    if letra == "B":
        if 15 <= num <= 19: return "Hepatitis viral"
        elif 20 <= num <= 24: return "Enfermedad por VIH"
        elif 57 <= num <= 57: return "Enfermedad de Chagas"
        elif 58 <= num <= 58: return "Toxoplasmosis"
        elif 67 <= num <= 67: return "Equinococosis / Hidatidosis"
        elif 90 <= num <= 94: return "Secuelas de enfermedades infecciosas"
        else: return "Otras infecciosas (B)"

    if letra == "C":
        if 0 <= num <= 14: return "Cáncer de labio / boca / faringe"
        elif 15 <= num <= 26: return "Cáncer digestivo"
        elif 30 <= num <= 39: return "Cáncer respiratorio / intratorácico"
        elif 40 <= num <= 41: return "Cáncer de hueso / cartílago"
        elif 43 <= num <= 44: return "Melanoma / Cáncer de piel"
        elif 50 <= num <= 50: return "Cáncer de mama"
        elif 51 <= num <= 58: return "Cáncer genital femenino"
        elif 60 <= num <= 63: return "Cáncer genital masculino"
        elif 64 <= num <= 68: return "Cáncer de vías urinarias"
        elif 69 <= num <= 72: return "Cáncer de sistema nervioso central"
        elif 81 <= num <= 96: return "Cáncer linfoide / hematopoyético"
        else: return "Otros tumores malignos"

    if letra == "D":
        if 0 <= num <= 48: return "Tumores in situ o benignos"
        elif 50 <= num <= 53: return "Anemias nutricionales"
        elif 55 <= num <= 59: return "Anemias hemolíticas"
        elif 60 <= num <= 64: return "Aplasias y otras anemias"
        elif 65 <= num <= 69: return "Defectos de coagulación / púrpura"
        elif 80 <= num <= 89: return "Trastornos de inmunodeficiencia"
        else: return "Otros trastornos de la sangre"

    if letra == "E":
        if 0 <= num <= 7: return "Tiroides"
        elif 8 <= num <= 13: return "Diabetes"
        elif 15 <= num <= 16: return "Glucosa / hipoglucemia"
        elif 20 <= num <= 35: return "Otros endocrinos y metabólicos"
        elif 65 <= num <= 68: return "Obesidad y trastornos de hiperalimentación"
        elif 70 <= num <= 90: 
            if num == 78: return "Dislipidemia"
            elif num == 84: return "Fibrosis quística"
            return "Trastornos metabólicos"
        else: return "Otros metabólicos / nutricionales"

    if letra == "F":
        if 0 <= num <= 9: return "Trastornos mentales orgánicos (Demencias)"
        elif 10 <= num <= 19: return "Trastornos por uso de sustancias"
        elif 20 <= num <= 29: return "Esquizofrenia y trastornos psicóticos"
        elif 30 <= num <= 39: return "Trastornos del humor (Afectivos)"
        elif 40 <= num <= 48: return "Trastornos neuróticos y de ansiedad"
        elif 50 <= num <= 59: return "Trastornos de la conducta alimentaria / sueño"
        elif 60 <= num <= 69: return "Trastornos de la personalidad"
        elif 70 <= num <= 79: return "Discapacidad intelectual"
        elif 80 <= num <= 89: return "Trastornos del desarrollo psicobiológico (Autismo)"
        else: return "Otros trastornos mentales"

    if letra == "G":
        if 10 <= num <= 14: return "Atrofias sistémicas del SNC"
        elif 20 <= num <= 26: return "Trastornos extrapiramidales y del movimiento (Parkinson)"
        elif 30 <= num <= 32: return "Enfermedades degenerativas (Alzheimer)"
        elif 35 <= num <= 37: return "Enfermedades desmielinizantes (Esclerosis Múltiple)"
        elif 40 <= num <= 47: return "Trastornos episódicos y paroxísticos (Epilepsia, Migraña)"
        elif 50 <= num <= 59: return "Trastornos de nervios y plexos"
        elif 60 <= num <= 64: return "Polineuropatías"
        elif 70 <= num <= 73: return "Enfermedades de la unión neuromuscular (Miastenia)"
        elif 80 <= num <= 83: return "Parálisis cerebral y síndromes paralíticos"
        else: return "Otros trastornos neurológicos"

    if letra == "H":
        if 0 <= num <= 59: return "Ojo"
        elif 60 <= num <= 95: return "Oído"
        else: return "Otros órganos de los sentidos"

    if letra == "I":
        if 10 <= num <= 15: return "Hipertensión"
        elif 20 <= num <= 25: return "Cardiopatía isquémica"
        elif 26 <= num <= 28: return "Enfermedad cardiopulmonar"
        elif 30 <= num <= 52: return "Otras enfermedades del corazón (Insuficiencia Cardíaca)"
        elif 60 <= num <= 69: return "Cerebrovascular"
        elif 70 <= num <= 79: return "Enfermedades de arterias y capilares"
        elif 80 <= num <= 89: return "Enfermedades de venas y vasos linfáticos"
        else: return "Otros circulatorios"

    if letra == "J":
        if 0 <= num <= 6: return "Vías respiratorias altas"
        elif 9 <= num <= 18: return "Infecciones agudas / neumonía / influenza"
        elif 20 <= num <= 22: return "Infecciones respiratorias bajas"
        elif 30 <= num <= 39: return "Enfermedades de vías respiratorias superiores"
        elif 40 <= num <= 47: return "Asma / EPOC / bronquitis"
        elif 60 <= num <= 70: return "Enfermedades del pulmón por agentes externos (Neumoconiosis)"
        elif 80 <= num <= 84: return "Enfermedades pulmonares intersticiales"
        else: return "Otros respiratorios"

    if letra == "K":
        if 0 <= num <= 14: return "Boca / dientes / faringe"
        elif 20 <= num <= 31: return "Esófago / estómago / duodeno"
        elif 35 <= num <= 38: return "Apendicitis"
        elif 40 <= num <= 46: return "Hernias"
        elif 50 <= num <= 52: return "Enfermedad de Crohn y colitis"
        elif 55 <= num <= 63: return "Otras enfermedades de los intestinos"
        elif 70 <= num <= 77: return "Hígado"
        elif 80 <= num <= 87: return "Vesícula / vías biliares / páncreas"
        else: return "Otros digestivos"

    if letra == "L":
        if 20 <= num <= 30: return "Dermatitis y eczema"
        elif 40 <= num <= 45: return "Trastornos papuloescamosos (Psoriasis)"
        elif 50 <= num <= 54: return "Urticaria y eritema"
        elif 80 <= num <= 99: return "Trastornos de las faneras / Otros trastornos de piel"
        else: return "Otras enfermedades de la piel"

    if letra == "M":
        if 0 <= num <= 25: return "Artropatías"
        elif 30 <= num <= 36: return "Tejido conectivo (Lupus, etc.)"
        elif 40 <= num <= 54: return "Dorsopatías"
        elif 60 <= num <= 79: return "Tejidos blandos"
        elif 80 <= num <= 94: return "Osteopatías y condropatías (Osteoporosis)"
        else: return "Otros osteomusculares"

    if letra == "N":
        if 0 <= num <= 29: return "Riñón (Insuficiencia Renal Crónica)"
        elif 30 <= num <= 39: return "Vías urinarias bajas"
        elif 40 <= num <= 51: return "Genital masculino (Hiperplasia Prostática)"
        elif 60 <= num <= 64: return "Mama"
        elif 70 <= num <= 98: return "Genital femenino (Endometriosis, etc.)"
        else: return "Otros genitourinarios"
            
    if letra == "Q":
        if 0 <= num <= 7: return "Malformaciones del sistema nervioso (Espina bífida)"
        elif 20 <= num <= 28: return "Malformaciones cardíacas congénitas"
        elif 90 <= num <= 99: return "Anomalías cromosómicas (Síndrome de Down)"
        else: return "Otras malformaciones congénitas"

    if letra == "P":
        if num == 27: return "Enfermedad respiratoria crónica perinatal"
        else: return pd.NA

    if letra == "T":
        if 90 <= num <= 98: return "Secuelas crónicas de traumatismos"
        else: return pd.NA
            
    if letra == "U":
        if num == 9: return "Síndrome Post-COVID (Long COVID)"
        else: return "Otras condiciones especiales (U)"

    if letra == "Z":
        if 85 <= num <= 87: return "Historia personal de tumores / enfermedades"
        elif 89 <= num <= 90: return "Ausencia adquirida de miembros / órganos"
        elif 93 <= num <= 93: return "Aberturas artificiales (Ostomías)"
        elif 94 <= num <= 94: return "Estado de órgano trasplantado"
        elif 95 <= num <= 95: return "Presencia de implantes cardíacos / vasculares"
        elif 99 <= num <= 99: return "Dependencia de máquinas (diálisis, oxígeno)"
        else: return "Otros factores de salud"

    return "Desconocido"


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

class MotorEDADinamico:
    def __init__(self, ruta_npy):
        self.ruta_npy = ruta_npy
        self.df = None
        
        self.mapa_periodos = {
            1: '1. Premature (≤ 24h)',
            2: '2. Early (1 - 7 days)',
            3: '3. Late (8 - 15 days)'
        }

    def cargar_datos(self):
        if not os.path.exists(self.ruta_npy):
            return False
            
        arreglo_numpy = np.load(self.ruta_npy, allow_pickle=True)
        self.df = pd.DataFrame.from_records(arreglo_numpy)
        self.df['Periodo_Reingreso'] = self.df['target_15d_desde_alta_cat'].map(self.mapa_periodos)
        
        if 'IN_MOTING' in self.df.columns:
            self.df['CIE10_Agrupado'] = self.agrupar_cie10(self.df['IN_MOTING'])
        return True

    def agrupar_cie10(self, serie):
        primera_letra = serie.fillna('X').astype(str).str.strip().str.upper().str[0]
        
        # Mapeo macro exhaustivo del CIE-10 (A-Z)
        mapeo_macro = {
            'A': 'Infectious & Parasitic', 
            'B': 'Infectious & Parasitic', 
            'C': 'Oncology (Malignant)', 
            'D': 'Oncology (Benign) & Blood/Immunity', 
            'E': 'Endocrine, Nutritional & Metabolic', 
            'F': 'Mental Health & Behavioral', 
            'G': 'Neurology (Nervous System)', 
            'H': 'Sense Organs (Eye & Ear)', 
            'I': 'Cardiovascular', 
            'J': 'Respiratory', 
            'K': 'Digestive', 
            'L': 'Dermatology (Skin)', 
            'M': 'Musculoskeletal & Connective Tissue', 
            'N': 'Genitourinary',
            'O': 'Pregnancy & Obstetrics', 
            'P': 'Perinatal Conditions', 
            'Q': 'Congenital Anomalies', 
            'R': 'Unclassified Symptoms & Signs', 
            'S': 'Trauma & Poisoning', 
            'T': 'Trauma & Poisoning',
            'U': 'Special Conditions (e.g., COVID-19)', 
            'V': 'External Causes', 
            'W': 'External Causes', 
            'X': 'External Causes', 
            'Y': 'External Causes', 
            'Z': 'Health Service Contact'
        }
        
        return primera_letra.map(mapeo_macro).fillna('Other / Invalid Codes')

    def plot_incidencia_acumulada(self, variable_segmentacion='EST_paso_por_uti'):
        if 'tiempo_exacto_reingreso_horas_alta' not in self.df.columns or variable_segmentacion not in self.df.columns: return None

        df_plot = self.df.dropna(subset=['tiempo_exacto_reingreso_horas_alta']).copy()
        
        if variable_segmentacion == 'EST_paso_por_uti':
            df_plot['Grupo'] = df_plot[variable_segmentacion].astype(str).map({'1': 'ICU Stay', '1.0': 'ICU Stay', '0': 'General Ward', '0.0': 'General Ward'}).fillna('No Data')
            titulo_var = "Clinical Severity (ICU)"
        # --- BLOQUE: TRIAGE ---
        elif variable_segmentacion == 'TR_Prioridad':
            mapeo_prioridad = {
                '0': '0: Non-Urgent', '0.0': '0: Non-Urgent',
                '1': '1: Standard', '1.0': '1: Standard',
                '2': '2: Urgent', '2.0': '2: Urgent',
                '3': '3: Emergency', '3.0': '3: Emergency'
            }
            df_plot['Grupo'] = df_plot[variable_segmentacion].astype(str).map(mapeo_prioridad).fillna('No Data')
            titulo_var = "Triage Priority"
        # ----------------------------
        elif variable_segmentacion == 'pluripatologico':
            df_plot['Grupo'] = df_plot[variable_segmentacion].astype(str).map({'1': 'Multimorbidity', '1.0': 'Multimorbidity', '0': 'Single Pathology', '0.0': 'Single Pathology'}).fillna('No Data')
            titulo_var = "Multimorbidity"
        
        elif variable_segmentacion == 'visitas_guardia_6meses_previos':
            condiciones = [
                df_plot[variable_segmentacion] == 0,
                df_plot[variable_segmentacion].isin([1, 2]),
                df_plot[variable_segmentacion] >= 3
            ]
            df_plot['Grupo'] = np.select(condiciones, ['A: 0 visits', 'B: 1-2 visits', 'C: 3+ visits (Hyper-frequenters)'], default='No Data')
            titulo_var = "ER History"
        elif variable_segmentacion == 'PA_SITLABO_x':
            mapa_laboral = {
                'J': 'Retired',
                'B': 'Unemployed - Seeking Work',
                'A': 'Homemaker',
                'N': 'Unemployed - Not Seeking Work',
                'T': 'Employed or On Leave'
            }
            df_plot['Grupo'] = df_plot[variable_segmentacion].astype(str).str.strip().str.upper().map(mapa_laboral).fillna('No Data')
            titulo_var = "Employment Status"
        else:
            df_plot['Grupo'] = df_plot[variable_segmentacion].astype(str)
            titulo_var = str(variable_segmentacion)

        df_plot = df_plot[df_plot['Grupo'] != 'No Data']

        fig = px.ecdf(
            df_plot,
            x="tiempo_exacto_reingreso_horas_alta",
            color="Grupo",
            title=f"Failure Velocity: Cumulative Incidence Curve by {titulo_var}",
            labels={'tiempo_exacto_reingreso_horas_alta': 'Continuous Hours Since Discharge', 'Grupo': titulo_var},
            lines=True,
            markers=False
        )
        
        fig.add_vline(x=24, line_dash="dash", line_color="#e74c3c", annotation_text="Premature (24h)")
        fig.add_vline(x=168, line_dash="dash", line_color="#f39c12", annotation_text="Early (7d)")
        fig.add_vline(x=360, line_dash="dash", line_color="#2ecc71", annotation_text="Late (15d)")
        
        fig.update_xaxes(range=[-5, 380])
        fig.update_layout(yaxis_title="Cumulative Proportion of Readmitted Patients", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        return fig

    def plot_perfil_clinico(self, modo='absolute'):
        if 'pluripatologico' not in self.df.columns: return None
            
        df_plot = self.df.copy()
        df_plot['Condition'] = df_plot['pluripatologico'].astype(str).map({'1': 'Multimorbidity', '0': 'Single Pathology', '1.0': 'Multimorbidity', '0.0': 'Single Pathology'}).fillna('No Data')
        df_plot = df_plot[df_plot['Condition'] != 'No Data']
            
        if modo == 'relative':
            fig = px.histogram(
                df_plot, x="Periodo_Reingreso", color="Condition", barmode="stack", barnorm="percent", text_auto=".1f",
                title="Readmission Composition by Multimorbidity (%)", labels={'Periodo_Reingreso': 'Period', 'Condition': 'Condition'},
                color_discrete_sequence=['#ef553b', '#636efa']
            )
            fig.update_layout(yaxis_title="% of Patients in Period")
        else:
            fig = px.histogram(
                df_plot, x="Periodo_Reingreso", color="Condition", barmode="group", text_auto=True,
                title="Readmission Volume by Multimorbidity", labels={'Periodo_Reingreso': 'Period', 'Condition': 'Condition'},
                color_discrete_sequence=['#ef553b', '#636efa']
            )
            fig.update_layout(yaxis_title="Number of Patients")

        fig.update_layout(xaxis={'categoryorder':'category ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig

    def plot_gravedad_hospitalaria(self, variable_segmentacion='EST_paso_por_uti'):
        if 'dias_internados' not in self.df.columns or variable_segmentacion not in self.df.columns: return None
            
        if variable_segmentacion == 'EST_paso_por_uti':
            self.df['Severity_Label'] = self.df[variable_segmentacion].astype(str).map({'1': 'ICU Stay', '1.0': 'ICU Stay', '0': 'General Ward', '0.0': 'General Ward'}).fillna('No Data')
            titulo_eje = 'Severity (ICU)'
        elif variable_segmentacion == 'TR_Prioridad':
            mapeo_prioridad = {
                '0': '0: Non-Urgent', '0.0': '0: Non-Urgent',
                '1': '1: Standard', '1.0': '1: Standard',
                '2': '2: Urgent', '2.0': '2: Urgent',
                '3': '3: Emergency', '3.0': '3: Emergency'
            }
            self.df['Severity_Label'] = self.df[variable_segmentacion].astype(str).map(mapeo_prioridad).fillna('No Data')
            titulo_eje = 'Triage Priority'
        else:
            return None

        df_plot = self.df[self.df['Severity_Label'] != 'No Data'].copy()
        
        df_grouped = df_plot.groupby(['Periodo_Reingreso', 'Severity_Label'])['dias_internados'].agg(
            Mediana='median',
            Q25=lambda x: x.quantile(0.25),
            Q75=lambda x: x.quantile(0.75)
        ).reset_index()

        df_grouped['Mediana'] = df_grouped['Mediana'].round(1)
        df_grouped['Q25'] = df_grouped['Q25'].round(1)
        df_grouped['Q75'] = df_grouped['Q75'].round(1)

        fig = px.bar(
            df_grouped, x="Periodo_Reingreso", y="Mediana", color="Severity_Label", barmode="group",
            text="Mediana", 
            custom_data=["Q25", "Q75"], 
            title=f"Hospital Attrition: Median Length of Stay by {titulo_eje}",
            labels={'Mediana': 'Length of Stay (Median Days)', 'Periodo_Reingreso': 'Period', 'Severity_Label': titulo_eje}
        )
        
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                          titulo_eje + ": <b>%{series_name}</b><br>" +
                          "Typical Stay (Median): <b>%{y} days</b><br>" +
                          "Expected Range (IQR): <b>%{customdata[0]} to %{customdata[1]} days</b><extra></extra>"
        )

        fig.update_layout(xaxis={'categoryorder':'category ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig
        
    def plot_contexto_social(self, modo='absolute'):
        if 'PA_SITLABO_x' not in self.df.columns: return None
            
        df_plot = self.df.copy()
        
        mapa_laboral = {
            'J': 'Retired',
            'B': 'Unemployed - Seeking Work',
            'A': 'Homemaker',
            'N': 'Unemployed - Not Seeking Work',
            'T': 'Employed or On Leave'
        }
        
        df_plot['PA_SITLABO_x'] = df_plot['PA_SITLABO_x'].astype(str).str.strip().str.upper().map(mapa_laboral).fillna('No Data')
        df_plot = df_plot[df_plot['PA_SITLABO_x'] != 'No Data']
        
        if modo == 'relative':
            fig = px.histogram(
                df_plot, x="Periodo_Reingreso", color="PA_SITLABO_x", barmode="stack", barnorm="percent", text_auto=".1f",
                title="Employment Status Composition by Period (%)", labels={'Periodo_Reingreso': 'Readmission Period', 'PA_SITLABO_x': 'Employment Status'}
            )
            fig.update_layout(yaxis_title="% of Patients in Period")
        else:
            fig = px.histogram(
                df_plot, x="Periodo_Reingreso", color="PA_SITLABO_x", barmode="group", text_auto=True,
                title="Employment Status Volume by Period", labels={'Periodo_Reingreso': 'Readmission Period', 'PA_SITLABO_x': 'Employment Status'}
            )
            fig.update_layout(yaxis_title="Number of Patients")

        fig.update_layout(xaxis={'categoryorder':'category ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig
    
    def plot_historial_paciente(self, modo='absolute'):
        if 'visitas_guardia_6meses_previos' not in self.df.columns: return None

        condiciones = [
            self.df['visitas_guardia_6meses_previos'] == 0,
            self.df['visitas_guardia_6meses_previos'].isin([1, 2]),
            self.df['visitas_guardia_6meses_previos'] >= 3
        ]
        opciones = ['A: 0 visits', 'B: 1-2 visits', 'C: 3+ visits (Hyper-frequenters)']
        self.df['Categoria_Visitas'] = np.select(condiciones, opciones, default='No Data')
        
        df_plot = self.df[self.df['Categoria_Visitas'] != 'No Data'].copy()

        if modo == 'relative':
            fig = px.histogram(
                df_plot, x="Periodo_Reingreso", color="Categoria_Visitas", barmode="stack", barnorm="percent", text_auto=".1f",
                title="The Revolving Door: Patient Composition by Recent ER History (%)",
                labels={'Periodo_Reingreso': 'Period', 'Categoria_Visitas': 'Prior Visits (6 Months)'},
                color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'] 
            )
            fig.update_layout(yaxis_title="% of Patients in Period")
        else:
            fig = px.histogram(
                df_plot, x="Periodo_Reingreso", color="Categoria_Visitas", barmode="group", text_auto=True,
                title="The Revolving Door: Patient Volume by Recent ER History",
                labels={'Periodo_Reingreso': 'Period', 'Categoria_Visitas': 'Prior Visits (6 Months)'},
                color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'] 
            )
            fig.update_layout(yaxis_title="Number of Patients")
            
        fig.update_layout(xaxis={'categoryorder':'category ascending'}, legend={'traceorder':'normal'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig

    def plot_motivo_ingreso(self):
        if 'CIE10_Agrupado' not in self.df.columns: return None
            
        df_plot = self.df[~self.df['CIE10_Agrupado'].isin(['Other Pathologies', 'Health Service Contact'])].copy()
        matriz = pd.crosstab(df_plot['CIE10_Agrupado'], df_plot['Periodo_Reingreso'], normalize='columns') * 100
        matriz = matriz.fillna(0)
        
        col_prematuro = [c for c in matriz.columns if 'Premature' in c]
        if col_prematuro: matriz = matriz.sort_values(by=col_prematuro[0], ascending=False)
            
        fig = px.imshow(
            matriz, text_auto=".1f", aspect="auto", color_continuous_scale="Reds",
            title="Clinical Signature: ICD-10 Distribution by Period (%)",
            labels=dict(x="Readmission Period", y="ICD-10 Chapter", color="% of Patients")
        )
        fig.update_xaxes(categoryorder='category ascending')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig
    def analizar(self, tipo_analisis, variable_segmentacion='EST_paso_por_uti', modo='absolute'):
        opciones = {
            'curva': lambda: self.plot_incidencia_acumulada(variable_segmentacion), 
            'clinico': lambda: self.plot_perfil_clinico(modo),
            'gravedad': lambda: self.plot_gravedad_hospitalaria(variable_segmentacion),
            'social': lambda: self.plot_contexto_social(modo),
            'historial': lambda: self.plot_historial_paciente(modo),
            'cie10': self.plot_motivo_ingreso # Ahora encontrará la función correctamente
        }
        if tipo_analisis.lower() in opciones: 
            return opciones[tipo_analisis.lower()]()
        return None

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Safe Discharge Simulator", layout="wide")
st.markdown("<h2 style='font-size: 32px; font-weight: 600; margin-bottom: 20px;'>🏥 Clinical Safe Discharge Simulator (15 Days)</h2>", unsafe_allow_html=True)

# ==========================================
# 2. MODEL AND DATA LOADING (CACHE)
# ==========================================
@st.cache_resource
def cargar_entorno():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(directorio_actual, 'modelo_reingreso_nlp_41vars_v2.pkl')
    
    paquete = joblib.load(ruta_modelo)
    pipeline = paquete['pipeline']
    umbral = paquete['umbral']
    cols_modelo = paquete['nombres_columnas']
    
    return pipeline, umbral, cols_modelo


pipeline, umbral, columnas_modelo = cargar_entorno()



# ==========================================
# 3. INTERFACE CAPTURE (MANUAL CORE & NLP AUTOMATION)
# ==========================================
st.sidebar.header("🩺 Phenotype Loading")

# --- REFINED DICTIONARIES ---
opciones_edad_dict = {
    "Young Adult": "Adulto joven", 
    "Middle-aged Adult": "Adulto de mediana edad", 
    "Older Adult": "Adulto mayor"
}

area_map = {
    "Internal Medicine": "Clinica_Medica",
    "ER (Emergency Room)": "Emerg_Guardias"
}

perfil_clinico_map = {
    "Initial Admission": "Internacion inicial",
    "Admission-Associated Complication": "Complicacion asociada a la internecion",
    "Multimorbidity Decompensation": "Descompensacion de pluripatologia",
    "Same Cause": "Misma causa",
    "Unrelated Readmission": "Reinternacion no asociada"
}

cro_dict = {
    'Active Smoking': 'tabaquismo_activo', 
    'Polypharmacy': 'polifarmacia', 
    'History of Falls': 'historial_caidas', 
    'Medication Abandonment': 'abandono_medicacion',
    'Geriatric Frailty': 'fragilidad_geriatrica',
    'Poor Support Network': 'red_apoyo_deficiente',
    'Comprehension Barrier': 'barrera_comprension'
}

ing_dict = {
    'Mental Alteration': 'alteracion_mental', 
    'Repeated Consultations': 'consultas_reiteradas', 
    'Functional Dependency': 'dependencia_funcional', 
    'Device Bearer': 'portador_dispositivos', 
    'Hemorrhagic Risk': 'riesgo_hemorragico', 
    'Active Infection': 'infeccion_activa',
    'Severe Multimorbidity': 'multimorbilidad_severa'
}

evo_dict = {
    'Infectious Isolation': 'aislamiento_infeccioso', 
    'Mental Alteration': 'alteracion_mental', 
    'Hospitalization Complication': 'complicacion_internacion', 
    'Functional Dependency': 'dependencia_funcional', 
    'Device Bearer': 'portador_dispositivos', 
    'Major Therapeutic Change': 'cambio_terapeutico_mayor', 
    'Surgical Intervention': 'intervencion_quirurgica', 
    'Transfusion Support': 'soporte_transfusional',
    'Prolonged IV Therapy': 'terapia_endovenosa_prolongada',
    'Residual Instability': 'inestabilidad_residual'
}

# State Initialization
for key in ["ui_cro_sel", "ui_ing_sel", "ui_evo_sel"]:
    if key not in st.session_state: st.session_state[key] = []
for key in ["ui_ing_dolor", "ui_ing_grav", "ui_evo_dolor", "ui_evo_grav"]:
    if key not in st.session_state: st.session_state[key] = 0 if 'dolor' in key else 5
if 'nlp_processed' not in st.session_state: st.session_state.nlp_processed = False
if 'nlp_quotes' not in st.session_state: st.session_state.nlp_quotes = {}

# --- BLOQUE 1: INPUT MANUAL OBLIGATORIO ---
st.sidebar.subheader("1. Core Parameters (Manual Entry)")
cie10_input = st.sidebar.text_input("Reason for admission (ICD-10 Code):", value="I10", help="Example: I10, E11, J44")
dias_internados = st.sidebar.number_input("Number of days hospitalized:", min_value=0, max_value=150, value=5)

rango_edad_ui = st.sidebar.selectbox("Patient Age Range:", list(opciones_edad_dict.keys()))
rango_edad = opciones_edad_dict[rango_edad_ui].upper()

sexo_map = {"Male": "MASCULINO", "Female": "FEMENINO"}
sexo_ui = st.sidebar.selectbox("Sex:", list(sexo_map.keys()))
sexo_input = sexo_map[sexo_ui]

area_ui = st.sidebar.selectbox("Admission Area:", list(area_map.keys()))
area_input = area_map[area_ui]

perfil_ui = st.sidebar.selectbox("Admission Clinical Profile:", list(perfil_clinico_map.keys()))
perfil_input = perfil_clinico_map[perfil_ui].upper()

complejidad_input = st.sidebar.number_input("Complexity Level (IN_COMPLEJIDAD):", min_value=1, step=1, value=1)
# --- NUEVA LÍNEA ---
prioridad_input = st.sidebar.selectbox("Triage Priority (TR_Prioridad):", options=[0, 1, 2, 3], index=0, help="0: Non-urgent, 3: Resuscitation/Emergency")

interconsultas_input = st.sidebar.number_input("Interconsultations:", min_value=0, value=0)
visitas_guardia_input = st.sidebar.number_input("ER Visits (Previous 6 months):", min_value=0, value=0)

es_pluripatologico = st.sidebar.checkbox("Has Multimorbidity (Pluripathological)?", value=False)
ingreso_ambulancia = st.sidebar.checkbox("Arrived by Ambulance?", value=False)
paso_por_uti = st.sidebar.checkbox("ICU Stay during admission?", value=False)

st.sidebar.markdown("#### High-Risk Medications")
med_cardio = st.sidebar.checkbox("Cardiovascular / Inotropes", value=False)
med_psico = st.sidebar.checkbox("Psychotropics / Neurologicals", value=False)

st.sidebar.markdown("---")

# --- BLOQUE 2: MOTOR NLP (AUTOMATIZACIÓN) ---
st.sidebar.subheader("2. Narrative Phenotype (NLP)")

api_key_default = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key_default = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key_default = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass 

api_key = st.sidebar.text_input(
    "Gemini API Key (Live Triage):", 
    value=api_key_default,
    type="password", 
    help="Automatically loaded from Streamlit Secrets."
)

ing_text = st.sidebar.text_area("Admission Notes:", height=100)
evo_text = st.sidebar.text_area("Evolution Notes:", height=100)

if st.sidebar.button("🧠 Run NLP Extraction", use_container_width=True):
    if not api_key:
        st.sidebar.error("API Key required.")
    elif not ing_text and not evo_text:
        st.sidebar.warning("Please provide clinical notes.")
    else:
        with st.spinner("Forensic extraction in progress..."):
            try:
                genai.configure(api_key=api_key)
                modelo_nlp = genai.GenerativeModel(
                    'models/gemini-2.5-flash', 
                    generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.0)
                )

                prompt_sistema = """
Eres un auditor médico forense estricto. Tu tarea es extraer variables de riesgo clínico de la historia clínica mediante un análisis determinista y algorítmico.

REGLAS DE ORO (INQUEBRANTABLES):
1. CERO INVENCIÓN (EXTRACCIÓN SEMÁNTICA): No inventes diagnósticos. DEBES marcar una variable como 'true' SOLO si el texto describe explícitamente los términos, signos, síntomas o equivalentes clínicos definidos en nuestro Diccionario.
2. NEGACIONES ABSOLUTAS: Si un síntoma/condición está negado ("niega", "sin", "no presenta"), equivale a que no lo tiene. Valor = false, cita = "". EXCEPCIÓN: Si se niega el dolor ("sin dolor"), el valor de dolor_eva DEBE ser 0, no null.
3. SEPARACIÓN TEMPORAL (POR CONTEXTO, NO POR TÍTULO): 
   - Las variables "ING_" se evalúan buscando la descripción del estado del paciente *al momento de llegar al hospital*, sin importar bajo qué título esté escrito.
   - Las variables "EVO_" se evalúan buscando eventos o estados que ocurrieron *durante los días de internación*.
4. TRAZABILIDAD FORENSE (COPY-PASTE): La 'cita' DEBE ser una extracción LITERAL (verbatim) del texto original (máximo 15 palabras). PROHIBIDO parafrasear o resumir. Si el valor es true, la cita no puede estar vacía.
5. CÁLCULO DE POLIFARMACIA: Estás autorizado a contar. Si en el plan o indicaciones se listan 5 o más fármacos diferentes administrados simultáneamente, marca LLM_polifarmacia como true.

DICCIONARIO DE VARIABLES Y CRITERIOS CLÍNICOS EXACTOS:

-- A. Estáticas (Evaluar en todo el texto histórico y actual):
- LLM_tabaquismo_activo: Fuma actualmente (excluye ex-fumadores sin recaída).
- LLM_polifarmacia: Toma 5+ fármacos simultáneos o el texto dice "polifarmacia".
- LLM_historial_caidas: Traumatismos o caídas ocurridas PREVIAMENTE al ingreso actual.
- LLM_abandono_medicacion: El paciente/familia dejó de tomar la medicación por decisión propia, mala adherencia o razones socioeconómicas. (NO aplica si la suspensión fue indicación médica).
- LLM_fragilidad_geriatrica: Mención literal de fragilidad, debilidad senil, deterioro senil, agotamiento por edad, sarcopenia o postración por edad.
- LLM_red_apoyo_deficiente: Vive solo, falta de cuidador principal, institucionalización previa (geriátrico/asilo), situación de calle, o mención de conflicto/abandono familiar.
- LLM_barrera_comprension: Barrera idiomática, hipoacusia severa, ceguera, o deterioro cognitivo leve/moderado que dificulta comprender pautas.

-- B. Dinámicas (Prefijo ING_ para Ingreso, EVO_ para Evolución):
- dolor_eva (entero 0-10): Extrae el valor numérico SOLO si se asocia explícitamente a la intensidad del dolor (ej. "EVA 7", "dolor 8/10"). ¡CUIDADO: No confundas fechas (ej. 7/10 como 7 de octubre) ni puntajes de Glasgow con el dolor! Si usa palabras, mapea: "sin dolor"=0, "leve"=2, "moderado"=5, "intenso"=8, "insoportable"=10. Si NO se menciona el dolor, usa el primitivo null estricto (sin comillas).
- gravedad_percibida (entero 1-10): Mapea el contexto clínico: 1-3 (Ambulatorio/Leve), 4-6 (Sala general estable), 7-8 (Cuidados Intermedios/Descompensado), 9-10 (UTI/Shock/Reanimación/Asistencia Respiratoria). Si no hay datos suficientes, usa null.
- alteracion_mental (bool): Mención de delirium, excitación, desorientación, confusión, obnubilación, letargo, sopor, coma, somnolencia excesiva o Glasgow < 15.
- dependencia_funcional (bool): Requiere asistencia para actividades básicas, paciente postrado, hemiplejía, cuadriplejía, paresia severa, o ACV secuelar motor.
- portador_dispositivos (bool): Sonda, colostomía, PICC/vía central, traqueostomía, catéteres, drenajes.

-- C. Exclusivas de Ingreso (Solo estado al momento de la admisión):
- ING_consultas_reiteradas (bool): Ya había consultado en días previos por este mismo episodio.
- ING_riesgo_hemorragico (bool): Sangrado activo, melena, hematemesis, epistaxis severa, plaquetopenia/trombocitopenia, coagulopatía o bajo anticoagulación activa.
- ING_infeccion_activa (bool): Mención de fiebre al ingreso, sospecha de sepsis, uso empírico de antibióticos desde la guardia o foco infeccioso claro (neumonía, ITU, celulitis).
- ING_multimorbilidad_severa (bool): Mención textual de 3 o más comorbilidades crónicas activas descompensadas simultáneamente (ej. "paciente diabético, hipertenso y con ERC reagudizada").

-- D. Exclusivas de Evolución (Solo eventos del curso de la internación):
- EVO_complicacion_internacion (bool): Infección intrahospitalaria, nueva caída, flebitis, intercurrencia nueva, shock, sepsis, descompensación hemodinámica, o necesidad de pase a UTI.
- EVO_aislamiento_infeccioso (bool): Aislamiento de contacto o respiratorio.
- EVO_cambio_terapeutico_mayor (bool): Inicio de insulina, anticoagulación, inotrópicos, o anticonvulsivantes durante la internación.
- EVO_intervencion_quirurgica (bool): Mención de paso por quirófano, cirugía, o procedimiento invasivo mayor (endoscopía, cateterismo).
- EVO_soporte_transfusional (bool): Requirió transfusión de hemoderivados durante su estadía.
- EVO_terapia_endovenosa_prolongada (bool): Requirió medicación endovenosa por más de 3 días consecutivos.
- EVO_inestabilidad_residual (bool): Alta con síntomas residuales documentados (ej. "febrícula", "disnea leve persistente").

FORMATO DE SALIDA OBLIGATORIO:
Devuelve un JSON válido donde CADA CLAVE es un objeto con 'valor' y 'cita'.
- Booleanos no mencionados/negados: {"valor": false, "cita": ""}
- Enteros no mencionados: {"valor": null, "cita": ""}
Claves esperadas:
LLM_tabaquismo_activo, LLM_polifarmacia, LLM_historial_caidas, LLM_abandono_medicacion, LLM_fragilidad_geriatrica, LLM_red_apoyo_deficiente, LLM_barrera_comprension, ING_dolor_eva, ING_gravedad_percibida, ING_alteracion_mental, ING_dependencia_funcional, ING_portador_dispositivos, ING_consultas_reiteradas, ING_riesgo_hemorragico, ING_infeccion_activa, ING_multimorbilidad_severa, EVO_dolor_eva, EVO_gravedad_percibida, EVO_alteracion_mental, EVO_dependencia_funcional, EVO_portador_dispositivos, EVO_complicacion_internacion, EVO_aislamiento_infeccioso, EVO_cambio_terapeutico_mayor, EVO_intervencion_quirurgica, EVO_soporte_transfusional, EVO_terapia_endovenosa_prolongada, EVO_inestabilidad_residual.

A continuación la historia clínica para auditar:
"""
                
                bloque_clinico = f"\n\n--- ESTADO AL INGRESO ---\n{ing_text}\n\n--- CURSO DE INTERNACIÓN ---\n{evo_text}"
                respuesta = modelo_nlp.generate_content(prompt_sistema + bloque_clinico)
                
                match = re.search(r'\{.*\}', respuesta.text, re.DOTALL)
                if match:
                    json_extraido = json_repair.loads(match.group(0))
                    
                    cro_activos, ing_activos, evo_activos = [], [], []
                    quotes = {}
                    
                    for k, item in json_extraido.items():
                        if isinstance(item, dict):
                            val = item.get('valor')
                            cita = str(item.get('cita', "")).strip()
                            
                            if val is not None and ('eva' in k or 'gravedad' in k):
                                try:
                                    num_val = int(float(val))
                                    if k == 'ING_dolor_eva': st.session_state.ui_ing_dolor = num_val
                                    if k == 'ING_gravedad_percibida': st.session_state.ui_ing_grav = num_val
                                    if k == 'EVO_dolor_eva': st.session_state.ui_evo_dolor = num_val
                                    if k == 'EVO_gravedad_percibida': st.session_state.ui_evo_grav = num_val
                                except: pass
                            elif val in [True, 1, 'true', 'yes']:
                                base_k = k.replace('LLM_', '').replace('ING_', '').replace('EVO_', '')
                                if "LLM_" in k: cro_activos.append(base_k)
                                elif "ING_" in k: ing_activos.append(base_k)
                                elif "EVO_" in k: evo_activos.append(base_k)
                                
                            if val and cita:
                                quotes[k] = cita
                                
                    st.session_state.ui_cro_sel = [k for k, v in cro_dict.items() if v in cro_activos]
                    st.session_state.ui_ing_sel = [k for k, v in ing_dict.items() if v in ing_activos]
                    st.session_state.ui_evo_sel = [k for k, v in evo_dict.items() if v in evo_activos]
                    st.session_state.nlp_quotes = quotes
                    st.session_state.nlp_processed = True
                    st.rerun() 
            except Exception as e:
                st.sidebar.error(f"NLP Extraction Error: {e}")

if st.session_state.nlp_processed and st.session_state.nlp_quotes:
    with st.sidebar.expander("📝 View Extracted Evidence", expanded=True):
        
        inv_cro = {v: k for k, v in cro_dict.items()}
        inv_ing = {v: k for k, v in ing_dict.items()}
        inv_evo = {v: k for k, v in evo_dict.items()}
        
        for var, quote in st.session_state.nlp_quotes.items():
            var_clean = var.replace("LLM_", "").replace("ING_", "").replace("EVO_", "")
            
            if var.startswith("LLM_"):
                nombre_base = inv_cro.get(var_clean, var_clean.replace('_', ' ').title())
                var_traducida = f"History: {nombre_base}"
                
            elif var.startswith("ING_"):
                if 'dolor' in var_clean: nombre_base = "Pain (VAS)"
                elif 'gravedad' in var_clean: nombre_base = "Perceived Severity"
                else: nombre_base = inv_ing.get(var_clean, var_clean.replace('_', ' ').title())
                var_traducida = f"Admission: {nombre_base}"
                
            elif var.startswith("EVO_"):
                if 'dolor' in var_clean: nombre_base = "Pain (VAS)"
                elif 'gravedad' in var_clean: nombre_base = "Perceived Severity"
                else: nombre_base = inv_evo.get(var_clean, var_clean.replace('_', ' ').title())
                var_traducida = f"Evolution: {nombre_base}"
                
            else:
                var_traducida = var_clean.replace('_', ' ').title()
            
            st.markdown(f"**{var_traducida}:**\n> *\"{quote}\"*")

st.sidebar.markdown("---")

# --- BLOQUE 3: REVISIÓN HUMANA (AUDITORÍA) ---
st.sidebar.subheader("3. Patient Background (Review)")

cronicos_seleccionados_ui = st.sidebar.multiselect("Chronic Conditions & Habits (LLM):", list(cro_dict.keys()), key="ui_cro_sel")
cronicos_seleccionados = [cro_dict[k] for k in cronicos_seleccionados_ui]

st.sidebar.subheader("4. Clinical Evolution (Review)")
c_ing, c_evo = st.sidebar.columns(2)

with c_ing:
    st.markdown("**At Admission (ING)**")
    ing_dolor = st.slider("Initial Pain", 0, 10, key="ui_ing_dolor")
    ing_grav = st.slider("Initial Severity", 1, 10, key="ui_ing_grav")
    ing_sel_ui = st.multiselect("Complications (ING):", list(ing_dict.keys()), key="ui_ing_sel")
    ing_sel = [ing_dict[k] for k in ing_sel_ui]

with c_evo:
    st.markdown("**At Discharge (EVO)**")
    evo_dolor = st.slider("Current Pain", 0, 10, key="ui_evo_dolor")
    evo_grav = st.slider("Current Severity", 1, 10, key="ui_evo_grav")
    evo_sel_ui = st.multiselect("Complications (EVO):", list(evo_dict.keys()), key="ui_evo_sel")
    evo_sel = [evo_dict[k] for k in evo_sel_ui]

# ==========================================
# 4. MATHEMATICAL ASSEMBLY ENGINE
# ==========================================
paciente_data = {}

variables_categoricas_train = [
    'rango_edad', 'PA_NIVEL', 'PA_SITLABO', 'Area', 
    'TR_Prioridad', 'IN_COMPLEJIDAD', 'sexo', 
    'CIE10_MACRO', 'CIE10_SUBMACRO', 'HIST_condicion_ultimo_egreso',
    'perfil_clinico_ingreso'
]

# Inicialización segura
for col in columnas_modelo:
    if col in variables_categoricas_train:
        paciente_data[col] = "DESCONOCIDO"
    else:
        paciente_data[col] = 0.0

if 'rango_edad' in paciente_data: paciente_data['rango_edad'] = rango_edad
if 'dias_internados' in paciente_data: paciente_data['dias_internados'] = float(dias_internados)
if 'pluripatologico' in paciente_data: paciente_data['pluripatologico'] = 1.0 if es_pluripatologico else 0.0

codigo_normalizado = normalizar_cie10(cie10_input)
categoria_cie10 = mapear_cie10_macro(codigo_normalizado)
if 'CIE10_MACRO' in paciente_data: paciente_data['CIE10_MACRO'] = categoria_cie10

for cro in cronicos_seleccionados: 
    if f"LLM_{cro}" in paciente_data: paciente_data[f"LLM_{cro}"] = 1.0

ing_dolor_val = float(ing_dolor)
ing_grav_val = float(ing_grav)
if 'ING_dolor_eva' in paciente_data: paciente_data['ING_dolor_eva'] = ing_dolor_val
if 'ING_gravedad_percibida' in paciente_data: paciente_data['ING_gravedad_percibida'] = ing_grav_val
for ing in ing_sel:
    if f"ING_{ing}" in paciente_data: paciente_data[f"ING_{ing}"] = 1.0

evo_dolor_val = float(evo_dolor)
evo_grav_val = float(evo_grav)
if 'EVO_dolor_eva' in paciente_data: paciente_data['EVO_dolor_eva'] = evo_dolor_val
if 'EVO_gravedad_percibida' in paciente_data: paciente_data['EVO_gravedad_percibida'] = evo_grav_val
for evo in evo_sel:
    if f"EVO_{evo}" in paciente_data: paciente_data[f"EVO_{evo}"] = 1.0

def calcular_delta_seguro(col_delta, col_evo, col_ing):
    if col_delta in paciente_data:
        val_evo = paciente_data.get(col_evo, 0.0)
        val_ing = paciente_data.get(col_ing, 0.0)
        paciente_data[col_delta] = val_evo - val_ing

calcular_delta_seguro('DELTA_dolor_eva', 'EVO_dolor_eva', 'ING_dolor_eva')
calcular_delta_seguro('DELTA_gravedad_percibida', 'EVO_gravedad_percibida', 'ING_gravedad_percibida')
calcular_delta_seguro('DELTA_alteracion_mental', 'EVO_alteracion_mental', 'ING_alteracion_mental')
calcular_delta_seguro('DELTA_dependencia_funcional', 'EVO_dependencia_funcional', 'ING_dependencia_funcional')
calcular_delta_seguro('DELTA_portador_dispositivos', 'EVO_portador_dispositivos', 'ING_portador_dispositivos')

# Inserción de variables administrativas y de riesgo
if 'sexo' in paciente_data: paciente_data['sexo'] = sexo_input
if 'Area' in paciente_data: paciente_data['Area'] = area_input
if 'perfil_clinico_ingreso' in paciente_data: paciente_data['perfil_clinico_ingreso'] = perfil_input
if 'IN_COMPLEJIDAD' in paciente_data: paciente_data['IN_COMPLEJIDAD'] = str(int(complejidad_input))
# --- NUEVA LÍNEA ---
if 'TR_Prioridad' in paciente_data: paciente_data['TR_Prioridad'] = prioridad_input
if 'cantidad_interconsultas' in paciente_data: paciente_data['cantidad_interconsultas'] = float(interconsultas_input)
if 'visitas_guardia_6meses_previos' in paciente_data: paciente_data['visitas_guardia_6meses_previos'] = float(visitas_guardia_input)
if 'EST_ingreso_ambulancia' in paciente_data: paciente_data['EST_ingreso_ambulancia'] = 1.0 if ingreso_ambulancia else 0.0
if 'EST_paso_por_uti' in paciente_data: paciente_data['EST_paso_por_uti'] = 1.0 if paso_por_uti else 0.0
if 'Riesgo_Cardiovasculares_Inotropicos' in paciente_data: paciente_data['Riesgo_Cardiovasculares_Inotropicos'] = 1.0 if med_cardio else 0.0
if 'Riesgo_Psicofarmacos_Neurologicos' in paciente_data: paciente_data['Riesgo_Psicofarmacos_Neurologicos'] = 1.0 if med_psico else 0.0

# Blindaje Final (Mayúsculas)
for col in variables_categoricas_train:
    if col in paciente_data:
        paciente_data[col] = str(paciente_data[col]).strip().upper()

df_paciente = pd.DataFrame([paciente_data])[columnas_modelo]

# ==========================================
# TABS & DASHBOARD
# ==========================================
tab_diagnostico, tab_estrategia, tab_evidencia, tab_umap, tab_eda = st.tabs([
    "📊 1. Current Risk & Audit", 
    "🧭 2. Stabilization & Simulation", 
    "🧬 3. Cohort & Inspector (KNN)",
    "🌌 4. Global Universe (UMAP)",
    "📈 5. Exploratory Data"
])

with tab_diagnostico:
    riesgo = pipeline.predict_proba(df_paciente)[0][1]
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns([1, 1.2, 1.5])
    
    with col_kpi1:
        st.subheader("Readmission Risk")
        st.metric(label="15-Day Probability", value=f"{riesgo*100:.1f}%")
        
    with col_kpi2:
        st.write("") 
        if riesgo > umbral:
            st.error(f"⚠️ **CLINICAL ALERT**\n\nExceeds safety threshold ({umbral*100:.1f}%)")
        else:
            st.success(f"✅ **SAFE DISCHARGE**\n\nWithin permitted threshold")
            
    with col_kpi3:
        st.write("")
        cie10_ui_dict = {
            "Tuberculosis": "Tuberculosis", "Lepra": "Leprosy", "Sífilis": "Syphilis", 
            "Otras infecciosas (A)": "Other infectious (A)", "Hepatitis viral": "Viral hepatitis", 
            "Enfermedad por VIH": "HIV disease", "Enfermedad de Chagas": "Chagas disease", 
            "Toxoplasmosis": "Toxoplasmosis", "Equinococosis / Hidatidosis": "Echinococcosis / Hydatidosis", 
            "Secuelas de enfermedades infecciosas": "Sequelae of infectious diseases", "Otras infecciosas (B)": "Other infectious (B)",
            "Cáncer de labio / boca / faringe": "Lip / mouth / pharynx cancer", "Cáncer digestivo": "Digestive cancer", 
            "Cáncer respiratorio / intratorácico": "Respiratory / intrathoracic cancer", "Cáncer de hueso / cartílago": "Bone / cartilage cancer", 
            "Melanoma / Cáncer de piel": "Melanoma / Skin cancer", "Cáncer de mama": "Breast cancer", 
            "Cáncer genital femenino": "Female genital cancer", "Cáncer genital masculino": "Male genital cancer", 
            "Cáncer de vías urinarias": "Urinary tract cancer", "Cáncer de sistema nervioso central": "Central nervous system cancer", 
            "Cáncer linfoide / hematopoyético": "Lymphoid / hematopoietic cancer", "Otros tumores malignos": "Other malignant tumors", 
            "Tumores in situ o benignos": "In situ or benign tumors", "Anemias nutricionales": "Nutritional anemias", 
            "Anemias hemolíticas": "Hemolytic anemias", "Aplasias y otras anemias": "Aplasias and other anemias", 
            "Defectos de coagulación / púrpura": "Coagulation defects / purpura", "Trastornos de inmunodeficiencia": "Immunodeficiency disorders", 
            "Otros trastornos de la sangre": "Other blood disorders",
            "Tiroides": "Thyroid", "Diabetes": "Diabetes", "Glucosa / hipoglucemia": "Glucose / hypoglycemia", 
            "Otros endocrinos y metabólicos": "Other endocrine and metabolic", "Obesidad y trastornos de hiperalimentación": "Obesity and hyperalimentation disorders", 
            "Dislipidemia": "Dyslipidemia", "Fibrosis quística": "Cystic fibrosis", "Trastornos metabólicos": "Metabolic disorders", 
            "Otros metabólicos / nutricionales": "Other metabolic / nutritional",
            "Trastornos mentales orgánicos (Demencias)": "Organic mental disorders (Dementias)", "Trastornos por uso de sustancias": "Substance use disorders", 
            "Esquizofrenia y trastornos psicóticos": "Schizophrenia and psychotic disorders", "Trastornos del humor (Afectivos)": "Mood (Affective) disorders", 
            "Trastornos neuróticos y de ansiedad": "Neurotic and anxiety disorders", "Trastornos de la conducta alimentaria / sueño": "Eating / sleep disorders", 
            "Trastornos de la personalidad": "Personality disorders", "Discapacidad intelectual": "Intellectual disability", 
            "Trastornos del desarrollo psicobiológico (Autismo)": "Psychobiological development disorders (Autism)", "Otros trastornos mentales": "Other mental disorders",
            "Atrofias sistémicas del SNC": "Systemic atrophies of CNS", "Trastornos extrapiramidales y del movimiento (Parkinson)": "Extrapyramidal and movement disorders (Parkinson's)", 
            "Enfermedades degenerativas (Alzheimer)": "Degenerative diseases (Alzheimer's)", "Enfermedades desmielinizantes (Esclerosis Múltiple)": "Demyelinating diseases (Multiple Sclerosis)", 
            "Trastornos episódicos y paroxísticos (Epilepsia, Migraña)": "Episodic and paroxysmal disorders (Epilepsy, Migraine)", "Trastornos de nervios y plexos": "Nerve and plexus disorders", 
            "Polineuropatías": "Polyneuropathies", "Enfermedades de la unión neuromuscular (Miastenia)": "Diseases of the neuromuscular junction (Myasthenia)", 
            "Parálisis cerebral y síndromes paralíticos": "Cerebral palsy and paralytic syndromes", "Otros trastornos neurológicos": "Other neurological disorders",
            "Ojo": "Eye", "Oído": "Ear", "Otros órganos de los sentidos": "Other sense organs",
            "Hipertensión": "Hypertension", "Cardiopatía isquémica": "Ischemic heart disease", "Enfermedad cardiopulmonar": "Cardiopulmonary disease", 
            "Otras enfermedades del corazón (Insuficiencia Cardíaca)": "Other heart diseases (Heart Failure)", "Cerebrovascular": "Cerebrovascular", 
            "Enfermedades de arterias y capilares": "Diseases of arteries and capillaries", "Enfermedades de venas y vasos linfáticos": "Diseases of veins and lymphatic vessels", 
            "Otros circulatorios": "Other circulatory",
            "Vías respiratorias altas": "Upper respiratory tract", "Infecciones agudas / neumonía / influenza": "Acute infections / pneumonia / influenza", 
            "Infecciones respiratorias bajas": "Lower respiratory infections", "Enfermedades de vías respiratorias superiores": "Diseases of upper respiratory tract", 
            "Asma / EPOC / bronquitis": "Asthma / COPD / bronchitis", "Enfermedades del pulmón por agentes externos (Neumoconiosis)": "Lung diseases due to external agents (Pneumoconiosis)", 
            "Enfermedades pulmonares intersticiales": "Interstitial lung diseases", "Otros respiratorios": "Other respiratory",
            "Boca / dientes / faringe": "Mouth / teeth / pharynx", "Esófago / estómago / duodeno": "Esophagus / stomach / duodenum", 
            "Apendicitis": "Appendicitis", "Hernias": "Hernias", "Enfermedad de Crohn y colitis": "Crohn's disease and colitis", 
            "Otras enfermedades de los intestinos": "Other diseases of the intestines", "Hígado": "Liver", 
            "Vesícula / vías biliares / páncreas": "Gallbladder / biliary tract / pancreas", "Otros digestivos": "Other digestive",
            "Dermatitis y eczema": "Dermatitis and eczema", "Trastornos papuloescamosos (Psoriasis)": "Papulosquamous disorders (Psoriasis)", 
            "Urticaria y eritema": "Urticaria and erythema", "Trastornos de las faneras / Otros trastornos de piel": "Disorders of skin appendages / Other skin disorders", 
            "Otras enfermedades de la piel": "Other skin diseases",
            "Artropatías": "Arthropathies", "Tejido conectivo (Lupus, etc.)": "Connective tissue (Lupus, etc.)", "Dorsopatías": "Dorsopathies", 
            "Tejidos blandos": "Soft tissues", "Osteopatías y condropatías (Osteoporosis)": "Osteopathies and chondropathies (Osteoporosis)", 
            "Otros osteomusculares": "Other musculoskeletal",
            "Riñón (Insuficiencia Renal Crónica)": "Kidney (Chronic Renal Failure)", "Vías urinarias bajas": "Lower urinary tract", 
            "Genital masculino (Hiperplasia Prostática)": "Male genital (Prostatic Hyperplasia)", "Mama": "Breast", 
            "Genital femenino (Endometriosis, etc.)": "Female genital (Endometriosis, etc.)", "Otros genitourinarios": "Other genitourinary",
            "Malformaciones del sistema nervioso (Espina bífida)": "Malformations of the nervous system (Spina bifida)", "Malformaciones cardíacas congénitas": "Congenital heart malformations", 
            "Anomalías cromosómicas (Síndrome de Down)": "Chromosomal abnormalities (Down Syndrome)", "Otras malformaciones congénitas": "Other congenital malformations",
            "Enfermedad respiratoria crónica perinatal": "Chronic perinatal respiratory disease", 
            "Secuelas crónicas de traumatismos": "Chronic sequelae of injuries",
            "Síndrome Post-COVID (Long COVID)": "Post-COVID Syndrome (Long COVID)", "Otras condiciones especiales (U)": "Other special conditions (U)",
            "Historia personal de tumores / enfermedades": "Personal history of tumors / diseases", "Ausencia adquirida de miembros / órganos": "Acquired absence of limbs / organs", 
            "Aberturas artificiales (Ostomías)": "Artificial openings (Ostomies)", "Estado de órgano trasplantado": "Transplanted organ status", 
            "Presencia de implantes cardíacos / vasculares": "Presence of cardiac / vascular implants", "Dependencia de máquinas (diálisis, oxígeno)": "Machine dependence (dialysis, oxygen)", 
            "Otros factores de salud": "Other health factors",
            "DESCONOCIDO": "UNKNOWN"
        }
        
        if pd.isna(categoria_cie10):
            categoria_cie10_ingles = "N/A"
        else:
            categoria_cie10_ingles = cie10_ui_dict.get(categoria_cie10, categoria_cie10)
        
        st.info(f"**Mapped Diagnosis:**\n\n{categoria_cie10_ingles}\n\n(Code: {codigo_normalizado})")
    
    st.markdown("---")
    st.subheader("Decision Audit & Clinical Context")
    
    col_shap, col_trayectoria = st.columns(2)
    
    with col_shap:
        with st.container():
            st.markdown("#### 🔍 1. Prescriptive Explainability (SHAP)")
            filtrar_activos = st.checkbox("🎯 Show only the impact of present conditions", value=False)
            
            try:
                clf = pipeline.named_steps['clasificador']
                prep = pipeline.named_steps['preprocesador']
                has_selector = 'feature_selection' in pipeline.named_steps
                selector = pipeline.named_steps['feature_selection'] if has_selector else None
                
                X_proc = prep.transform(df_paciente)
                if selector:
                    X_proc = selector.transform(X_proc)
                    
                X_proc_dense = X_proc.toarray() if hasattr(X_proc, 'toarray') else np.array(X_proc)
                
                if selector:
                    try:
                        nombres_crudos = selector.get_feature_names_out()
                    except Exception:
                        nombres_crudos = [f"Feature_{i}" for i in range(X_proc_dense.shape[1])]
                else:
                    nombres_crudos = prep.get_feature_names_out()
                
                shap_ui_dict = {
                    'dias_internados': 'Hospitalization Days', 'pluripatologico': 'Multimorbidity',
                    'ING_dolor_eva': 'Initial Pain', 'ING_gravedad_percibida': 'Initial Severity',
                    'EVO_dolor_eva': 'Current Pain', 'EVO_gravedad_percibida': 'Current Severity',
                    'DELTA_dolor_eva': 'Pain Delta', 'DELTA_gravedad_percibida': 'Severity Delta',
                    'DELTA_alteracion_mental': 'Mental Alt. Delta', 'DELTA_dependencia_funcional': 'Func. Dep. Delta',
                    'DELTA_portador_dispositivos': 'Device Bearer Delta', 'ING_alteracion_mental': 'Initial Mental Alt.',
                    'ING_consultas_reiteradas': 'Initial Repeated Consults', 'ING_dependencia_funcional': 'Initial Func. Dep.',
                    'ING_portador_dispositivos': 'Initial Device Bearer', 'ING_riesgo_hemorragico': 'Initial Hemorrhagic Risk',
                    'ING_infeccion_activa': 'Initial Active Infection', 'ING_multimorbilidad_severa': 'Initial Sev. Multimorbidity',
                    'EVO_aislamiento_infeccioso': 'Current Infect. Isolation', 'EVO_alteracion_mental': 'Current Mental Alt.',
                    'EVO_complicacion_internacion': 'Current Hosp. Complication',
                    'EVO_dependencia_funcional': 'Current Func. Dep.',
                    'EVO_portador_dispositivos': 'Current Device Bearer',
                    'EVO_cambio_terapeutico_mayor': 'Current Major Ther. Change',
                    'EVO_intervencion_quirurgica': 'Current Surgical Interv.',
                    'EVO_soporte_transfusional': 'Current Transfusion Support',
                    'EVO_terapia_endovenosa_prolongada': 'Current Prolonged IV',
                    'EVO_inestabilidad_residual': 'Current Residual Instab.',
                    'LLM_abandono_medicacion': 'Chronic: Med. Abandonment',
                    'LLM_historial_caidas': 'Chronic: History of Falls', 'LLM_fragilidad_geriatrica': 'Chronic: Geriatric Frailty',
                    'LLM_red_apoyo_deficiente': 'Social: Poor Support Net', 'LLM_barrera_comprension': 'Social: Comp. Barrier',
                    'LLM_polifarmacia': 'Chronic: Polypharmacy', 'LLM_tabaquismo_activo': 'Chronic: Active Smoking',
                    'sexo': 'Sex', 'Area': 'Admission Area', 'IN_COMPLEJIDAD': 'Complexity Level','TR_Prioridad': 'Triage Priority',
                    'cantidad_interconsultas': 'Interconsultations', 'visitas_guardia_6meses_previos': 'ER Visits (6m)',
                    'EST_ingreso_ambulancia': 'Ambulance Arrival', 'perfil_clinico_ingreso': 'Admission Profile',
                    'EST_paso_por_uti': 'ICU Stay', 'Riesgo_Cardiovasculares_Inotropicos': 'High-Risk Med: Cardiovascular',
                    'Riesgo_Psicofarmacos_Neurologicos': 'High-Risk Med: Neuro/Psycho'
                }
                
                nombres_limpios_traducidos = []
                human_data = [] 
                cie10_upper = {k.upper(): v for k, v in cie10_ui_dict.items()}
                
                for nombre_crudo in nombres_crudos:
                    traducido = nombre_crudo.split('__')[-1]
                    nombre_base = traducido 
                    
                    for sufijo in ['_1.0', '_1', '_True', '_true', '_0.0', '_0', '_False', '_false']:
                        if traducido.endswith(sufijo):
                            traducido = traducido[:-len(sufijo)]
                            nombre_base = nombre_base[:-len(sufijo)]
                            break
                    
                    if "CIE10_MACRO" in nombre_crudo:
                        cat_es = traducido.replace("CIE10_MACRO_", "")
                        cat_en = cie10_upper.get(cat_es.upper(), cat_es.replace('_', ' ').title())
                        traducido = f"Diagnosis: {cat_en}"
                    elif "rango_edad" in nombre_crudo:
                        cat_es = traducido.replace("rango_edad_", "")
                        match_en = "Unknown Age"
                        for en_k, es_v in opciones_edad_dict.items():
                            if es_v.upper().replace(' ', '_') == cat_es.upper() or es_v.upper() == cat_es.upper():
                                match_en = en_k
                                break
                        traducido = f"Age: {match_en}"
                    elif "perfil_clinico_ingreso" in nombre_crudo:
                        cat_es = traducido.replace("perfil_clinico_ingreso_", "")
                        match_en = "Unknown Profile"
                        for en_k, es_v in perfil_clinico_map.items():
                            if es_v.upper().replace(' ', '_') == cat_es.upper() or es_v.upper() == cat_es.upper():
                                match_en = en_k
                                break
                        traducido = f"Profile: {match_en}"
                    else:
                        match_encontrado = False
                        for var_es, var_en in shap_ui_dict.items():
                            if var_es in nombre_crudo:
                                traducido = var_en
                                match_encontrado = True
                                break
                        if not match_encontrado:
                            traducido = traducido.replace("_", " ").title()
                            
                    nombres_limpios_traducidos.append(traducido)
                    
                    if nombre_base in df_paciente.columns:
                        val = df_paciente[nombre_base].iloc[0]
                        try:
                            val = float(val)
                            if val.is_integer(): val = int(val)
                        except:
                            pass
                        human_data.append(val)
                    else:
                        val_real = 0
                        columnas_categoricas = ['CIE10_MACRO', 'CIE10_SUBMACRO', 'rango_edad', 'PA_NIVEL', 'PA_SITLABO', 'Area', 'TR_Prioridad', 'IN_COMPLEJIDAD', 'sexo', 'perfil_clinico_ingreso']
                        for col_cat in columnas_categoricas:
                            if nombre_base.startswith(col_cat + '_'):
                                valor_categoria_columna = nombre_base.replace(col_cat + '_', '')
                                if col_cat in df_paciente.columns:
                                    valor_real_paciente = str(df_paciente[col_cat].iloc[0])
                                    if valor_categoria_columna.strip().upper() == valor_real_paciente.strip().upper():
                                        val_real = 1
                                break
                        human_data.append(val_real)
                
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_proc_dense, check_additivity=False)
                except Exception:
                    explainer = shap.LinearExplainer(clf, X_proc_dense) if hasattr(clf, 'coef_') else shap.Explainer(clf, X_proc_dense)
                    shap_vals = explainer(X_proc_dense)
                
                shap_array = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals
                if isinstance(shap_array, list): 
                    shap_array = shap_array[1] if len(shap_array) > 1 else shap_array[0]
                if len(shap_array.shape) > 2: 
                    shap_array = shap_array[:, :, 1]
                
                exp_val = explainer.expected_value
                if hasattr(exp_val, 'values'): exp_val = exp_val.values
                if isinstance(exp_val, (list, np.ndarray)):
                    exp_val = exp_val[1] if len(exp_val) > 1 else exp_val[0]
                
                shap_vals_pct = shap_array[0] * 100
                exp_val_pct = exp_val * 100
    
                if not filtrar_activos:
                    explicacion_completa = shap.Explanation(
                        values=np.array(shap_vals_pct), 
                        base_values=exp_val_pct, 
                        data=np.array(human_data), 
                        feature_names=nombres_limpios_traducidos
                    )
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(12, 4.5)) 
                    shap.waterfall_plot(explicacion_completa, show=False, max_display=10) 
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
                    
                else:
                    indices_activos = []
                    variables_continuas = ['Days', 'Pain', 'Severity', 'Delta', 'Consultations', 'Complexity']
    
                    for i, (val_real, nombre_traducido) in enumerate(zip(human_data, nombres_limpios_traducidos)):
                        es_continua = any(kw in nombre_traducido for kw in variables_continuas)
                        es_inactivo = False
                        
                        if not es_continua:
                            val_str = str(val_real).strip().upper()
                            if val_str in ['0', '0.0', 'FALSE', 'NONE', 'N/A', 'NAN', '']: es_inactivo = True
                            
                        if abs(shap_vals_pct[i]) < 0.01: es_inactivo = True
                        
                        if not es_inactivo: indices_activos.append(i)
    
                    if not indices_activos:
                        st.info("No significant active clinical factors to isolate.")
                    else:
                        activos_vals = [shap_vals_pct[i] for i in indices_activos]
                        activos_nombres = [nombres_limpios_traducidos[i] for i in indices_activos]
                        
                        datos_ordenados = sorted(zip(activos_vals, activos_nombres), key=lambda x: abs(x[0]))
                        y_vals = [x[0] for x in datos_ordenados]
                        y_names = [x[1] for x in datos_ordenados]
                        
                        fig_bar = go.Figure(go.Bar(
                            x=y_vals, y=y_names, orientation='h',
                            marker_color=['#FF4444' if v > 0 else '#00C851' for v in y_vals],
                            hoverinfo='none' 
                        ))
                        
                        fig_bar.update_layout(
                            title="Isolation of Present Clinical Factors",
                            xaxis_title="Relative Impact Weight", 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=max(350, len(y_names) * 45),
                            margin=dict(l=10, r=40, t=40, b=10),
                            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(128,128,128,0.6)', showticklabels=False),
                            yaxis=dict(showgrid=False)
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
    
            except Exception as e:
                st.error("SHAP computation failed.")
                st.warning(str(e))

    with col_trayectoria:
        with st.container():
            st.markdown("#### 📉 2. Dynamic Trajectory")
            try:
                df_row = df_paciente.iloc[[0]].copy()
                pares_clinicos = {
                    'Pain (VAS)': ('ING_dolor_eva', 'EVO_dolor_eva'),
                    'Severity': ('ING_gravedad_percibida', 'EVO_gravedad_percibida'),
                    'Mental Alt.': ('ING_alteracion_mental', 'EVO_alteracion_mental'),
                    'Func. Dep.': ('ING_dependencia_funcional', 'EVO_dependencia_funcional'),
                    'Devices': ('ING_portador_dispositivos', 'EVO_portador_dispositivos')
                }
                
                fig_slope = go.Figure()
                y_ing_coords = []
                y_evo_coords = []
                
                for label, (col_ing, col_evo) in pares_clinicos.items():
                    val_ing = float(df_row[col_ing].values[0]) if col_ing in df_row.columns else 0.0
                    val_evo = float(df_row[col_evo].values[0]) if col_evo in df_row.columns else 0.0
                    
                    y_ing_coords.append(val_ing)
                    y_evo_coords.append(val_evo)
                    
                    color_linea = '#00C851' if val_evo <= val_ing else '#FF4444'
                    
                    fig_slope.add_trace(go.Scatter(
                        x=['Admission', 'Current'], y=[val_ing, val_evo],
                        mode='lines+markers',
                        line=dict(color=color_linea, width=4),
                        marker=dict(size=12, color=color_linea, line=dict(color='white', width=1)),
                        name=label,
                        hoverinfo='text',
                        hovertext=f"<b>{label}</b><br>Admission: {val_ing:.1f}<br>Current: {val_evo:.1f}"
                    ))
    
                def separar_superposiciones(valores, margen=0.45):
                    ordenados = sorted(enumerate(valores), key=lambda x: x[1])
                    res = {}
                    if not ordenados: return res
                    res[ordenados[0][0]] = ordenados[0][1]
                    last_y = ordenados[0][1]
                    for idx, y in ordenados[1:]:
                        nuevo_y = last_y + margen if y < last_y + margen else y
                        res[idx] = nuevo_y
                        last_y = nuevo_y
                    desplazamiento = (sum(res.values()) - sum(valores)) / len(valores) if valores else 0
                    return {k: v - desplazamiento for k, v in res.items()}
    
                textos_ing_y = separar_superposiciones(y_ing_coords)
                textos_evo_y = separar_superposiciones(y_evo_coords)
    
                for i, (label, _) in enumerate(pares_clinicos.items()):
                    val_ing = y_ing_coords[i]
                    val_evo = y_evo_coords[i]
                    color_linea = '#00C851' if val_evo <= val_ing else '#FF4444'
                    
                    fig_slope.add_annotation(
                        x='Admission', y=textos_ing_y[i], text=f"{label} ({val_ing:.1f})",
                        showarrow=False, xanchor='right', xshift=-15,
                        font=dict(size=12) 
                    )
                    fig_slope.add_annotation(
                        x='Current', y=textos_evo_y[i], text=f"({val_evo:.1f}) {label}",
                        showarrow=False, xanchor='left', xshift=15,
                        font=dict(size=12, color=color_linea) 
                    )
    
                fig_slope.add_vline(x='Admission', line_width=1.5, line_dash="dash", line_color="rgba(128,128,128,0.4)")
                fig_slope.add_vline(x='Current', line_width=1.5, line_dash="dash", line_color="rgba(128,128,128,0.4)")
    
                fig_slope.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    height=350, margin=dict(l=100, r=100, t=30, b=20),
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=14, weight='bold')),
                    yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False)
                )
                st.plotly_chart(fig_slope, use_container_width=True)
                
            except Exception as e:
                st.error("Trajectory unavailable.")
                st.warning(str(e))

# ==========================================================================
# tab_estrategia — VERSIÓN CORREGIDA
# --------------------------------------------------------------------------
# Requiere las mismas variables externas que tu script original:
#   st, pd, np, go, os, dice_ml, shap, pipeline, df_paciente, umbral, riesgo
#
# Cambios respecto a la versión original:
#   1. Las variables EVO_ binarias se declaran CATEGÓRICAS para DiCE
#      (no se incluyen en continuous_features) -> DiCE ya no genera 0.42,
#      solo muestrea 0/1.
#   2. Se fija continuous_features_precision=0 para las variables que en
#      la práctica clínica son enteras (dias_internados, dolor, gravedad).
#      Esto alinea la validación INTERNA de DiCE con el redondeo que
#      mostramos después -> elimina el "salto" de precisión.
#   3. El umbral de búsqueda y el de mitigación ya NO son una constante
#      absoluta (0.5): son siempre relativos al umbral clínico real,
#      con escalado graduado (varios intentos cada vez más permisivos).
#   4. VALIDACIÓN FINAL: cada ruta candidata, ya redondeada, se vuelve a
#      pasar por el pipeline REAL (no el calibrado) antes de mostrarse.
#      Si no supera la validación, se descarta. Lo que el médico ve en
#      pantalla es el mismo cálculo que el Sandbox de abajo.
#   5. La sincronización de deltas (EVO - ING) vive en UNA sola función
#      (sincronizar_deltas) reusada por ModeloSincronizado, por la
#      validación de rutas y por el motor SHAP -> una sola fuente de verdad.
# ==========================================================================

with tab_estrategia:
    # ----------------------------------------------------------------
    # HELPERS COMPARTIDOS (una sola fuente de verdad para los deltas)
    # ----------------------------------------------------------------
    PARES_DELTA = {
        'DELTA_dolor_eva': ('EVO_dolor_eva', 'ING_dolor_eva'),
        'DELTA_gravedad_percibida': ('EVO_gravedad_percibida', 'ING_gravedad_percibida'),
        'DELTA_alteracion_mental': ('EVO_alteracion_mental', 'ING_alteracion_mental'),
        'DELTA_dependencia_funcional': ('EVO_dependencia_funcional', 'ING_dependencia_funcional'),
        'DELTA_portador_dispositivos': ('EVO_portador_dispositivos', 'ING_portador_dispositivos'),
    }

    def sincronizar_deltas(df):
        """Recalcula las columnas DELTA_* in-place a partir de EVO_ - ING_."""
        for col_delta, (col_evo, col_ing) in PARES_DELTA.items():
            if col_evo in df.columns and col_ing in df.columns:
                df[col_delta] = df[col_evo] - df[col_ing]
        return df

    def construir_fila_simulada(df_base, overrides=None):
        """Clona la fila base del paciente, aplica overrides y resincroniza deltas."""
        fila = df_base.copy()
        if overrides:
            for col, val in overrides.items():
                fila[col] = val
        return sincronizar_deltas(fila)

    col_simulador, col_rutas = st.columns([1, 1.2])

    # ==================================================================
    # COLUMNA IZQUIERDA: SANDBOX (sin cambios funcionales)
    # ==================================================================
    with col_simulador:
        st.markdown("---")
        st.markdown("#### 🧪 Clinical Hypothesis Simulator (SandBox)")

        with st.expander("🛠️ Configure Stabilization Scenario", expanded=True):
            dias_base = int(float(df_paciente['dias_internados'].iloc[0] if 'dias_internados' in df_paciente.columns else 1.0))
            max_permitido = max(dias_base + 20, 30)

            dias_sim = st.slider(
                label="Hospitalization Stay (Days):",
                min_value=0, max_value=max_permitido, value=dias_base, step=1,
                key=f"sim_dias_base_{dias_base}",
                help="Drag left to simulate premature discharge or right to project extended stay impacts."
            )
            st.markdown("---")

            st.markdown("**Continuous Evolution Metrics** - *Simulate symptom progression*")
            col_dolor, col_severidad = st.columns(2)

            with col_dolor:
                dolor_crudo = df_paciente['EVO_dolor_eva'].iloc[0] if 'EVO_dolor_eva' in df_paciente.columns else 0.0
                dolor_base = int(float(dolor_crudo))
                dolor_sim = st.slider("Current Pain Level (VAS 0-10):", min_value=0, max_value=10, value=dolor_base, step=1, key=f"sim_dolor_base_{dolor_base}")

            with col_severidad:
                sev_crudo = df_paciente['EVO_gravedad_percibida'].iloc[0] if 'EVO_gravedad_percibida' in df_paciente.columns else 0.0
                sev_base = int(float(sev_crudo))
                severidad_sim = st.slider("Current Perceived Severity (0-10):", min_value=0, max_value=10, value=sev_base, step=1, key=f"sim_sev_base_{sev_base}")

            st.markdown("---")
            st.markdown("**Evolution Status (EVO)** - *Toggle acquired complications or resolved states*")
            sim_evo_map = {
                'Mental Alteration': 'EVO_alteracion_mental',
                'Functional Dependency': 'EVO_dependencia_funcional',
                'Medical Devices': 'EVO_portador_dispositivos',
                'Infectious Isolation': 'EVO_aislamiento_infeccioso',
                'Hosp. Complication': 'EVO_complicacion_internacion',
                'Major Ther. Change': 'EVO_cambio_terapeutico_mayor',
                'Surgical Interv.': 'EVO_intervencion_quirurgica',
                'Transfusion Support': 'EVO_soporte_transfusional',
                'Prolonged IV Therapy': 'EVO_terapia_endovenosa_prolongada',
                'Residual Instability': 'EVO_inestabilidad_residual'
            }

            cols_evo = st.columns(4)
            status_evo_sim = {}

            for i, (label, col) in enumerate(sim_evo_map.items()):
                with cols_evo[i % 4]:
                    val_raw = df_paciente[col].iloc[0] if col in df_paciente.columns else 0
                    val_str = str(val_raw).strip().upper()
                    val_init = True if val_str in ['1', '1.0', 'TRUE', 'YES'] else False
                    status_evo_sim[col] = st.toggle(label, value=val_init, key=f"sim_{col}_base_{val_init}")

    # ==================================================================
    # COLUMNA DERECHA: RUTAS DE ESTABILIZACIÓN (DiCE) — CORREGIDO
    # ==================================================================
    with col_rutas:
        st.markdown("---")
        st.markdown("#### Clinical Stabilization Routes")

        class ModeloSincronizado:
            """Recalibra la probabilidad real para que el corte 'umbral' quede en 0.5,
            como exige el criterio desired_class='opposite' de DiCE."""

            def __init__(self, pipeline_original, columnas_modelo, umbral_real):
                self.pipeline = pipeline_original
                self.columnas_modelo = columnas_modelo
                self.umbral = umbral_real

            def predict_proba(self, X):
                if isinstance(X, np.ndarray):
                    X_sync = pd.DataFrame(X, columns=self.columnas_modelo)
                else:
                    X_sync = X.copy()

                sincronizar_deltas(X_sync)  # única fuente de verdad

                probas_originales = self.pipeline.predict_proba(X_sync)
                probas_calibradas = np.zeros_like(probas_originales)
                p1 = probas_originales[:, 1]

                mask_low = p1 <= self.umbral
                mask_high = p1 > self.umbral

                p1_calib = np.zeros_like(p1)
                if np.any(mask_low):
                    p1_calib[mask_low] = p1[mask_low] * (0.5 / self.umbral)
                if np.any(mask_high):
                    p1_calib[mask_high] = 0.5 + ((p1[mask_high] - self.umbral) * (0.5 / (1.0 - self.umbral)))

                probas_calibradas[:, 1] = p1_calib
                probas_calibradas[:, 0] = 1.0 - p1_calib

                return probas_calibradas

        if riesgo <= umbral:
            st.info("The patient is in optimal condition for discharge. No stabilization targets required.")
        else:
            with st.spinner("Calculating multiple clinically viable stabilization routes..."):
                try:
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                    ruta_ext = os.path.join(BASE_DIR, 'matriz_extended_display_llm.npy')
                    ruta_cols = os.path.join(BASE_DIR, 'columnas_display_llm.npy')

                    if not os.path.exists(ruta_ext) or not os.path.exists(ruta_cols):
                        raise FileNotFoundError("Historical data matrices are missing from the server volume.")

                    matriz_extended = np.load(ruta_ext, allow_pickle=True)
                    nombres_columnas = np.load(ruta_cols, allow_pickle=True)

                    df_background_raw = pd.DataFrame(matriz_extended, columns=nombres_columnas)
                    columnas_modelo = df_paciente.columns.tolist()

                    for col in columnas_modelo:
                        if col not in df_background_raw.columns:
                            df_background_raw[col] = 0.0

                    df_dice_train = df_background_raw[columnas_modelo].copy()
                    df_dice_train['target'] = df_background_raw['target'].astype(int)

                    for col in columnas_modelo:
                        if df_paciente[col].dtype == object:
                            df_dice_train[col] = df_background_raw[col].astype(str).str.strip().str.upper()
                        else:
                            df_dice_train[col] = pd.to_numeric(df_background_raw[col], errors='coerce')

                    df_dice_train = df_dice_train.sample(n=min(1000, len(df_dice_train)), random_state=42)

                    df_paciente_para_dice = df_paciente.copy()
                    df_paciente_para_dice['target'] = 1
                    df_dice_train = pd.concat([df_dice_train, df_paciente_para_dice], ignore_index=True)

                    # ----------------------------------------------------------------
                    # FIX 1: separar correctamente continuas vs categóricas + precisión
                    # ----------------------------------------------------------------
                    variables_accionables = [col for col in columnas_modelo if col.startswith('EVO_')]

                    features_binarias_evo = [
                        col for col in variables_accionables
                        if 'dolor' not in col and 'gravedad' not in col
                    ]

                    features_continuas = [
                        col for col in df_paciente.select_dtypes(include=[np.number]).columns.tolist()
                        if col not in features_binarias_evo
                    ]

                    precisiones_enteras = {
                        'dias_internados': 0,
                        'EVO_dolor_eva': 0,
                        'EVO_gravedad_percibida': 0,
                        'ING_dolor_eva': 0,
                        'ING_gravedad_percibida': 0,
                    }
                    precisiones_aplicables = {
                        k: v for k, v in precisiones_enteras.items() if k in features_continuas
                    }

                    d = dice_ml.Data(
                        dataframe=df_dice_train,
                        continuous_features=features_continuas,
                        outcome_name='target',
                        continuous_features_precision=precisiones_aplicables
                    )

                    rangos_permitidos = {}
                    vars_a_variar = []

                    for col in variables_accionables:
                        val_actual = df_paciente[col].iloc[0]

                        if 'gravedad' in col:
                            if val_actual > 1.0:
                                rangos_permitidos[col] = [1.0, float(val_actual)]
                                vars_a_variar.append(col)
                        elif 'dolor' in col:
                            if val_actual > 0.0:
                                rangos_permitidos[col] = [0.0, float(val_actual)]
                                vars_a_variar.append(col)
                        else:
                            if val_actual == 1.0:
                                rangos_permitidos[col] = [0, 1]
                                vars_a_variar.append(col)

                    if 'dias_internados' in df_paciente.columns:
                        val_dias = float(df_paciente['dias_internados'].iloc[0])
                        rangos_permitidos['dias_internados'] = [val_dias, val_dias + 7.0]
                        vars_a_variar.append('dias_internados')

                    if not vars_a_variar:
                        st.error("There are no modifiable clinical targets in the patient's current evolution that can improve their condition.")
                    else:
                        # ----------------------------------------------------------------
                        # FIX 2 + 3: umbrales relativos (no 0.5 fijo) + validación final
                        # ----------------------------------------------------------------
                        MARGEN_BUSQUEDA = 0.01          # colchón para que la búsqueda no quede al filo
                        PASOS_MITIGACION = [0.05, 0.10, 0.15]  # cuánto por encima del umbral real se relaja

                        def generar_y_validar(umbral_calibracion, umbral_validacion, total_cfs=12):
                            """Genera CFs con DiCE calibrado a 'umbral_calibracion' y valida
                            cada candidato redondeado contra el pipeline REAL exigiendo
                            riesgo_real < umbral_validacion. Devuelve un DataFrame ya
                            redondeado, deduplicado y validado, o None si no hay nada viable."""
                            modelo_sinc = ModeloSincronizado(pipeline, columnas_modelo, umbral_calibracion)
                            m_local = dice_ml.Model(model=modelo_sinc, backend="sklearn")
                            exp_local = dice_ml.Dice(d, m_local, method="random")

                            try:
                                dice_exp_local = exp_local.generate_counterfactuals(
                                    df_paciente, total_CFs=total_cfs, desired_class="opposite",
                                    features_to_vary=vars_a_variar, permitted_range=rangos_permitidos,
                                    random_seed=42
                                )
                                cf_local = dice_exp_local.cf_examples_list[0].final_cfs_df
                            except Exception:
                                cf_local = None

                            if cf_local is None or cf_local.empty:
                                return None

                            cf_local = cf_local.copy()

                            # Redondeo a unidades clínicas (ya debería venir casi exacto
                            # gracias a continuous_features_precision, esto es solo blindaje)
                            for col in vars_a_variar:
                                if col not in ['EVO_dolor_eva', 'EVO_gravedad_percibida', 'dias_internados']:
                                    cf_local[col] = (cf_local[col] >= 0.5).astype(int)
                                else:
                                    cf_local[col] = cf_local[col].round()

                            cf_local = cf_local.drop_duplicates(subset=vars_a_variar).reset_index(drop=True)

                            # --- VALIDACIÓN FINAL: recalcular con el pipeline REAL ---
                            riesgos_reales = []
                            es_valida = []
                            for idx_row in range(len(cf_local)):
                                overrides = {col: cf_local.iloc[idx_row][col] for col in vars_a_variar}
                                fila_sim = construir_fila_simulada(df_paciente, overrides)
                                riesgo_real_ruta = pipeline.predict_proba(fila_sim)[0][1]
                                riesgos_reales.append(riesgo_real_ruta)
                                es_valida.append(riesgo_real_ruta < umbral_validacion)

                            cf_local['riesgo_real_post_redondeo'] = riesgos_reales
                            cf_local = cf_local[es_valida].reset_index(drop=True)

                            if cf_local.empty:
                                return None

                            # Deduplicar por firma cualitativa, priorizando menor esfuerzo clínico
                            firmas_vistas = {}
                            rutas_unicas = []
                            for idx_row, row in cf_local.iterrows():
                                firma_cualitativa = []
                                esfuerzo_clinico = 0
                                for col in vars_a_variar:
                                    val_orig = df_paciente.iloc[0][col]
                                    val_cf = row[col]
                                    if val_orig != val_cf:
                                        firma_cualitativa.append(col)
                                        if 'dolor' in col or 'gravedad' in col:
                                            esfuerzo_clinico += abs(val_orig - val_cf)

                                firma_str = str(sorted(firma_cualitativa))
                                if firma_str not in firmas_vistas:
                                    firmas_vistas[firma_str] = (idx_row, esfuerzo_clinico)
                                    rutas_unicas.append(idx_row)
                                else:
                                    idx_previo, esfuerzo_previo = firmas_vistas[firma_str]
                                    if esfuerzo_clinico < esfuerzo_previo:
                                        firmas_vistas[firma_str] = (idx_row, esfuerzo_clinico)
                                        rutas_unicas.remove(idx_previo)
                                        rutas_unicas.append(idx_row)

                            cf_local = cf_local.loc[rutas_unicas].reset_index(drop=True)

                            cf_local['DELTA_dolor_eva'] = cf_local['EVO_dolor_eva'] - df_paciente.iloc[0]['ING_dolor_eva']
                            cf_local['DELTA_gravedad_percibida'] = cf_local['EVO_gravedad_percibida'] - df_paciente.iloc[0]['ING_gravedad_percibida']
                            cf_local['DELTA_alteracion_mental'] = cf_local['EVO_alteracion_mental'] - df_paciente.iloc[0]['ING_alteracion_mental']
                            cf_local['DELTA_dependencia_funcional'] = cf_local['EVO_dependencia_funcional'] - df_paciente.iloc[0]['ING_dependencia_funcional']
                            cf_local['DELTA_portador_dispositivos'] = cf_local['EVO_portador_dispositivos'] - df_paciente.iloc[0]['ING_portador_dispositivos']

                            return cf_local

                        # --- Construir la escalera de etapas: alta segura -> mitigación graduada ---
                        etapas = [(umbral - MARGEN_BUSQUEDA, umbral, True)]
                        for paso in PASOS_MITIGACION:
                            objetivo = umbral + paso
                            if objetivo < riesgo - 0.01:  # solo tiene sentido si mejora el riesgo actual
                                etapas.append((objetivo - MARGEN_BUSQUEDA, objetivo, False))

                        cf_df = None
                        modo_mitigacion = False
                        umbral_logrado = None

                        for umbral_calib, umbral_valid, es_alta_segura in etapas:
                            resultado = generar_y_validar(umbral_calib, umbral_valid, total_cfs=12)
                            if resultado is not None and not resultado.empty:
                                cf_df = resultado
                                modo_mitigacion = not es_alta_segura
                                umbral_logrado = umbral_valid
                                break

                        if cf_df is None or cf_df.empty:
                            st.error("No mathematically viable target routes were found for discharge or mitigation, even after relaxing the clinical target.")
                        else:
                            if modo_mitigacion:
                                st.warning(
                                    f"⚠️ **{len(cf_df)} HARM REDUCTION ROUTES FOUND:**\n"
                                    f"Safe discharge threshold (< {umbral*100:.1f}%) is not mathematically viable in one step. "
                                    f"Showing intermediate targets that bring real risk below {umbral_logrado*100:.1f}% "
                                    f"(still above the safe-discharge threshold)."
                                )
                            else:
                                st.success(f"✅ **{len(cf_df)} UNIQUE CLINICAL STABILIZATION TARGETS FOUND (verified < {umbral*100:.1f}%):**")

                            evo_output_dict = {
                                'EVO_dolor_eva': 'Current Pain', 'EVO_gravedad_percibida': 'Current Severity',
                                'EVO_aislamiento_infeccioso': 'Infectious Isolation', 'EVO_alteracion_mental': 'Mental Alteration',
                                'EVO_complicacion_internacion': 'Hospitalization Complication',
                                'EVO_dependencia_funcional': 'Functional Dependency',
                                'EVO_portador_dispositivos': 'Device Bearer',
                                'EVO_cambio_terapeutico_mayor': 'Major Therapeutic Change',
                                'EVO_intervencion_quirurgica': 'Surgical Intervention',
                                'EVO_soporte_transfusional': 'Transfusion Support',
                                'dias_internados': 'Additional Hospitalization Days',
                                'EVO_terapia_endovenosa_prolongada': 'Prolonged IV Therapy', 
                                'EVO_inestabilidad_residual': 'Residual Instability'
                            }

                            with st.container(height=500):
                                for r_idx in range(len(cf_df)):
                                    with st.expander(f"➔ 🛤️ Alternative Target Route {r_idx + 1}", expanded=(r_idx == 0)):
                                        cambios_detectados = 0
                                        st.markdown("##### 🎯 Stabilization Actions:")

                                        for col in vars_a_variar:
                                            val_orig = df_paciente.iloc[0][col]
                                            val_cf = cf_df.iloc[r_idx][col]

                                            if val_orig != val_cf:
                                                cambios_detectados += 1
                                                col_en = evo_output_dict.get(col, col)

                                                if 'dolor' in col or 'gravedad' in col:
                                                    st.write(f"- 💊 **{col_en}**: Target reduction ➔ **[{val_cf:.0f}]** (Currently: {val_orig:.0f})")
                                                elif col == 'dias_internados':
                                                    dias_extra = val_cf - val_orig
                                                    st.write(f"- ⏳ **{col_en}**: Extend stay by ➔ **[+{dias_extra:.0f} days]** (Total target: {val_cf:.0f})")
                                                else:
                                                    status_en = "Resolved/Absent" if val_cf == 0 else "Present"
                                                    st.write(f"- 🛡️ **{col_en}**: Target status ➔ **[{status_en}]**")

                                        riesgo_ruta = cf_df.iloc[r_idx]['riesgo_real_post_redondeo']
                                        st.markdown(
                                            f"📉 **Verified real risk after this route (recomputed on the actual pipeline): "
                                            f"{riesgo_ruta*100:.1f}%** &nbsp; (safe-discharge threshold: {umbral*100:.1f}%)"
                                        )

                                        if cambios_detectados == 0:
                                            st.write("This alternative suggests maintaining current parameters based on marginal risk stability.")
                                        else:
                                            radar_map = {
                                                'Δ Pain': ('EVO_dolor_eva', 'ING_dolor_eva'),
                                                'Δ Severity': ('EVO_gravedad_percibida', 'ING_gravedad_percibida'),
                                                'Δ Mental Alt.': ('EVO_alteracion_mental', 'ING_alteracion_mental'),
                                                'Δ Func. Dep.': ('EVO_dependencia_funcional', 'ING_dependencia_funcional'),
                                                'Δ Devices': ('EVO_portador_dispositivos', 'ING_portador_dispositivos')
                                            }

                                            categorias_radar = list(radar_map.keys())
                                            valores_actuales_radar = []
                                            valores_meta_radar = []

                                            for cat, (col_evo, col_ing) in radar_map.items():
                                                v_ing = df_paciente.iloc[0].get(col_ing, 0)
                                                v_evo_act = df_paciente.iloc[0].get(col_evo, 0)
                                                valores_actuales_radar.append(v_evo_act - v_ing)

                                                v_evo_meta = cf_df.iloc[r_idx].get(col_evo, v_evo_act)
                                                valores_meta_radar.append(v_evo_meta - v_ing)

                                            cat_cerradas = categorias_radar + [categorias_radar[0]]
                                            val_act_cerrados = valores_actuales_radar + [valores_actuales_radar[0]]
                                            val_meta_cerrados = valores_meta_radar + [valores_meta_radar[0]]

                                            fig_radar = go.Figure()

                                            fig_radar.add_trace(go.Scatterpolar(
                                                r=val_act_cerrados, theta=cat_cerradas,
                                                fill='toself', fillcolor='rgba(214, 39, 40, 0.25)',
                                                line=dict(color='#D62728', width=2.5), name='Current State'
                                            ))

                                            color_target = '#FF8C00' if modo_mitigacion else '#2CA02C'
                                            fill_target = 'rgba(255, 140, 0, 0.25)' if modo_mitigacion else 'rgba(44, 160, 44, 0.25)'

                                            fig_radar.add_trace(go.Scatterpolar(
                                                r=val_meta_cerrados, theta=cat_cerradas,
                                                fill='toself', fillcolor=fill_target,
                                                line=dict(color=color_target, width=2.5), name='DiCE Target'
                                            ))

                                            fig_radar.update_layout(
                                                polar=dict(
                                                    radialaxis=dict(visible=True, range=[-2, 8]),
                                                    bgcolor='rgba(0,0,0,0)'
                                                ),
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                margin=dict(l=40, r=40, t=40, b=40),
                                                height=450,
                                                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
                                            )

                                            st.plotly_chart(fig_radar, use_container_width=True, key=f"dice_radar_ruta_cf_{r_idx}")

                except Exception as e:
                    st.error("Counterfactual engine is currently unavailable.")
                    st.warning(f"Technical Context: {str(e)}")

    # ==========================================
    # --- MOTOR DE IMPACTO CLÍNICO ACCIONABLE (SHAP ENGINE) ---
    # ==========================================
    st.markdown("---")
    try:
        with st.spinner("Analyzing precise clinical impacts of simulated trajectory..."):
            overrides_sandbox = {
                'dias_internados': float(dias_sim),
                'EVO_dolor_eva': float(dolor_sim),
                'EVO_gravedad_percibida': float(severidad_sim),
            }
            for col, val in status_evo_sim.items():
                overrides_sandbox[col] = 1.0 if val else 0.0

            # Misma función usada para validar las rutas de DiCE -> consistencia garantizada
            df_sim = construir_fila_simulada(df_paciente, overrides_sandbox)

            # Cálculo de riesgos marginales
            riesgo_base = pipeline.predict_proba(df_paciente)[0][1]
            riesgo_simulado = pipeline.predict_proba(df_sim)[0][1]
            variacion_riesgo = (riesgo_simulado - riesgo_base) * 100

            # Extracción del pipeline
            prep = pipeline.named_steps['preprocesador']
            clf = pipeline.named_steps['clasificador']

            X_sim_proc = prep.transform(df_sim)
            X_sim_dense = X_sim_proc.toarray() if hasattr(X_sim_proc, 'toarray') else np.array(X_sim_proc)

            columnas_modelo_final = list(prep.get_feature_names_out())

            # --- DICCIONARIO ACTUALIZADO ---
            ui_dict = {
                'dias_internados': 'Hospitalization Length of Stay',
                'DELTA_dolor_eva': 'Δ Pain Progression (Discharge - Admission)',
                'DELTA_gravedad_percibida': 'Δ Severity Progression (Discharge - Admission)',
                'DELTA_alteracion_mental': 'Δ Mental State Progression',
                'DELTA_dependencia_funcional': 'Δ Functional Dependency Progression',
                'DELTA_portador_dispositivos': 'Δ Invasive Devices Progression',
                'EVO_dolor_eva': 'Current Discharge Pain (VAS)',
                'EVO_gravedad_percibida': 'Current Discharge Severity Score',
                'EVO_aislamiento_infeccioso': 'Infectious Isolation Status',
                'EVO_complicacion_internacion': 'Acquired Hospital Complication',
                'EVO_alteracion_mental': 'Active Delirium / Mental Alteration',
                'EVO_dependencia_funcional': 'Severe Functional Dependency',
                'EVO_portador_dispositivos': 'Active Medical Device Bearer',
                'EVO_cambio_terapeutico_mayor': 'Major Therapeutic Change',
                'EVO_intervencion_quirurgica': 'Surgical Intervention Performed',
                'EVO_soporte_transfusional': 'Transfusion Support Required',
                'EVO_terapia_endovenosa_prolongada': 'Prolonged IV Therapy', 
                'EVO_inestabilidad_residual': 'Residual Instability'
            }

            nombre_modelo = type(clf).__name__
            if 'XGB' in nombre_modelo:
                explainer = shap.TreeExplainer(clf.get_booster())
                shap_sim = explainer.shap_values(X_sim_dense, check_additivity=False)
            elif 'Forest' in nombre_modelo or 'Boost' in nombre_modelo:
                explainer = shap.TreeExplainer(clf)
                shap_sim = explainer.shap_values(X_sim_dense, check_additivity=False)
            else:
                explainer = shap.LinearExplainer(clf, np.zeros((1, X_sim_dense.shape[1])))
                shap_sim = explainer.shap_values(X_sim_dense)

            if isinstance(shap_sim, list):
                shap_sim = shap_sim[1][0] if len(shap_sim) > 1 else shap_sim[0][0]
            elif len(shap_sim.shape) == 3:
                shap_sim = shap_sim[0, :, 1]
            elif len(shap_sim.shape) == 2:
                shap_sim = shap_sim[0]

            cambios_impacto = []
            for i, feat_name in enumerate(columnas_modelo_final):
                n_clean = str(feat_name).replace('num__', '').replace('cat__', '').split('_1')[0].split('_TRUE')[0].split('_1.0')[0]

                if n_clean.startswith(('EVO_', 'DELTA_')) or n_clean == 'dias_internados':
                    peso_pct = shap_sim[i] * 100

                    # --- CORRECCIÓN DE VISIBILIDAD: OCULTAR LO INACTIVO ---
                    es_valido = True
                    
                    val_orig = float(df_paciente.get(n_clean, pd.Series([0.0])).iloc[0])
                    val_sim = float(df_sim.get(n_clean, pd.Series([0.0])).iloc[0])
                    
                    # Si el paciente nunca tuvo el síntoma (ni al ingreso ni en la simulación), se oculta
                    if val_orig == 0.0 and val_sim == 0.0:
                        es_valido = False

                    if es_valido and abs(peso_pct) > 0.01:
                        label_display = ui_dict.get(n_clean, n_clean.replace('_', ' ').title())
                        cambios_impacto.append((label_display, peso_pct))

            if not cambios_impacto:
                st.info("The current configuration of evolutionary variables has a neutral impact on risk.")
            else:
                cambios_impacto = sorted(cambios_impacto, key=lambda x: x[1])

                fig_delta = go.Figure(go.Bar(
                    x=[x[1] for x in cambios_impacto],
                    y=[x[0] for x in cambios_impacto],
                    orientation='h',
                    marker_color=['#FF4444' if x[1] > 0 else '#00C851' for x in cambios_impacto],
                    text=[f"+{x[1]:.1f}%" if x[1] > 0 else f"{x[1]:.1f}%" for x in cambios_impacto],
                    textposition='outside',
                    textfont=dict(size=12)
                ))

                fig_delta.update_layout(
                    title="Actionable Phenotype Risk Contributions",
                    xaxis_title="Impact on Readmission Probability (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=50, t=40, b=10),
                    height=max(320, len(cambios_impacto) * 42),
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True, zerolinecolor='rgba(128,128,128,0.6)', zerolinewidth=1.5
                    )
                )

                col_l1, col_l2 = st.columns([1, 3])

                with col_l2:
                    st.plotly_chart(fig_delta, use_container_width=True)

                with col_l1:
                    st.markdown("### 📊 Target Risk")
                    st.metric(
                        label="Hypothetical 15-Day Prob.",
                        value=f"{riesgo_simulado*100:.1f}%",
                        delta=f"{variacion_riesgo:+.1f}% vs Admission",
                        delta_color="inverse"
                    )
                    st.markdown("---")

    except Exception as e:
        st.error("Simulation engine encountered an error.")
        st.warning(f"Error detail: {str(e)}")


# ==========================================
# --- 8 TAB EVIDENCIA ---
# ==========================================

with tab_evidencia:
    st.markdown("#### Clinical Similarity Network & Cohort Audit")
    
    def format_clinical_value(key_es, value):
        val_str = str(value).strip().upper()
        if key_es == 'rango_edad':
            traducciones_edad = {
                'ADULTO DE MEDIANA EDAD': 'Middle-Aged Adult', 
                'ADULTO MAYOR': 'Older Adult', 
                'ADULTO JOVEN': 'Young Adult'
            }
            return traducciones_edad.get(val_str, value)
        if key_es == 'sexo':
            return {'MASCULINO': 'Male', 'FEMENINO': 'Female'}.get(val_str, value)
        if key_es == 'Area':
            traduccion_area = {
                'CLINICA_MEDICA': 'Internal Medicine', 'CLINICA MEDICA': 'Internal Medicine', 
                'EMERG_GUARDIAS': 'ER', 'GUARDIA': 'ER'
            }
            return traduccion_area.get(val_str, value)
        if key_es == 'perfil_clinico_ingreso':
            traducciones_perfil = {
                'INTERNACION INICIAL': 'Initial Admission',
                'COMPLICACION ASOCIADA A LA INTERNECION': 'Admission-Associated Complication',
                'DESCOMPENSACION DE PLURIPATOLOGIA': 'Multimorbidity Decompensation',
                'MISMA CAUSA': 'Same Cause',
                'REINTERNACION NO ASOCIADA': 'Unrelated Readmission'
            }
            return traducciones_perfil.get(val_str, value)
        if key_es == 'IN_COMPLEJIDAD':
            traduccion_comp = {'ALTA': 'High', 'MEDIA': 'Medium', 'BAJA': 'Low'}
            return traduccion_comp.get(val_str, value) if not val_str.isdigit() else str(int(float(value)))
        if key_es == 'CIE10_MACRO':
            cie10_ui_dict = {
                "Tuberculosis": "Tuberculosis", "Lepra": "Leprosy", "Sífilis": "Syphilis", 
                "Otras infecciosas (A)": "Other infectious (A)", "Hepatitis viral": "Viral hepatitis", 
                "Enfermedad por VIH": "HIV disease", "Enfermedad de Chagas": "Chagas disease", 
                "Toxoplasmosis": "Toxoplasmosis", "Equinococosis / Hidatidosis": "Echinococcosis / Hydatidosis", 
                "Secuelas de enfermedades infecciosas": "Sequelae of infectious diseases", "Otras infecciosas (B)": "Other infectious (B)",
                "Cáncer de labio / boca / faringe": "Lip / mouth / pharynx cancer", "Cáncer digestivo": "Digestive cancer", 
                "Cáncer respiratorio / intratorácico": "Respiratory / intrathoracic cancer", "Cáncer de hueso / cartílago": "Bone / cartilage cancer", 
                "Melanoma / Cáncer de piel": "Melanoma / Skin cancer", "Cáncer de mama": "Breast cancer", 
                "Cáncer genital femenino": "Female genital cancer", "Cáncer genital masculino": "Male genital cancer", 
                "Cáncer de vías urinarias": "Urinary tract cancer", "Cáncer de sistema nervioso central": "Central nervous system cancer", 
                "Cáncer linfoide / hematopoyético": "Lymphoid / hematopoietic cancer", "Otros tumores malignos": "Other malignant tumors", 
                "Tumores in situ o benignos": "In situ or benign tumors", "Anemias nutricionales": "Nutritional anemias", 
                "Anemias hemolíticas": "Hemolytic anemias", "Aplasias y otras anemias": "Aplasias and other anemias", 
                "Defectos de coagulación / púrpura": "Coagulation defects / purpura", "Trastornos de inmunodeficiencia": "Immunodeficiency disorders", 
                "Otros trastornos de la sangre": "Other blood disorders",
                "Tiroides": "Thyroid", "Diabetes": "Diabetes", "Glucosa / hipoglucemia": "Glucose / hypoglycemia", 
                "Otros endocrinos y metabólicos": "Other endocrine and metabolic", "Obesidad y trastornos de hiperalimentación": "Obesity and hyperalimentation disorders", 
                "Dislipidemia": "Dyslipidemia", "Fibrosis quística": "Cystic fibrosis", "Trastornos metabólicos": "Metabolic disorders", 
                "Otros metabólicos / nutricionales": "Other metabolic / nutritional",
                "Trastornos mentales orgánicos (Demencias)": "Organic mental disorders (Dementias)", "Trastornos por uso de sustancias": "Substance use disorders", 
                "Esquizofrenia y trastornos psicóticos": "Schizophrenia and psychotic disorders", "Trastornos del humor (Afectivos)": "Mood (Affective) disorders", 
                "Trastornos neuróticos y de ansiedad": "Neurotic and anxiety disorders", "Trastornos de la conducta alimentaria / sueño": "Eating / sleep disorders", 
                "Trastornos de la personalidad": "Personality disorders", "Discapacidad intelectual": "Intellectual disability", 
                "Trastornos del desarrollo psicobiológico (Autismo)": "Psychobiological development disorders (Autism)", "Otros trastornos mentales": "Other mental disorders",
                "Atrofias sistémicas del SNC": "Systemic atrophies of CNS", "Trastornos extrapiramidales y del movimiento (Parkinson)": "Extrapyramidal and movement disorders (Parkinson's)", 
                "Enfermedades degenerativas (Alzheimer)": "Degenerative diseases (Alzheimer's)", "Enfermedades desmielinizantes (Esclerosis Múltiple)": "Demyelinating diseases (Multiple Sclerosis)", 
                "Trastornos episódicos y paroxísticos (Epilepsia, Migraña)": "Episodic and paroxysmal disorders (Epilepsy, Migraine)", "Trastornos de nervios y plexos": "Nerve and plexus disorders", 
                "Polineuropatías": "Polyneuropathies", "Enfermedades de la unión neuromuscular (Miastenia)": "Diseases of the neuromuscular junction (Myasthenia)", 
                "Parálisis cerebral y síndromes paralíticos": "Cerebral palsy and paralytic syndromes", "Otros trastornos neurológicos": "Other neurological disorders",
                "Ojo": "Eye", "Oído": "Ear", "Otros órganos de los sentidos": "Other sense organs",
                "Hipertensión": "Hypertension", "Cardiopatía isquémica": "Ischemic heart disease", "Enfermedad cardiopulmonar": "Cardiopulmonary disease", 
                "Otras enfermedades del corazón (Insuficiencia Cardíaca)": "Other heart diseases (Heart Failure)", "Cerebrovascular": "Cerebrovascular", 
                "Enfermedades de arterias y capilares": "Diseases of arteries and capillaries", "Enfermedades de venas y vasos linfáticos": "Diseases of veins and lymphatic vessels", 
                "Otros circulatorios": "Other circulatory",
                "Vías respiratorias altas": "Upper respiratory tract", "Infecciones agudas / neumonía / influenza": "Acute infections / pneumonia / influenza", 
                "Infecciones respiratorias bajas": "Lower respiratory infections", "Enfermedades de vías respiratorias superiores": "Diseases of upper respiratory tract", 
                "Asma / EPOC / bronquitis": "Asthma / COPD / bronchitis", "Enfermedades del pulmón por agentes externos (Neumoconiosis)": "Lung diseases due to external agents (Pneumoconiosis)", 
                "Enfermedades pulmonares intersticiales": "Interstitial lung diseases", "Otros respiratorios": "Other respiratory",
                "Boca / dientes / faringe": "Mouth / teeth / pharynx", "Esófago / estómago / duodeno": "Esophagus / stomach / duodenum", 
                "Apendicitis": "Appendicitis", "Hernias": "Hernias", "Enfermedad de Crohn y colitis": "Crohn's disease and colitis", 
                "Otras enfermedades de los intestinos": "Other diseases of the intestines", "Hígado": "Liver", 
                "Vesícula / vías biliares / páncreas": "Gallbladder / biliary tract / pancreas", "Otros digestivos": "Other digestive",
                "Dermatitis y eczema": "Dermatitis and eczema", "Trastornos papuloescamosos (Psoriasis)": "Papulosquamous disorders (Psoriasis)", 
                "Urticaria y eritema": "Urticaria and erythema", "Trastornos de las faneras / Otros trastornos de piel": "Disorders of skin appendages / Other skin disorders", 
                "Otras enfermedades de la piel": "Other skin diseases",
                "Artropatías": "Arthropathies", "Tejido conectivo (Lupus, etc.)": "Connective tissue (Lupus, etc.)", "Dorsopatías": "Dorsopathies", 
                "Tejidos blandos": "Soft tissues", "Osteopatías y condropatías (Osteoporosis)": "Osteopathies and chondropathies (Osteoporosis)", 
                "Otros osteomusculares": "Other musculoskeletal",
                "Riñón (Insuficiencia Renal Crónica)": "Kidney (Chronic Renal Failure)", "Vías urinarias bajas": "Lower urinary tract", 
                "Genital masculino (Hiperplasia Prostática)": "Male genital (Prostatic Hyperplasia)", "Mama": "Breast", 
                "Genital femenino (Endometriosis, etc.)": "Female genital (Endometriosis, etc.)", "Otros genitourinarios": "Other genitourinary",
                "Malformaciones del sistema nervioso (Espina bífida)": "Malformations of the nervous system (Spina bifida)", "Malformaciones cardíacas congénitas": "Congenital heart malformations", 
                "Anomalías cromosómicas (Síndrome de Down)": "Chromosomal abnormalities (Down Syndrome)", "Otras malformaciones congénitas": "Other congenital malformations",
                "Enfermedad respiratoria crónica perinatal": "Chronic perinatal respiratory disease", 
                "Secuelas crónicas de traumatismos": "Chronic sequelae of injuries",
                "Síndrome Post-COVID (Long COVID)": "Post-COVID Syndrome (Long COVID)", "Otras condiciones especiales (U)": "Other special conditions (U)",
                "Historia personal de tumores / enfermedades": "Personal history of tumors / diseases", "Ausencia adquirida de miembros / órganos": "Acquired absence of limbs / organs", 
                "Aberturas artificiales (Ostomías)": "Artificial openings (Ostomies)", "Estado de órgano trasplantado": "Transplanted organ status", 
                "Presencia de implantes cardíacos / vasculares": "Presence of cardiac / vascular implants", "Dependencia de máquinas (diálisis, oxígeno)": "Machine dependence (dialysis, oxygen)", 
                "Otros factores de salud": "Other health factors",
                "DESCONOCIDO": "UNKNOWN"
            }
            traducciones_cie = {k.upper(): v for k, v in cie10_ui_dict.items()}
            return traducciones_cie.get(val_str, value)
        
        bool_suffixes = ('_mental', '_funcional', '_dispositivos', '_reiteradas', '_hemorragico', '_internacion', '_infeccioso', '_activa', '_mayor', '_quirurgica', '_transfusional', '_prolongada', '_residual', '_severa')
        if key_es.startswith('LLM_') or key_es.startswith('EST_') or key_es.startswith('Riesgo_') or key_es in ('pluripatologico') or (key_es.endswith(bool_suffixes) and not key_es.startswith('DELTA_')):
            try:
                return "Yes" if float(value) == 1.0 else "No"
            except ValueError:
                if val_str in ['TRUE', 'YES', '1']: return "Yes"
                if val_str in ['FALSE', 'NO', '0']: return "No"
                pass
                
        try:
            f_val = float(value)
            if f_val.is_integer(): return str(int(f_val))
        except ValueError:
            pass
        return value

    def safe_int(value, default="N/A"):
        try:
            if pd.isna(value) or value == "": return default
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def traducir_ninguno(texto):
        texto_str = str(texto).strip()
        if texto_str.upper() == 'NINGUNO': return 'None'
        elif texto_str == '': return 'Unknown'
        return texto_str

    def renderizar_notas_gemelo(texto_evolucion, citas_llm, lista_enfermedades):
        if not isinstance(texto_evolucion, str) or not texto_evolucion:
            return "No narrative context available."
            
        texto_resaltado = texto_evolucion
        if isinstance(citas_llm, dict):
            for cita, variable in citas_llm.items():
                if isinstance(cita, str) and cita.strip():
                    cita_escapada = re.escape(cita)
                    marcador = f"<mark style='background-color: #FFF2CC; color: #000000; border-radius: 3px; padding: 2px 4px;'><b>{cita}</b> <span style='font-size: 0.7em; background-color: #FFD966; padding: 2px 5px; border-radius: 8px; color: #594000; margin-left: 4px; display: inline-block; vertical-align: middle; line-height: 1;'>{variable}</span></mark>"
                    texto_resaltado = re.sub(cita_escapada, marcador, texto_resaltado, flags=re.IGNORECASE)
                    
        if isinstance(lista_enfermedades, list):
            for enfermedad in lista_enfermedades:
                if isinstance(enfermedad, str) and enfermedad.strip():
                    patron = rf"\b({re.escape(enfermedad)})\b"
                    marcador = r"<mark style='background-color: #FFCCCC; color: #000000; border-radius: 3px; padding: 0px 2px;'>\1</mark>"
                    texto_resaltado = re.sub(patron, marcador, texto_resaltado, flags=re.IGNORECASE)
                    
        return f"""
        <div style='line-height: 1.8; font-size: 14px; padding: 15px; background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1)); color: var(--text-color, inherit); border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.2);'>
            {texto_resaltado}
        </div>
        """

    TRANSLATION_DICT = {
        'dias_internados': 'Length of Stay (Days)', 'rango_edad': 'Age Range', 'pluripatologico': 'Multimorbidity',
        'CIE10_MACRO': 'Primary Diagnosis (ICD-10)', 'LLM_tabaquismo_activo': 'Active Smoking', 
        'LLM_polifarmacia': 'Polypharmacy',
        'LLM_historial_caidas': 'History of Falls', 'LLM_abandono_medicacion': 'Medication Abandonment',
        'ING_dolor_eva': 'Admission: Pain (VAS)', 'ING_gravedad_percibida': 'Admission: Perceived Severity',
        'ING_alteracion_mental': 'Admission: Mental Alteration', 'ING_dependencia_funcional': 'Admission: Functional Dependency',
        'ING_portador_dispositivos': 'Admission: Device Bearer', 'ING_consultas_reiteradas': 'Admission: Repeated Consultations',
        'ING_riesgo_hemorragico': 'Admission: Hemorrhagic Risk', 'ING_infeccion_activa': 'Admission: Active Infection',
        'EVO_dolor_eva': 'Evolution: Pain (VAS)', 'EVO_gravedad_percibida': 'Evolution: Perceived Severity', 
        'EVO_alteracion_mental': 'Evolution: Mental Alteration', 'EVO_dependencia_funcional': 'Evolution: Functional Dependency', 
        'EVO_portador_dispositivos': 'Evolution: Device Bearer', 'EVO_complicacion_internacion': 'Evolution: Hospital Complication', 
        'EVO_aislamiento_infeccioso': 'Evolution: Infectious Isolation', 
        'EVO_cambio_terapeutico_mayor': 'Evolution: Major Therapeutic Change', 'EVO_intervencion_quirurgica': 'Evolution: Surgical Intervention', 
        'EVO_soporte_transfusional': 'Evolution: Transfusion Support', 'DELTA_dolor_eva': 'Δ Pain (VAS)',
        'DELTA_gravedad_percibida': 'Δ Perceived Severity', 'DELTA_alteracion_mental': 'Δ Mental Alteration',
        'DELTA_dependencia_funcional': 'Δ Functional Dependency', 'DELTA_portador_dispositivos': 'Δ Device Bearer',
        'sexo': 'Sex', 'Area': 'Admission Area', 'IN_COMPLEJIDAD': 'Complexity Level', 'TR_Prioridad': 'Triage Priority',
        'cantidad_interconsultas': 'Interconsultations', 'visitas_guardia_6meses_previos': 'ER Visits (6m)',
        'EST_ingreso_ambulancia': 'Ambulance Arrival', 
        'HIST_condicion_ultimo_egreso': 'Previous Discharge Condition',
        'perfil_clinico_ingreso': 'Admission Profile',
        'EST_paso_por_uti': 'ICU Stay (UTI)',
        'Riesgo_Cardiovasculares_Inotropicos': 'High-Risk Meds: Cardiovascular/Inotropes',
        'Riesgo_Psicofarmacos_Neurologicos': 'High-Risk Meds: Psychotropics/Neurological'
    }

    @st.cache_resource
    def load_similarity_assets():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ruta_x = os.path.join(BASE_DIR, 'X_train_proc_llm.npy')
        ruta_ext = os.path.join(BASE_DIR, 'matriz_extended_display_llm.npy')
        ruta_cols = os.path.join(BASE_DIR, 'columnas_display_llm.npy')
        
        if not all(os.path.exists(p) for p in [ruta_x, ruta_ext, ruta_cols]):
            raise FileNotFoundError("Similarity assets missing. Ensure matrices are in the server volume.")
            
        X_train_proc = np.load(ruta_x)
        matriz_ext = np.load(ruta_ext, allow_pickle=True)
        nombres_columnas = np.load(ruta_cols, allow_pickle=True)
        
        # --- APLICACIÓN DEL FILTRO FORENSE (RUIDO TOPOLÓGICO) ---
        VARIABLES_RUIDO = [
            'cat__CIE10_MACRO_DERMATITIS Y ECZEMA', 'cat__CIE10_MACRO_OTROS TRASTORNOS NEUROLÓGICOS', 'cat__CIE10_MACRO_OTRAS INFECCIOSAS (B)', 'cat__CIE10_MACRO_TRASTORNOS DE LAS FANERAS / OTROS TRASTORNOS DE PIEL', 'cat__CIE10_MACRO_TEJIDO CONECTIVO (LUPUS, ETC.)', 'cat__CIE10_MACRO_OTROS TUMORES MALIGNOS', 'cat__CIE10_MACRO_OTRAS ENFERMEDADES DE LOS INTESTINOS', 'cat__CIE10_MACRO_HERNIAS', 'cat__CIE10_MACRO_ENFERMEDADES DE LA UNIÓN NEUROMUSCULAR (MIASTENIA)', 'cat__CIE10_MACRO_OTROS FACTORES DE SALUD', 'cat__CIE10_MACRO_ANEMIAS HEMOLÍTICAS', 'cat__CIE10_MACRO_OTROS OSTEOMUSCULARES', 'cat__CIE10_MACRO_VESÍCULA / VÍAS BILIARES / PÁNCREAS', 'cat__CIE10_MACRO_POLINEUROPATÍAS', 'cat__CIE10_MACRO_GENITAL MASCULINO (HIPERPLASIA PROSTÁTICA)', 'cat__CIE10_MACRO_ENFERMEDADES DESMIELINIZANTES (ESCLEROSIS MÚLTIPLE)', 'cat__CIE10_MACRO_TRASTORNOS EXTRAPIRAMIDALES Y DEL MOVIMIENTO (PARKINSON)', 'cat__CIE10_MACRO_CÁNCER DE SISTEMA NERVIOSO CENTRAL', 'cat__CIE10_MACRO_GLUCOSA / HIPOGLUCEMIA', 'cat__CIE10_MACRO_ARTROPATÍAS', 'cat__CIE10_MACRO_OTRAS MALFORMACIONES CONGÉNITAS', 'cat__CIE10_MACRO_OTROS TRASTORNOS MENTALES', 'cat__CIE10_MACRO_TRASTORNOS PAPULOESCAMOSOS (PSORIASIS)', 'cat__CIE10_MACRO_ENFERMEDADES DEGENERATIVAS (ALZHEIMER)', 'cat__CIE10_MACRO_ESQUIZOFRENIA Y TRASTORNOS PSICÓTICOS', 'cat__CIE10_MACRO_TRASTORNOS DE NERVIOS Y PLEXOS', 'cat__CIE10_MACRO_TUMORES IN SITU O BENIGNOS', 'cat__CIE10_MACRO_TRASTORNOS NEURÓTICOS Y DE ANSIEDAD', 'cat__CIE10_MACRO_ANEMIAS NUTRICIONALES', 'cat__CIE10_MACRO_CÁNCER DE VÍAS URINARIAS', 'cat__CIE10_MACRO_OSTEOPATÍAS Y CONDROPATÍAS (OSTEOPOROSIS)', 'cat__CIE10_MACRO_OTROS TRASTORNOS DE LA SANGRE', 'cat__CIE10_MACRO_OTRAS ENFERMEDADES DE LA PIEL', 'cat__CIE10_MACRO_GENITAL FEMENINO (ENDOMETRIOSIS, ETC.)', 'cat__CIE10_MACRO_ENFERMEDADES PULMONARES INTERSTICIALES', 'cat__CIE10_MACRO_CÁNCER DE LABIO / BOCA / FARINGE', 'cat__CIE10_MACRO_ESÓFAGO / ESTÓMAGO / DUODENO', 'cat__CIE10_MACRO_ENFERMEDAD DE CROHN Y COLITIS', 'cat__CIE10_MACRO_TRASTORNOS DE LA PERSONALIDAD', 'cat__CIE10_MACRO_OTROS ENDOCRINOS Y METABÓLICOS', 'cat__CIE10_MACRO_OJO', 'cat__CIE10_MACRO_DEFECTOS DE COAGULACIÓN / PÚRPURA', 'cat__CIE10_MACRO_ENFERMEDAD POR VIH', 'cat__CIE10_MACRO_TRASTORNOS DE LA CONDUCTA ALIMENTARIA / SUEÑO', 'cat__CIE10_MACRO_CÁNCER RESPIRATORIO / INTRATORÁCICO', 'cat__CIE10_MACRO_CÁNCER DE MAMA', 'cat__CIE10_MACRO_DISLIPIDEMIA', 'cat__CIE10_MACRO_ATROFIAS SISTÉMICAS DEL SNC', 'cat__CIE10_MACRO_OÍDO'
        ]
        
        prep = pipeline.named_steps['preprocesador']
        nombres_expandidos = list(prep.get_feature_names_out())
        
        # Generar máscara para aislar variables predictivas reales
        mask_limpia = np.array([col not in VARIABLES_RUIDO for col in nombres_expandidos])
        
        # Extirpar las columnas sucias de la matriz de entrenamiento
        X_train_limpio = X_train_proc[:, mask_limpia]
        
        knn_engine = NearestNeighbors(n_neighbors=100, metric='cosine')
        knn_engine.fit(X_train_limpio)
        
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        umap_embeddings = umap_reducer.fit_transform(X_train_limpio)
        
        return knn_engine, matriz_ext, nombres_columnas, X_train_limpio, umap_embeddings, mask_limpia

    plt.close('all') 
    
    try:
        with st.spinner("Calculating topological metrics & dynamic geometry..."):
            knn_global, matriz_extended, nombres_columnas, X_train_limpio, umap_embeddings, mask_limpia = load_similarity_assets()
            
            prep = pipeline.named_steps['preprocesador']
            X_paciente_proc = prep.transform(df_paciente)
            X_paciente_dense = X_paciente_proc.toarray() if hasattr(X_paciente_proc, 'toarray') else np.array(X_paciente_proc)
            
            # --- CORTAR LAS VARIABLES DE RUIDO DEL PACIENTE ACTUAL ---
            X_paciente_limpio = X_paciente_dense[:, mask_limpia]
            
           

            # --- ESTRATEGIA 3: SELECCIÓN DINÁMICA DE VARIABLES (FILTRO ESTRICTO DE ACTIVACIÓN) ---
            nombres_expandidos = list(prep.get_feature_names_out())
            columnas_limpias = [col for col, is_clean in zip(nombres_expandidos, mask_limpia) if is_clean]
            
            def get_category_tag(base):
                if base in ['rango_edad', 'sexo']: return '👤 [Demographics]'
                if base.startswith('LLM_'): return '🧠 [NLP Phenotype]'
                if base.startswith('ING_') or base in ['TR_Prioridad', 'Area', 'EST_ingreso_ambulancia', 'perfil_clinico_ingreso', 'IN_COMPLEJIDAD']: return '🚨 [Admission & Triage]'
                if base.startswith('EVO_') or base in ['dias_internados', 'EST_paso_por_uti', 'cantidad_interconsultas']: return '🏥 [Evolution & Burden]'
                if base.startswith('DELTA_'): return '📉 [Clinical Deltas]'
                if base in ['CIE10_MACRO', 'pluripatologico', 'HIST_condicion_ultimo_egreso', 'visitas_guardia_6meses_previos']: return '🩺 [Clinical History]'
                if base.startswith('Riesgo_'): return '💊 [Medication Risk]'
                return '📊 [Other]'

            def humanize_col(col):
                clean_col = col.replace('num__', '').replace('cat__', '')
                matched_base = None
                
                # Sort descending length to catch full base variables before substrings
                for base in sorted(TRANSLATION_DICT.keys(), key=len, reverse=True):
                    if clean_col.startswith(base):
                        matched_base = base
                        break
                        
                if matched_base:
                    base_en = TRANSLATION_DICT[matched_base]
                    tag = get_category_tag(matched_base)
                    suffix = clean_col[len(matched_base):].strip('_')
                    
                    if suffix: # Manejo de One-Hot Encoded (ej. rango_edad_ADULTO MAYOR)
                        translated_val = format_clinical_value(matched_base, suffix)
                        # Si format_clinical_value no lo tradujo, lo formateamos limpio
                        if translated_val.upper() == suffix.upper():
                            translated_val = suffix.replace('_', ' ').title()
                        return f"{tag} {base_en}: {translated_val}"
                    else:
                        return f"{tag} {base_en}"
                else:
                    return f"📊 [Other] {clean_col.replace('_', ' ').title()}"

            # --- NUEVO: FILTRO ESTRICTO DE OPCIONES DISPONIBLES ---
            opciones_multiselect = {}
            default_cols = []
            claves_prioridad = ['dias_internados', 'TR_Prioridad', 'rango_edad', 'EST_paso_por_uti', 'pluripatologico', 'DELTA_gravedad_percibida', 'cantidad_interconsultas', 'CIE10_MACRO', 'perfil_clinico']
            
            for idx_col, col_name in enumerate(columnas_limpias):
                valor_paciente = X_paciente_limpio[0, idx_col]
                
                # Identificamos si es continua
                es_continua = any(p in col_name for p in ['dias_internados', 'cantidad_interconsultas', 'DELTA_', 'dolor_eva'])
                
                # REGLA ESTRICTA: Solo guardamos la opción si el paciente la tiene activa (o si es continua)
                if es_continua or valor_paciente > 0:
                    nombre_humano = humanize_col(col_name)
                    opciones_multiselect[nombre_humano] = col_name
                    
                    # Además, si es de prioridad, la guardamos para las sugerencias por defecto
                    if any(k in col_name for k in claves_prioridad):
                        if col_name not in default_cols:
                            default_cols.append(col_name)

            # Limitamos a las 6 más descriptivas para no saturar el selector inicial
            default_cols = default_cols[:6]
            
            default_selections = [humanize_col(c) for c in default_cols if humanize_col(c) in opciones_multiselect]
            if not default_selections:
                default_selections = list(opciones_multiselect.keys())[:5]

            st.info("🔬 **Active Similarity Variables:** The system pre-selected the active clinical features driving this patient's specific risk profile. Modify them below to adjust the neighborhood calculation.")
            selected_human_names = st.multiselect(
                "Select variables for KNN similarity (Dynamic Subspace):",
                options=list(opciones_multiselect.keys()),  # AHORA LA LISTA ESTÁ LIMPIA DE OPCIONES INÚTILES
                default=default_selections
            )


            
            if not selected_human_names:
                st.warning("Please select at least one variable to calculate similarity.")
                st.stop()
                
            selected_cols = [opciones_multiselect[name] for name in selected_human_names]
            selected_indices = [columnas_limpias.index(col) for col in selected_cols]
            
            # --- KNN LOCAL (Sub-espacio dimensional) ---
            X_train_dinamico = X_train_limpio[:, selected_indices]
            X_paciente_dinamico = X_paciente_limpio[:, selected_indices]
            
            knn_dinamico = NearestNeighbors(n_neighbors=100, metric='cosine')
            knn_dinamico.fit(X_train_dinamico)
            
            distancias, indices = knn_dinamico.kneighbors(X_paciente_dinamico)
            vecinos_idx_pool = indices[0]
            similitudes_brutas_pool = np.maximum(0, (1 - distancias[0])) * 100
            
            col_slider, col_empty = st.columns([2, 1])
            with col_slider:
                umbral_similitud = st.slider(
                    "Minimum Similarity Threshold (%)", 
                    min_value=50, max_value=99, value=70, step=1,
                    help="Higher values restrict the network to nearly identical admissions."
                )
            
            st.markdown("---")
            
            mask_umbral = similitudes_brutas_pool >= umbral_similitud
            vecinos_idx = vecinos_idx_pool[mask_umbral]
            similitudes_brutas = similitudes_brutas_pool[mask_umbral]
            
            if len(vecinos_idx) == 0:
                st.warning(f"No historical admissions found with a similarity match equal to or greater than {umbral_similitud}%. Please lower the threshold or adjust variables.")
            else:
                col_idx = {col: i for i, col in enumerate(nombres_columnas)}
                
                COLOR_NEW_PATIENT = '#87CEEB' 
                COLOR_HIST_READMIT = '#FF4444' 
                COLOR_HIST_SAFE = '#00C851'    
                COLOR_ARCHETYPE = '#FFD700' 
                SIZE_NEW_PATIENT = 2500 
                
                min_sim = min(similitudes_brutas) if len(similitudes_brutas) > 1 else similitudes_brutas[0]
                max_sim = max(similitudes_brutas) if len(similitudes_brutas) > 1 else similitudes_brutas[0]
                rango_sim = max_sim - min_sim if max_sim != min_sim else 1.0 
                
                G = nx.Graph()
                nodo_paciente = "Current\nAdmission"
                G.add_node(nodo_paciente, color=COLOR_NEW_PATIENT, size=SIZE_NEW_PATIENT, edge_color='black', line_width=3)
                
                nodos_gemelos = []
                info_inspeccion = {}
                invalid_markers = ["N/A", "MISSING_DATA", "NONE", "NAN", ""]
                
                cohort_outcomes, cohort_ages, cohort_sex, cohort_diagnoses, cohort_multimorbidity = [], [], [], [], []
                cohort_los, cohort_icu, cohort_consults, cohort_triage, cohort_ambulance = [], [], [], [], []
                cohort_med_cardio, cohort_med_neuro, cohort_perfil = [], [], []
                
                prefijos_nlp = ('LLM_', 'ING_', 'EVO_', 'DELTA_', 'rango_', 'pluripatologico', 'dias_', 'CIE10_MACRO', 'sexo', 'IN_COMPLEJIDAD', 'cantidad_interconsultas', 'visitas_', 'EST_', 'Area', 'HIST_condicion_ultimo_egreso', 'perfil_clinico_ingreso', 'Riesgo_')
                columnas_comunes_dinamicas = [col for col in nombres_columnas if str(col).startswith(prefijos_nlp)]
                
                for i, (idx, similitud_pct) in enumerate(zip(vecinos_idx, similitudes_brutas)):
                    reingreso_real = float(matriz_extended[idx, col_idx['target']])
                    cohort_outcomes.append(reingreso_real)
                    
                    color_nodo = COLOR_HIST_READMIT if reingreso_real == 1.0 else COLOR_HIST_SAFE
                    raw_ing = str(matriz_extended[idx, col_idx.get('texto_anamnesis_ingreso', -1)] if 'texto_anamnesis_ingreso' in col_idx else "")
                    raw_evo = str(matriz_extended[idx, col_idx.get('texto_evolucion_internacion', -1)] if 'texto_evolucion_internacion' in col_idx else "")
                    tiene_texto = (raw_ing.upper().strip() not in invalid_markers) or (raw_evo.upper().strip() not in invalid_markers)
                    icono_texto = " [TXT]" if tiene_texto else ""
                    
                    label_grafo = f"Case {i+1}{icono_texto}\n({similitud_pct:.1f}%)"
                    nodos_gemelos.append(label_grafo)
                    
                    norm_sim = (similitud_pct - min_sim) / rango_sim
                    tamaño_dinamico = 400 + (norm_sim * 1400)
                    
                    G.add_node(label_grafo, color=color_nodo, size=tamaño_dinamico, edge_color='white', line_width=1.5)
                    G.add_edge(nodo_paciente, label_grafo, weight=similitud_pct/10) 
                    
                    cohort_ages.append(matriz_extended[idx, col_idx.get('rango_edad', -1)])
                    cohort_sex.append(matriz_extended[idx, col_idx.get('sexo', -1)])
                    cohort_diagnoses.append(matriz_extended[idx, col_idx.get('CIE10_MACRO', -1)])
                    cohort_multimorbidity.append(matriz_extended[idx, col_idx.get('pluripatologico', -1)])
                    
                    cohort_los.append(matriz_extended[idx, col_idx.get('dias_internados', -1)])
                    cohort_icu.append(matriz_extended[idx, col_idx.get('EST_paso_por_uti', -1)])
                    cohort_consults.append(matriz_extended[idx, col_idx.get('cantidad_interconsultas', -1)])
                    cohort_triage.append(matriz_extended[idx, col_idx.get('TR_Prioridad', -1)])
                    cohort_ambulance.append(matriz_extended[idx, col_idx.get('EST_ingreso_ambulancia', -1)])
                    cohort_med_cardio.append(matriz_extended[idx, col_idx.get('Riesgo_Cardiovasculares_Inotropicos', -1)])
                    cohort_med_neuro.append(matriz_extended[idx, col_idx.get('Riesgo_Psicofarmacos_Neurologicos', -1)])
                    cohort_perfil.append(matriz_extended[idx, col_idx.get('perfil_clinico_ingreso', -1)])
                    
                    datos_gemelo = {
                        "similitud": similitud_pct,
                        "idx_matriz": idx,
                        "outcome_text": "Readmitted" if reingreso_real == 1.0 else "Safe Discharge",
                        "farmacos": matriz_extended[idx, col_idx.get('FARMACOS_TEXTO', -1)] if 'FARMACOS_TEXTO' in col_idx else "N/A",
                        "diagsec": matriz_extended[idx, col_idx.get('DIAGNOSTICOS_SEC_ACTIVOS', -1)] if 'DIAGNOSTICOS_SEC_ACTIVOS' in col_idx else "N/A",
                        "datos_comunes": {}
                    }
                    for col_comun in columnas_comunes_dinamicas:
                        datos_gemelo["datos_comunes"][col_comun] = matriz_extended[idx, col_idx[col_comun]]
                        
                    info_inspeccion[label_grafo] = datos_gemelo
        
                if len(vecinos_idx) > 1:
                    X_gemelos = X_train_dinamico[vecinos_idx]
                    dist_gemelos = pairwise_distances(X_gemelos, metric='cosine')
                    dist_triu = dist_gemelos[np.triu_indices_from(dist_gemelos, k=1)]
                    umbral_conexion = np.percentile(dist_triu, 30) if len(dist_triu) > 0 else 0
                    
                    for i in range(len(vecinos_idx)):
                        for j in range(i + 1, len(vecinos_idx)):
                            if dist_gemelos[i, j] < umbral_conexion:
                                peso_interno = max(0.1, 1 - dist_gemelos[i, j])
                                G.add_edge(nodos_gemelos[i], nodos_gemelos[j], weight=peso_interno * 2)
                
                arquetipo_label = None
                centrality = nx.harmonic_centrality(G)
                centrality.pop(nodo_paciente, None) 
                if centrality:
                    arquetipo_label = max(centrality, key=centrality.get)
                    G.nodes[arquetipo_label]['edge_color'] = COLOR_ARCHETYPE
                    G.nodes[arquetipo_label]['line_width'] = 4.5
                    info_inspeccion[arquetipo_label]["is_archetype"] = True
        
                fig, ax = plt.subplots(figsize=(8, 6))
                pos = nx.spring_layout(G, seed=42, k=0.85)
                
                node_colors = [data['color'] for node, data in G.nodes(data=True)]
                node_sizes = [data['size'] for node, data in G.nodes(data=True)]
                edge_colors = [data.get('edge_color', 'white') for node, data in G.nodes(data=True)]
                line_widths = [data.get('line_width', 1) for node, data in G.nodes(data=True)]
                
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='#A0A0A0')
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors=edge_colors, linewidths=line_widths, alpha=0.95)
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold', font_color='black')
                ax.axis('off')
                
                col_grafo, col_panel = st.columns([1.5, 1.2])
                
                with col_grafo:
                    st.pyplot(fig)
                    plt.close(fig)
                    with st.expander("🗺️ Topology Legend", expanded=False):
                        st.markdown("""
                        - 🫧 **Node Size:** Proportional to the match percentage.
                        - 🎨 **Node Colors:** 🟢 **Safe Discharge** | 🔴 **Readmitted** | 🔵 **Current Admission**
                        - 🌟 **Gold Border:** The **Archetypal Admission** (Local cluster anchor).
                        """)
                
                with col_panel:
                    sub_tab_global, sub_tab_inspector = st.tabs(["📊 Gap Analysis", "🔍 Case Inspector"])
                    
                    with sub_tab_global:
                        st.markdown(f"### Gap Analysis: Current Patient vs. Neighborhood (n={len(vecinos_idx)})")
                        
                        tasa_reingreso = (sum(cohort_outcomes) / len(cohort_outcomes)) * 100 if cohort_outcomes else 0.0
                        st.metric("Historical Readmission Rate of this Cohort", f"{tasa_reingreso:.1f}%")
                        st.progress(tasa_reingreso / 100)
                        
                        # --- EXTRACCIÓN DE DATOS DEL PACIENTE ACTUAL ---
                        p_age = format_clinical_value('rango_edad', df_paciente.get('rango_edad', pd.Series(['Unknown'])).iloc[0])
                        p_sex = format_clinical_value('sexo', df_paciente.get('sexo', pd.Series(['Unknown'])).iloc[0])
                        p_pluri = format_clinical_value('pluripatologico', df_paciente.get('pluripatologico', pd.Series(['0'])).iloc[0])
                        p_diag = format_clinical_value('CIE10_MACRO', df_paciente.get('CIE10_MACRO', pd.Series(['Unknown'])).iloc[0])
                        p_los = safe_int(df_paciente.get('dias_internados', pd.Series([0])).iloc[0])
                        p_icu = format_clinical_value('EST_paso_por_uti', df_paciente.get('EST_paso_por_uti', pd.Series(['0'])).iloc[0])
                        p_cons = safe_int(df_paciente.get('cantidad_interconsultas', pd.Series([0])).iloc[0])
                        p_amb = format_clinical_value('EST_ingreso_ambulancia', df_paciente.get('EST_ingreso_ambulancia', pd.Series(['0'])).iloc[0])
                        
                        st.markdown("#### 👤 Demographics & Profile Gap")
                        hombres = sum(1 for s in cohort_sex if str(s).strip().upper() == 'MASCULINO')
                        pct_hombres = (hombres / len(cohort_sex)) * 100 if cohort_sex else 0.0
                        pct_mujeres = 100 - pct_hombres if cohort_sex else 0.0
                        
                        c_dem_1, c_dem_2 = st.columns(2)
                        with c_dem_1:
                            st.markdown("**Current Patient**")
                            st.markdown(f"- **Gender:** {p_sex}")
                            st.markdown(f"- **Age:** {p_age}")
                            st.markdown(f"- **Multimorbidity:** {p_pluri}")
                            st.markdown(f"- **Diagnosis:** {p_diag}")
                        with c_dem_2:
                            st.markdown("**Cohort Majority**")
                            st.markdown(f"- **Gender:** {'Male' if pct_hombres > 50 else 'Female'} ({max(pct_hombres, pct_mujeres):.0f}%)")
                            
                            edades_traducidas = [format_clinical_value('rango_edad', e) for e in cohort_ages]
                            top_edad = pd.Series(edades_traducidas).mode()[0] if cohort_ages else "Unknown"
                            pct_edad = (edades_traducidas.count(top_edad) / len(cohort_ages)) * 100 if cohort_ages else 0
                            st.markdown(f"- **Age:** {top_edad} ({pct_edad:.0f}%)")
                            
                            pluri_count = sum(1 for p in cohort_multimorbidity if str(p).strip().upper() in ['1', '1.0', 'TRUE', 'YES'])
                            pct_pluri = (pluri_count / len(cohort_multimorbidity)) * 100 if cohort_multimorbidity else 0.0
                            st.markdown(f"- **Multimorbidity Present:** {pct_pluri:.0f}%")
                            
                            diagnosticos_traducidos = [format_clinical_value('CIE10_MACRO', d) for d in cohort_diagnoses]
                            top_diag = pd.Series(diagnosticos_traducidos).mode()[0] if cohort_diagnoses else "Unknown"
                            pct_diag = (diagnosticos_traducidos.count(top_diag) / len(cohort_diagnoses)) * 100 if cohort_diagnoses else 0
                            st.markdown(f"- **Diagnosis:** {top_diag} ({pct_diag:.0f}%)")

                        st.markdown("#### 🏥 Hospital Burden & Acuity Gap")
                        c1, c2, c3 = st.columns(3)
                        
                        median_los = pd.to_numeric(pd.Series(cohort_los), errors='coerce').median()
                        if pd.isna(median_los): median_los = 0.0
                        delta_los = p_los - median_los
                        c1.metric("Length of Stay", f"{p_los} days", delta=f"{delta_los:+.1f} vs Cohort ({median_los:.1f})", delta_color="inverse")
                        
                        icu_count = sum(1 for x in cohort_icu if str(x).strip().upper() in ['1', '1.0', 'TRUE', 'YES'])
                        pct_icu = (icu_count / len(cohort_icu)) * 100 if cohort_icu else 0.0
                        c2.metric("ICU Stay", f"{p_icu}", delta=f"Cohort Rate: {pct_icu:.1f}%", delta_color="off")
                        
                        avg_cons = pd.to_numeric(pd.Series(cohort_consults), errors='coerce').mean()
                        if pd.isna(avg_cons): avg_cons = 0.0
                        delta_cons = p_cons - avg_cons
                        c3.metric("Interconsultations", f"{p_cons}", delta=f"{delta_cons:+.1f} vs Cohort ({avg_cons:.1f})", delta_color="inverse")

                        c4, c5, c6 = st.columns(3)
                        
                        amb_count = sum(1 for x in cohort_ambulance if str(x).strip().upper() in ['1', '1.0', 'TRUE', 'YES'])
                        pct_amb = (amb_count / len(cohort_ambulance)) * 100 if cohort_ambulance else 0.0
                        c4.metric("Ambulance Arrival", f"{p_amb}", delta=f"Cohort Rate: {pct_amb:.1f}%", delta_color="off")
                        
                        triage_map = {
                            '0': '0: Non-Urgent', '0.0': '0: Non-Urgent',
                            '1': '1: Standard', '1.0': '1: Standard',
                            '2': '2: Urgent', '2.0': '2: Urgent',
                            '3': '3: Emergency', '3.0': '3: Emergency'
                        }
                        if 'TR_Prioridad' in df_paciente.columns:
                            p_triage_raw = str(df_paciente['TR_Prioridad'].iloc[0]).strip()
                            p_triage = triage_map.get(p_triage_raw, 'Unknown')
                        else:
                            p_triage = 'Unknown'
                            
                        triage_traducido = [triage_map.get(str(t).strip(), "Unknown") for t in cohort_triage] if cohort_triage else []
                        top_triage = pd.Series(triage_traducido).mode()[0] if triage_traducido else "Unknown"
                        pct_triage = (triage_traducido.count(top_triage) / len(triage_traducido)) * 100 if triage_traducido else 0
                        
                        p_triage_display = p_triage if p_triage == "Unknown" else p_triage[:4]
                        top_triage_display = top_triage if top_triage == "Unknown" else top_triage[:4]
                        
                        c5.metric("Triage Priority", f"{p_triage_display}", delta=f"Cohort Mode: {top_triage_display} ({pct_triage:.0f}%)", delta_color="off")
                        
                        if 'perfil_clinico_ingreso' in df_paciente.columns:
                            p_perfil = format_clinical_value('perfil_clinico_ingreso', df_paciente['perfil_clinico_ingreso'].iloc[0])
                        else:
                            p_perfil = "Unknown"
                            
                        perfil_traducido = [format_clinical_value('perfil_clinico_ingreso', p) for p in cohort_perfil] if cohort_perfil else []
                        perfiles_validos = [p for p in perfil_traducido if str(p).strip() not in ["N/A", "-1", "UNKNOWN", "Unknown"]]
                        top_perfil = pd.Series(perfiles_validos).mode()[0] if perfiles_validos else "Unknown"
                        pct_perfil = (perfiles_validos.count(top_perfil) / len(perfiles_validos)) * 100 if perfiles_validos else 0
                        
                        p_perfil_short = p_perfil.split(' ')[0] if p_perfil != "Unknown" else "Unknown"
                        top_perfil_short = top_perfil.split(' ')[0] if top_perfil != "Unknown" else "Unknown"
                        
                        c6.metric("Admission Profile", f"{p_perfil_short}", delta=f"Mode: {top_perfil_short} ({pct_perfil:.0f}%)", delta_color="off")

                    with sub_tab_inspector:
                        lista_nodos = list(info_inspeccion.keys())
                        if not lista_nodos:
                            st.info("No admissions available to inspect.")
                        else:
                            idx_arquetipo_selector = lista_nodos.index(arquetipo_label) if arquetipo_label in lista_nodos else 0
                            seleccion = st.selectbox("Inspect Similar Admission:", lista_nodos, index=idx_arquetipo_selector)
                            
                            if seleccion:
                                data = info_inspeccion[seleccion]
                                idx_gemelo_matriz = data['idx_matriz']
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric(label="Clinical Match", value=f"{data['similitud']:.1f}%")
                                with c2:
                                    if data['outcome_text'] == "Readmitted":
                                        st.error(f"**Outcome:**\n{data['outcome_text']}")
                                    else:
                                        st.success(f"**Outcome:**\n{data['outcome_text']}")
                                
                                st.markdown("---")
                                
                                if data.get("is_archetype"):
                                    st.markdown("""
                                    <div style='padding: 12px; background-color: rgba(255, 215, 0, 0.1); border-left: 4px solid #FFD700; border-radius: 4px; margin-bottom:15px;'>
                                        <h5 style='margin-top:0; color:#FFD700; font-size:14px;'>🎯 Micro-Cluster Anchor</h5>
                                        <p style='font-size:12px; margin-bottom:0;'>This historical admission represents the mathematical center of gravity for the current neighborhood.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("#### 📋 Shared Clinical Profile")
                                with st.container(height=280):
                                    for nombre_var_es, valor_var in data["datos_comunes"].items():
                                        nombre_en = TRANSLATION_DICT.get(nombre_var_es, nombre_var_es.replace('_', ' ').title())
                                        valor_en = format_clinical_value(nombre_var_es, valor_var)
                                        st.markdown(f"**{nombre_en}:** {valor_en}")
                                
                                st.markdown("---")
                                diagsec_en = traducir_ninguno(data['diagsec'])
                                farmacos_raw = data['farmacos']
                                
                                st.markdown("#### 🏥 Retrospective Details")
                                st.markdown(f"**Secondary Diagnoses:**\n{diagsec_en}")
                                st.markdown(f"**Medications:**")
                                
                                if str(farmacos_raw).strip().upper() in ('NINGUNO', '', 'NONE', 'N/A'):
                                    st.markdown("None")
                                else:
                                    lista_farmacos = [f.strip() for f in str(farmacos_raw).split(',')]
                                    try:
                                        from translations import FARMACOS_TRANSLATION_DICT
                                        dict_farmacos_upper = {k.upper(): v for k, v in FARMACOS_TRANSLATION_DICT.items()}
                                        lista_traducida = [dict_farmacos_upper.get(f.upper(), f.strip().title()) for f in lista_farmacos]
                                    except ImportError:
                                        lista_traducida = [f.strip().title() for f in lista_farmacos]
                                        
                                    for f in lista_traducida:
                                        st.markdown(f"- {f}")
                                
                                st.markdown("#### 📜 Narrative Phenotype")
                                raw_ing = str(matriz_extended[idx_gemelo_matriz, col_idx.get('texto_anamnesis_ingreso', -1)] if 'texto_anamnesis_ingreso' in col_idx else "")
                                raw_evo = str(matriz_extended[idx_gemelo_matriz, col_idx.get('texto_evolucion_internacion', -1)] if 'texto_evolucion_internacion' in col_idx else "")
                                
                                texto_ing = "" if raw_ing.upper().strip() in invalid_markers else raw_ing
                                texto_evo = "" if raw_evo.upper().strip() in invalid_markers else raw_evo
                                
                                if not texto_ing and not texto_evo:
                                    st.info("ℹ️ No narrative clinical notes available for this admission.")
                                else:
                                    texto_completo = ""
                                    if texto_ing: texto_completo += f"**Admission:**\n{texto_ing}\n\n"
                                    if texto_evo: texto_completo += f"**Evolution:**\n{texto_evo}"
                                    
                                    citas_gemelo = {} 
                                    for col_nombre in nombres_columnas:
                                        if col_nombre.startswith("TX_"):
                                            cita_val = str(matriz_extended[idx_gemelo_matriz, col_idx[col_nombre]])
                                            if cita_val and cita_val.strip() not in ["nan", "None", "", "N/A"]:
                                                var_original = col_nombre.replace("TX_", "")
                                                var_traducida = TRANSLATION_DICT.get(var_original, var_original.replace('_', ' ').title())
                                                citas_gemelo[cita_val.strip()] = var_traducida
                                    
                                    enfermedades_a_resaltar = [
                                        "diabetes", "hipertensión", "epoc", "neumonía", "tuberculosis", "iam", "acv", "cáncer",
                                        "trombosis", "celulitis", "plaquetopenia", "fa", "fibrilación auricular", "insuficiencia cardíaca",
                                        "sepsis", "infarto", "arritmia", "infección", "isquemia", "shock", "aneurisma",
                                        "hipertensión arterial", "hta", "hipotensión", "bradicardia", "taquicardia",
                                        "asma", "bronquitis", "derrame pleural", "edema agudo de pulmón",
                                        "insuficiencia respiratoria", "itu", "infección urinaria", "insuficiencia renal",
                                        "hipotiroidismo", "hipertiroidismo", "cetoacidosis", "hipoglucemia", "hiperglucemia",
                                        "dislipidemia", "obesidad", "desnutrición", "sme metabólico",
                                        "convulsión", "epilepsia", "demencia", "alzheimer", "parkinson", "delirium",
                                        "cirrosis", "hepatitis", "pancreatitis", "colecistitis", "apendicitis", "peritonitis",
                                        "hemorragia digestiva", "úlcera", "obstrucción intestinal", "gastroenteritis",
                                        "anemia", "leucemia", "linfoma", "neutropenia", "coagulopatía", "metástasis",
                                        "bacteriemia", "shock séptico", "covid", "osteomielitis", "vih", "dengue", "tbc",
                                        "fractura", "luxación", "artrosis", "artritis", "úlcera por presión", "escara"
                                    ]
                                    texto_html = renderizar_notas_gemelo(texto_completo, citas_gemelo, enfermedades_a_resaltar)
                                    
                                    with st.expander("🔍 Inspect Original Clinical Notes", expanded=False):
                                        st.caption("🟡 **Yellow:** Phenotype Evidence | 🔴 **Red:** Disease Mention")
                                        st.markdown(texto_html, unsafe_allow_html=True)

    except Exception as e:
        st.error("Error generating similarity topology graph.")
        st.warning(f"Technical Detail: {str(e)}")

# ==========================================
# 9. GLOBAL UNIVERSE (UMAP) IN NEW TAB
# ==========================================
with tab_umap:
    st.markdown("#### Global Clinical Universe Mapping (UMAP)")
    try:
        with st.spinner("Projecting multi-dimensional space and calculating topological insights..."):
            modo_color = st.radio(
                "🎨 Select UMAP Coloring Mode:",
                ["Readmitted vs Safe Discharge", "Age Distribution", "Multimorbidity"],
                horizontal=True
            )
            
            col_mapa, col_insights = st.columns([2.2, 1.2])
            
            fig_umap = go.Figure()
            borde_marcador = dict(width=0.6, color='rgba(255,255,255,0.6)')
            leyenda_formas = "(⚪ Safe | 🔶 Readmit)" 
            
            col_idx = {col: i for i, col in enumerate(nombres_columnas)}
            
            y_hist_global = matriz_extended[:, col_idx['target']].astype(float)
            mask_safe_global = (y_hist_global == 0)
            mask_readmit_global = (y_hist_global == 1)
            
            total_internaciones = len(y_hist_global)
            tasa_reingreso_base = (np.sum(y_hist_global) / total_internaciones) * 100 if total_internaciones > 0 else 0.0
            
            col_edad = col_idx.get('rango_edad')
            edades_global = matriz_extended[:, col_edad] if col_edad is not None else np.full(total_internaciones, 'Unknown')
            
            col_diag = col_idx.get('CIE10_MACRO')
            diag_global = matriz_extended[:, col_diag] if col_diag is not None else np.full(total_internaciones, 'Unknown')
            
            col_pluri = col_idx.get('pluripatologico')
            pluri_global = matriz_extended[:, col_pluri] if col_pluri is not None else np.zeros(total_internaciones)

            col_dias = col_idx.get('dias_internados')
            dias_global_raw = matriz_extended[:, col_dias] if col_dias is not None else np.zeros(total_internaciones)
            dias_global_num = pd.to_numeric(dias_global_raw, errors='coerce')
            dias_global_num = np.nan_to_num(dias_global_num, nan=0.0)
            
            col_visitas = col_idx.get('visitas_guardia_6meses_previos')
            visitas_global_raw = matriz_extended[:, col_visitas] if col_visitas is not None else np.zeros(total_internaciones)
            visitas_global_num = pd.to_numeric(visitas_global_raw, errors='coerce')
            visitas_global_num = np.nan_to_num(visitas_global_num, nan=0.0)
            
            hover_texts_global = []
            for i in range(len(y_hist_global)):
                outcome_str = "Readmitted" if y_hist_global[i] == 1 else "Safe Discharge"
                diag_val = format_clinical_value('CIE10_MACRO', diag_global[i])
                edad_val = format_clinical_value('rango_edad', edades_global[i])
                dias_val = safe_int(dias_global_raw[i])
                hover_texts_global.append(
                    f"<b>Outcome:</b> {outcome_str}<br>"
                    f"<b>Diagnosis:</b> {diag_val}<br>"
                    f"<b>Age:</b> {edad_val}<br>"
                    f"<b>Stay:</b> {dias_val} days"
                )
            hover_texts_global = np.array(hover_texts_global)

            def agregar_capa_umap(fig, mascara_filtro, color_hex, nombre_grupo, es_activo=True):
                grupo_legend_id = str(nombre_grupo).replace(" ", "_").lower()
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    name=f"{nombre_grupo} {leyenda_formas}", legendgroup=grupo_legend_id,
                    marker=dict(color=color_hex, size=10, symbol='square') 
                ))
                
                fig.add_trace(go.Scatter(
                    x=umap_embeddings[mascara_filtro & mask_safe_global, 0], 
                    y=umap_embeddings[mascara_filtro & mask_safe_global, 1],
                    mode='markers', legendgroup=grupo_legend_id, showlegend=False, 
                    text=hover_texts_global[mascara_filtro & mask_safe_global], hoverinfo='text',
                    marker=dict(color=color_hex, size=5 if es_activo else 4, opacity=0.5 if es_activo else 0.3, symbol='circle', line=borde_marcador)
                ))
                
                fig.add_trace(go.Scatter(
                    x=umap_embeddings[mascara_filtro & mask_readmit_global, 0], 
                    y=umap_embeddings[mascara_filtro & mask_readmit_global, 1],
                    mode='markers', legendgroup=grupo_legend_id, showlegend=False, 
                    text=hover_texts_global[mascara_filtro & mask_readmit_global], hoverinfo='text',
                    marker=dict(color=color_hex, size=8 if es_activo else 6, opacity=0.9 if es_activo else 0.5, symbol='diamond', line=dict(color='white', width=0.8) if es_activo else borde_marcador)
                ))
            
            if 'vecinos_idx_pool' not in locals() and 'vecinos_idx_pool' not in globals():
                # BLOQUE DE RESCATE: Si no se calculó en tab_evidencia, lo hacemos aquí con el filtro
                prep = pipeline.named_steps['preprocesador']
                X_pac_proc = prep.transform(df_paciente)
                X_pac_dense = X_pac_proc.toarray() if hasattr(X_pac_proc, 'toarray') else np.array(X_pac_proc)
                
                # --- APLICAR MÁSCARA AL PACIENTE (RESCATE) ---
                X_pac_limpio = X_pac_dense[:, mask_limpia]
                
                distancias_rescue, indices_rescue = knn.kneighbors(X_pac_limpio)
                vecinos_idx_pool = indices_rescue[0]
                
            local_idx = vecinos_idx_pool[:20] 
            
            with col_mapa:
                if modo_color == "Readmitted vs Safe Discharge":
                    fig_umap.add_trace(go.Scatter(
                        x=umap_embeddings[mask_safe_global, 0], y=umap_embeddings[mask_safe_global, 1],
                        mode='markers', name='Safe Discharge',
                        text=hover_texts_global[mask_safe_global], hoverinfo='text',
                        marker=dict(color='#00C851', size=5, opacity=0.5, symbol='circle', line=borde_marcador)
                    ))
                    fig_umap.add_trace(go.Scatter(
                        x=umap_embeddings[mask_readmit_global, 0], y=umap_embeddings[mask_readmit_global, 1],
                        mode='markers', name='Readmitted',
                        text=hover_texts_global[mask_readmit_global], hoverinfo='text',
                        marker=dict(color='#FF4444', size=8, opacity=0.9, symbol='diamond', line=dict(color='white', width=0.8))
                    ))
                
                elif modo_color == "Age Distribution":
                    edades_trad_global = np.array([format_clinical_value('rango_edad', e) for e in edades_global])
                    unique_ages = sorted(list(set(edades_trad_global)))
                    color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
                    
                    for idx_color, age_group in enumerate(unique_ages):
                        mask_age = edades_trad_global == age_group
                        color_asignado = color_palette[idx_color % len(color_palette)]
                        agregar_capa_umap(fig_umap, mask_age, color_asignado, age_group, es_activo=True)
                        
                elif modo_color == "Multimorbidity":
                    mask_multi_yes = np.array([str(x).strip().upper() in ['1', '1.0', 'TRUE', 'YES'] for x in pluri_global])
                    mask_multi_no = ~mask_multi_yes
                    
                    agregar_capa_umap(fig_umap, mask_multi_no, '#AAB7B8', 'No Multimorbidity', es_activo=True)
                    agregar_capa_umap(fig_umap, mask_multi_yes, '#D2691E', 'Multimorbidity Present', es_activo=True)
                
                paciente_umap_coords = np.mean(umap_embeddings[vecinos_idx_pool[:3]], axis=0, keepdims=True)
                diag_paciente = format_clinical_value('CIE10_MACRO', df_paciente['CIE10_MACRO'].iloc[0] if 'CIE10_MACRO' in df_paciente.columns else 'N/A')
                edad_paciente_raw = df_paciente['rango_edad'].iloc[0] if 'rango_edad' in df_paciente.columns else 'N/A'
                edad_paciente = format_clinical_value('rango_edad', edad_paciente_raw)
                dias_paciente = safe_int(df_paciente['dias_internados'].iloc[0] if 'dias_internados' in df_paciente.columns else 0)
                
                paciente_hover = (f"<b>CURRENT ADMISSION</b><br>"
                                  f"<b>Diagnosis:</b> {diag_paciente}<br>"
                                  f"<b>Age:</b> {edad_paciente}<br>"
                                  f"<b>Stay:</b> {dias_paciente} days")
    
                fig_umap.add_trace(go.Scatter(
                    x=[paciente_umap_coords[0, 0]], y=[paciente_umap_coords[0, 1]],
                    mode='markers', name='Current Admission',
                    text=[paciente_hover], hoverinfo='text',
                    marker=dict(color='#87CEEB', size=20, symbol='star', line=dict(color='black', width=2.5))
                ))
                
                fig_umap.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=10, r=10, t=10, b=10), height=550,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_umap, use_container_width=True)

            with col_insights:
                st.markdown("### 📊 Clinical Topology Analysis")
                
                outcomes_locales = y_hist_global[local_idx]
                tasa_reingreso_local = (np.sum(outcomes_locales) / len(outcomes_locales)) * 100 if len(outcomes_locales) > 0 else tasa_reingreso_base
                
                st.markdown(
                    f"""
                    <div style='padding:10px; background-color:rgba(128,128,128,0.1); border-radius:5px; margin-bottom:15px; border-left: 4px solid #1E90FF;'>
                        <p style='margin:0; font-size:12px; color:gray;'>CENSUS PROJECTION (N = {total_internaciones:,})</p>
                        <p style='margin:0; font-size:15px;'>Global Hospital Readmission Rate: <b>{tasa_reingreso_base:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True
                )

                if modo_color == "Readmitted vs Safe Discharge":
                    st.markdown("#### 📍 Historical Outcomes Overview")
                    st.markdown(f"- **Local Cluster Readmission Rate:** `{tasa_reingreso_local:.1f}%`")
                    st.markdown("---")
                    
                    st.markdown("**Descriptive Observation:**")
                    if tasa_reingreso_local < tasa_reingreso_base - 3:
                        insight_txt = f"This admission maps to a cluster where historical readmissions ({tasa_reingreso_local:.1f}%) are lower than the hospital average. Similar past cases have predominantly resulted in safe discharges."
                    elif tasa_reingreso_local > tasa_reingreso_base + 5:
                        insight_txt = f"This admission maps to a cluster where historical readmissions ({tasa_reingreso_local:.1f}%) are notably higher than the hospital average. Statistical association suggests a complex profile."
                    else:
                        insight_txt = f"This admission maps to a cluster where historical readmissions ({tasa_reingreso_local:.1f}%) closely follow the hospital average. Outcomes for similar past cases show a mixed distribution."
                        
                    st.markdown(f"<div style='font-size:14px; line-height:1.5;'>{insight_txt}</div>", unsafe_allow_html=True)

                elif modo_color == "Age Distribution":
                    edades_trad_global = np.array([format_clinical_value('rango_edad', e) for e in edades_global])
                    edades_locales = np.array([format_clinical_value('rango_edad', e) for e in edades_global[local_idx]])
                    serie_edades_locales = pd.Series(edades_locales)
                    distribucion_local = serie_edades_locales.value_counts(normalize=True) * 100
                    
                    st.markdown("#### ⏳ Age Cohort Distribution")
                    st.markdown("**Local Cluster Breakdown:**")
                    for edad_cat, pct_local in distribucion_local.items():
                        pct_global = (np.sum(edades_trad_global == edad_cat) / total_internaciones) * 100 if total_internaciones > 0 else 0
                        st.markdown(f"- {edad_cat}: `{pct_local:.1f}%` *(Global: {pct_global:.1f}%)*")
                    
                    st.markdown("---")
                    st.markdown("**Descriptive Observation:**")
                    
                    edad_dominante_local = serie_edades_locales.mode()[0] if not serie_edades_locales.empty else "Unknown"
                    if edad_paciente != edad_dominante_local and edad_paciente not in ["Unknown", "N/A"]:
                        insight_txt = f"The case maps to a geometric cluster where the most frequent demographic is {edad_dominante_local}, differing from the current patient's chronological classification ({edad_paciente}). The grouping is driven by statistical similarities across clinical text and multi-dimensional factors rather than age constraints."
                    else:
                        insight_txt = f"The patient's chronological age class aligns with the dominant demographic ({edad_dominante_local}) of this local cluster, representing a statistically typical presentation for this cohort within historical records."
                    
                    st.markdown(f"<div style='font-size:14px; line-height:1.5;'>{insight_txt}</div>", unsafe_allow_html=True)

                elif modo_color == "Multimorbidity":
                    mask_multi_yes = np.array([str(x).strip().upper() in ['1', '1.0', 'TRUE', 'YES'] for x in pluri_global])
                    pct_pluri_global = (np.sum(mask_multi_yes) / total_internaciones) * 100 if total_internaciones > 0 else 0
                    
                    pluri_local_raw = pluri_global[local_idx]
                    mask_multi_local = np.array([str(x).strip().upper() in ['1', '1.0', 'TRUE', 'YES'] for x in pluri_local_raw])
                    pct_pluri_local = np.mean(mask_multi_local) * 100 if len(mask_multi_local) > 0 else 0
                    
                    st.markdown("#### 🏥 Multimorbidity Context")
                    st.markdown(f"- **Local Cluster Multimorbidity Density:** `{pct_pluri_local:.1f}%`")
                    st.markdown(f"- **Global Hospital Multimorbidity Rate:** `{pct_pluri_global:.1f}%`")
                    st.markdown("---")
                    
                    st.markdown("**Descriptive Observation:**")
                    if pct_pluri_local < 30 and tasa_reingreso_local > tasa_reingreso_base:
                        insight_txt = f"This cluster exhibits an increased historical readmission rate ({tasa_reingreso_local:.1f}%) alongside a low density of chronic multimorbidity ({pct_pluri_local:.1f}%), pointing toward mathematical similarities driven by acute clinical profiles, specialized procedures, or alternative non-chronic variables."
                    elif pct_pluri_local > 70:
                        insight_txt = f"This neighborhood is heavily saturated with multimorbidity ({pct_pluri_local:.1f}%), a baseline historically associated with complex longitudinal management and coordination of multiple disease tracks."
                    else:
                        insight_txt = f"The cluster contains a balanced distribution of chronic complexity, suggesting that past outcomes in this specific map region are shaped by a combination of acute severity and underlying chronic baselines."
                        
                    st.markdown(f"<div style='font-size:14px; line-height:1.5;'>{insight_txt}</div>", unsafe_allow_html=True)

                st.markdown("---")
                
                if tasa_reingreso_local > tasa_reingreso_base + 5:
                    box_color = "#FFBB33" 
                    box_title = "Cluster Summary: Elevated Historical Risk"
                elif tasa_reingreso_local < tasa_reingreso_base - 3:
                    box_color = "#00C851" 
                    box_title = "Cluster Summary: Standard Historical Risk"
                else:
                    box_color = "#33b5e5" 
                    box_title = "Cluster Summary: Average Historical Risk"
                
                if len(local_idx) > 0:
                    local_pluri_vals = [str(x).strip().upper() in ['1', '1.0', 'TRUE', 'YES'] for x in pluri_global[local_idx]]
                    pct_pluri_local_str = f"{(np.sum(local_pluri_vals) / len(local_pluri_vals)) * 100:.0f}%"
                    
                    diags_locales = [format_clinical_value('CIE10_MACRO', d) for d in diag_global[local_idx]]
                    diag_dominante = pd.Series(diags_locales).mode()[0] if diags_locales else "Unknown"
                    
                    los_global_mean = np.mean(dias_global_num)
                    los_local_mean = np.mean(dias_global_num[local_idx])
                    
                    visitas_local_mean = np.mean(visitas_global_num[local_idx])
                else:
                    pct_pluri_local_str = "N/A"
                    diag_dominante = "Unknown"
                    los_global_mean, los_local_mean, visitas_local_mean = 0.0, 0.0, 0.0
                
                st.markdown(f"#### 📋 {box_title}")
                st.markdown(
                    f"""
                    <div style='padding: 12px; background-color: {box_color}15; border-left: 4px solid {box_color}; border-radius: 4px;'>
                        <p style='margin: 0 0 8px 0; font-size:13px;'>The current admission aligns with a topological neighborhood characterized by:</p>
                        <ul style='margin: 0 0 8px 0; font-size:13px; padding-left: 20px;'>
                            <li>A historical readmission rate of <b>{tasa_reingreso_local:.1f}%</b>.</li>
                            <li>A dominant primary diagnosis of <b>{diag_dominante}</b>.</li>
                            <li>A multimorbidity prevalence of <b>{pct_pluri_local_str}</b> among nearby cases.</li>
                            <li>An average Length of Stay of <b>{los_local_mean:.1f} days</b> (vs. Global Hospital Mean: {los_global_mean:.1f} days).</li>
                            <li>An average of <b>{visitas_local_mean:.1f} ER visits</b> within the 6 months prior to admission.</li>
                        </ul>
                        <p style='margin: 0; font-size:12px; color: gray;'><i>Clinical Note: We recommend contextualizing these statistical associations with your clinical judgment or utilizing the Counterfactual Simulator to explore modifiable risk variables.</i></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error("Error generating UMAP projection and insights.")
        st.warning(f"Technical Detail: {str(e)}")

# ==========================================
# 10. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab_eda:
    st.markdown("#### Global Exploratory Clinical Insights")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_npy_eda = os.path.join(BASE_DIR, 'dataset_optimizado_eda.npy')
    
    eda_engine = MotorEDADinamico(ruta_npy_eda)
    datos_cargados = eda_engine.cargar_datos()
    
    if not datos_cargados:
        st.error(f"Please ensure 'dataset_optimizado_eda.npy' is uploaded to the application directory to view the EDA.")
    else:
        # Menú de selección interactivo principal
        col_selec1, col_selec2 = st.columns(2)
        
        with col_selec1:
            tipo_grafico = st.selectbox(
                "Select Analysis Dimension:",
                options=[
                    "Failure Velocity (Cumulative Incidence)",
                    "Clinical Profile (Multimorbidity)",
                    "Hospital Attrition (Severity)",
                    "The Revolving Door (ER History)",
                    "Clinical Signature (ICD-10 Heatmap)",
                    "Socioeconomic Context (Employment)"
                ]
            )
            
        # Mapeo de la selección del usuario a las funciones de la clase
        mapa_graficos = {
            "Failure Velocity (Cumulative Incidence)": "curva",
            "Clinical Profile (Multimorbidity)": "clinico",
            "Hospital Attrition (Severity)": "gravedad",
            "The Revolving Door (ER History)": "historial",
            "Clinical Signature (ICD-10 Heatmap)": "cie10",
            "Socioeconomic Context (Employment)": "social"
        }
        
        id_grafico = mapa_graficos[tipo_grafico]
        
        # --- ENRUTADOR DE FILTROS DINÁMICO ---
        filtro_seleccionado = 'EST_paso_por_uti' # Fallback seguro
        modo_visual = 'absolute'
        
        if id_grafico == "curva":
            with col_selec2:
                filtro_curva = st.selectbox(
                    "Select Segmentation Variable:",
                    options=[
                        "ICU Stay",
                        "Triage Priority", 
                        "Multimorbidity",
                        "ER Visits (Previous 6 months)",
                        
                        "Employment Status"
                    ]
                )
                
            mapa_filtros = {
                "ICU Stay": "EST_paso_por_uti",
                "Triage Priority": "TR_Prioridad",
                "Multimorbidity": "pluripatologico",
                "ER Visits (Previous 6 months)": "visitas_guardia_6meses_previos",
                
                "Employment Status": "PA_SITLABO_x"
            }
            filtro_seleccionado = mapa_filtros[filtro_curva]
            
        elif id_grafico == "gravedad":
            with col_selec2:
                filtro_gravedad = st.selectbox(
                    "Select Severity Metric:",
                    options=["ICU Stay", "Triage Priority"] 
                )
                
            mapa_filtros_gravedad = {
                "ICU Stay": "EST_paso_por_uti",
                "Triage Priority": "TR_Prioridad"
            }
            filtro_seleccionado = mapa_filtros_gravedad[filtro_gravedad]
            
        elif id_grafico in ["clinico", "social", "historial"]:
            with col_selec2:
                sel_modo = st.radio(
                    "Display Mode:", 
                    options=["Absolute (Volume)", "Relative (Composition %)"], 
                    horizontal=True
                )
                modo_visual = 'relative' if '%' in sel_modo else 'absolute'
                
        st.markdown("---")
        
        # Generar y renderizar el gráfico seleccionado
        with st.spinner("Generating Insights..."):
            fig_eda = eda_engine.analizar(
                id_grafico, 
                variable_segmentacion=filtro_seleccionado, 
                modo=modo_visual
            )
            
            if fig_eda:
                st.plotly_chart(fig_eda, use_container_width=True)
            else:
                st.warning("Insufficient data or missing columns in the dataset to render this specific chart.")
