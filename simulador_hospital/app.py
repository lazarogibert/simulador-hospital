
import streamlit as st
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
warnings.filterwarnings("ignore")

# ==========================================
# 0. FUNCIONES DE PROCESAMIENTO CLÍNICO (CIE-10)
# (ESTE BLOQUE DEBE PERMANECER EN ESPAÑOL PARA EL BACKEND)
# ==========================================

def normalizar_cie10(codigo):
    if pd.isna(codigo): return pd.NA
    codigo = str(codigo).strip().upper().replace(".", "").replace(" ", "")
    m = re.match(r'^([A-Z])(\d{2})', codigo)
    if not m: return pd.NA
    return f"{m.group(1)}{int(m.group(2)):02d}"

def mapear_cie10_macro(cod):
    # Si viene nulo del normalizador previo, salimos rápido
    if pd.isna(cod):
        return "Desconocido"

    # Extracción directa, confiando en el normalizador
    letra = cod[0]
    num = int(cod[1:3])

    # --- INFECCIOSAS Y PARASITARIAS CRÓNICAS (A y B) ---
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

    # --- ONCOLOGÍA Y HEMATOLOGÍA (C y D) ---
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

    # --- ENDOCRINAS Y METABÓLICAS (E) ---
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

    # --- PSIQUIATRÍA Y SALUD MENTAL (F) ---
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

    # --- NEUROLOGÍA (G) ---
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

    # --- SENTIDOS (H) ---
    if letra == "H":
        if 0 <= num <= 59: return "Ojo"
        elif 60 <= num <= 95: return "Oído"
        else: return "Otros órganos de los sentidos"

    # --- CARDIOVASCULAR (I) ---
    if letra == "I":
        if 10 <= num <= 15: return "Hipertensión"
        elif 20 <= num <= 25: return "Cardiopatía isquémica"
        elif 26 <= num <= 28: return "Enfermedad cardiopulmonar"
        elif 30 <= num <= 52: return "Otras enfermedades del corazón (Insuficiencia Cardíaca)"
        elif 60 <= num <= 69: return "Cerebrovascular"
        elif 70 <= num <= 79: return "Enfermedades de arterias y capilares"
        elif 80 <= num <= 89: return "Enfermedades de venas y vasos linfáticos"
        else: return "Otros circulatorios"

    # --- RESPIRATORIO (J) ---
    if letra == "J":
        if 0 <= num <= 6: return "Vías respiratorias altas"
        elif 9 <= num <= 18: return "Infecciones agudas / neumonía / influenza"
        elif 20 <= num <= 22: return "Infecciones respiratorias bajas"
        elif 30 <= num <= 39: return "Enfermedades de vías respiratorias superiores"
        elif 40 <= num <= 47: return "Asma / EPOC / bronquitis"
        elif 60 <= num <= 70: return "Enfermedades del pulmón por agentes externos (Neumoconiosis)"
        elif 80 <= num <= 84: return "Enfermedades pulmonares intersticiales"
        else: return "Otros respiratorios"

    # --- DIGESTIVO (K) ---
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

    # --- DERMATOLOGÍA (L) ---
    if letra == "L":
        if 20 <= num <= 30: return "Dermatitis y eczema"
        elif 40 <= num <= 45: return "Trastornos papuloescamosos (Psoriasis)"
        elif 50 <= num <= 54: return "Urticaria y eritema"
        elif 80 <= num <= 99: return "Trastornos de las faneras / Otros trastornos de piel"
        else: return "Otras enfermedades de la piel"

    # --- OSTEOMUSCULAR (M) ---
    if letra == "M":
        if 0 <= num <= 25: return "Artropatías"
        elif 30 <= num <= 36: return "Tejido conectivo (Lupus, etc.)"
        elif 40 <= num <= 54: return "Dorsopatías"
        elif 60 <= num <= 79: return "Tejidos blandos"
        elif 80 <= num <= 94: return "Osteopatías y condropatías (Osteoporosis)"
        else: return "Otros osteomusculares"

    # --- GENITOURINARIO (N) ---
    if letra == "N":
        if 0 <= num <= 29: return "Riñón (Insuficiencia Renal Crónica)"
        elif 30 <= num <= 39: return "Vías urinarias bajas"
        elif 40 <= num <= 51: return "Genital masculino (Hiperplasia Prostática)"
        elif 60 <= num <= 64: return "Mama"
        elif 70 <= num <= 98: return "Genital femenino (Endometriosis, etc.)"
        else: return "Otros genitourinarios"
            
    # --- CONGÉNITAS (Q) ---
    if letra == "Q":
        if 0 <= num <= 7: return "Malformaciones del sistema nervioso (Espina bífida)"
        elif 20 <= num <= 28: return "Malformaciones cardíacas congénitas"
        elif 90 <= num <= 99: return "Anomalías cromosómicas (Síndrome de Down)"
        else: return "Otras malformaciones congénitas"

    # --- CONDICIONES PERINATALES CRÓNICAS (P) ---
    if letra == "P":
        if num == 27: return "Enfermedad respiratoria crónica perinatal"
        else: return pd.NA

    # --- SECUELAS DE TRAUMATISMOS (T) ---
    if letra == "T":
        if 90 <= num <= 98: return "Secuelas crónicas de traumatismos"
        else: return pd.NA
            
    # --- CONDICIONES ESPECIALES Y POST-COVID (U) --- 
    if letra == "U":
        if num == 9: return "Síndrome Post-COVID (Long COVID)"
        else: return "Otras condiciones especiales (U)"

    # --- ESTADO DE SALUD Y DISPOSITIVOS (Z) ---
    if letra == "Z":
        if 85 <= num <= 87: return "Historia personal de tumores / enfermedades"
        elif 89 <= num <= 90: return "Ausencia adquirida de miembros / órganos"
        elif 93 <= num <= 93: return "Aberturas artificiales (Ostomías)"
        elif 94 <= num <= 94: return "Estado de órgano trasplantado"
        elif 95 <= num <= 95: return "Presencia de implantes cardíacos / vasculares"
        elif 99 <= num <= 99: return "Dependencia de máquinas (diálisis, oxígeno)"
        else: return "Otros factores de salud"

    return "Desconocido"

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Safe Discharge Simulator", layout="wide")
st.title("🏥 Clinical Safe Discharge Simulator (15 Days)")
st.markdown("Decision support tool based on narrative phenotypes and prescriptive explainability.")

# ==========================================
# 2. MODEL AND DATA LOADING (CACHE)
# ==========================================
@st.cache_resource
def cargar_entorno():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(directorio_actual, 'modelo_reingreso_nlp_41vars_v3rf.pkl')
    ruta_datos = os.path.join(directorio_actual, 'train_sample_quimera.csv')
    
    paquete = joblib.load(ruta_modelo)
    pipeline = paquete['pipeline']
    umbral = paquete['umbral']
    cols_modelo = paquete['nombres_columnas']
    
    try:
        df_train_sample = pd.read_csv(ruta_datos)
    except FileNotFoundError:
        st.error(f"⚠️ Missing file at: {ruta_datos}")
        df_train_sample = pd.DataFrame(np.zeros((50, len(cols_modelo))), columns=cols_modelo)
    
    return pipeline, umbral, cols_modelo, df_train_sample

pipeline, umbral, columnas_modelo, df_train_sample = cargar_entorno()

# ==========================================
# 3. INTERFACE CAPTURE (MANUAL CORE & NLP AUTOMATION)
# ==========================================
st.sidebar.header("🩺 Phenotype Loading")

opciones_edad_dict = {"Minor": "Menor de edad", "Young Adult": "Adulto Joven", "Middle-aged Adult": "Adulto de mediana edad", "Older Adult": "Adulto mayor", "Elderly": "Anciano"}
af_dict = {'Autoimmune': 'autoinmune', 'Other Cardiovascular': 'cardiovascular_otro', 'Diabetes': 'diabetes', 'Hypertension': 'hipertension', 'Other Metabolic': 'metabolico_otro', 'Neurological': 'neurologico', 'Oncological': 'oncologico', 'Psychiatric': 'psiquiatrico', 'Renal': 'renal', 'Respiratory': 'respiratorio'}
cro_dict = {'Medication Abandonment': 'abandono_medicacion', 'Alcoholism': 'alcoholismo', 'Severe Malnutrition': 'desnutricion_severa', 'Illicit Drugs': 'drogas_ilicitas', 'Geriatric Frailty': 'fragilidad_geriatrica', 'History of Falls': 'historial_caidas', 'Oxygen Dependent': 'oxigenodependiente', 'Polypharmacy': 'polifarmacia', 'Active Smoking': 'tabaquismo_activo'}
ing_dict = {'Mental Alteration': 'alteracion_mental', 'Repeated Consultations': 'consultas_reiteradas', 'Functional Dependency': 'dependencia_funcional', 'Device Bearer': 'portador_dispositivos', 'Hemorrhagic Risk': 'riesgo_hemorragico'}
evo_dict = {'Infectious Isolation': 'aislamiento_infeccioso', 'Mental Alteration': 'alteracion_mental', 'Hospitalization Complication': 'complicacion_internacion', 'Palliative Care': 'cuidados_paliativos', 'Functional Dependency': 'dependencia_funcional', 'Irregular Discharge / Escape': 'fuga_o_alta_irregular', 'Device Bearer': 'portador_dispositivos', 'Pressure Ulcers': 'ulceras_presion'}

# Inicialización de estado
for key in ["ui_af_sel", "ui_cro_sel", "ui_ing_sel", "ui_evo_sel"]:
    if key not in st.session_state: st.session_state[key] = []
for key in ["ui_ing_dolor", "ui_ing_grav", "ui_evo_dolor", "ui_evo_grav"]:
    if key not in st.session_state: st.session_state[key] = 0 if 'dolor' in key else 5
if 'nlp_processed' not in st.session_state: st.session_state.nlp_processed = False
if 'nlp_quotes' not in st.session_state: st.session_state.nlp_quotes = {}

# --- BLOQUE 1: INPUT MANUAL OBLIGATORIO ---
st.sidebar.subheader("1. Core Parameters (Manual Entry)")
cie10_input = st.sidebar.text_input("Reason for admission (ICD-10 Code):", value="I10", help="Example: I10, E11, J44")
dias_internados = st.sidebar.number_input("Number of days hospitalized:", min_value=1, max_value=150, value=5)
rango_edad_ui = st.sidebar.selectbox("Patient Age Range:", list(opciones_edad_dict.keys()))
rango_edad = opciones_edad_dict[rango_edad_ui].upper()
es_pluripatologico = st.sidebar.checkbox("Is the patient Pluripathological?", value=False)

st.sidebar.markdown("---")

# --- BLOQUE 2: MOTOR NLP (AUTOMATIZACIÓN) ---
st.sidebar.subheader("2. Narrative Phenotype (NLP)")

# 🌟 FIX DE SEGURIDAD: Leer API Key de st.secrets de forma segura
api_key_default = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key_default = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key_default = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass # Si st.secrets no está configurado, pasamos silenciosamente al ingreso manual

api_key = st.sidebar.text_input(
    "Gemini API Key (Live Triage):", 
    value=api_key_default,
    type="password", 
    help="Automatically loaded from Streamlit Secrets. Paste here if not configured."
)

ing_text = st.sidebar.text_area("Admission Notes:", height=100)
evo_text = st.sidebar.text_area("Evolution Notes:", height=100)

if st.sidebar.button("🧠 Run NLP Extraction", use_container_width=True):
    if not api_key:
        st.sidebar.error("API Key required. Please configure st.secrets or paste it above.")
    elif not ing_text and not evo_text:
        st.sidebar.warning("Please provide clinical notes.")
    else:
        with st.spinner("Forensic extraction in progress..."):
            try:
                # El resto de tu código NLP se mantiene exactamente igual a partir de aquí
                genai.configure(api_key=api_key)
                modelo_nlp = genai.GenerativeModel(
                    'models/gemini-2.5-flash', 
                    generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.0)
                )

                
                prompt_sistema = """Eres un auditor médico forense estricto. Extrae variables de riesgo.
REGLAS:
1. CERO INFERENCIA: Si no se menciona, valor es false/null, cita "".
2. NEGACIONES: "niega", "sin" = false.
3. SEPARACIÓN TEMPORAL: ING_ (Estado al Ingreso), EVO_ (Curso de Internación).
4. CITA EXACTA: Máximo 15 palabras.

Devuelve JSON. Claves: LLM_tabaquismo_activo, LLM_alcoholismo, LLM_drogas_ilicitas, LLM_fragilidad_geriatrica, LLM_polifarmacia, LLM_desnutricion_severa, LLM_oxigenodependiente, LLM_historial_caidas, LLM_abandono_medicacion, LLM_AF_diabetes, LLM_AF_hipertension, LLM_AF_cardiovascular_otro, LLM_AF_oncologico, LLM_AF_metabolico_otro, LLM_AF_neurologico, LLM_AF_psiquiatrico, LLM_AF_respiratorio, LLM_AF_renal, LLM_AF_autoinmune, ING_dolor_eva (0-10), ING_gravedad_percibida (1-10), ING_alteracion_mental, ING_dependencia_funcional, ING_portador_dispositivos, ING_consultas_reiteradas, ING_riesgo_hemorragico, EVO_dolor_eva (0-10), EVO_gravedad_percibida (1-10), EVO_alteracion_mental, EVO_dependencia_funcional, EVO_portador_dispositivos, EVO_complicacion_internacion, EVO_fuga_o_alta_irregular, EVO_cuidados_paliativos, EVO_ulceras_presion, EVO_aislamiento_infeccioso.
FORMATO: {"LLM_tabaquismo_activo": {"valor": true, "cita": "fuma 10 cigarrillos al día"}, ...}"""
                
                bloque_clinico = f"\n\n--- ESTADO AL INGRESO ---\n{ing_text}\n\n--- CURSO DE INTERNACIÓN ---\n{evo_text}"
                respuesta = modelo_nlp.generate_content(prompt_sistema + bloque_clinico)
                
                match = re.search(r'\{.*\}', respuesta.text, re.DOTALL)
                if match:
                    json_extraido = json_repair.loads(match.group(0))
                    
                    af_activos, cro_activos, ing_activos, evo_activos = [], [], [], []
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
                                base_k = k.replace('LLM_AF_', '').replace('LLM_', '').replace('ING_', '').replace('EVO_', '')
                                if "LLM_AF_" in k: af_activos.append(base_k)
                                elif "LLM_" in k: cro_activos.append(base_k)
                                elif "ING_" in k: ing_activos.append(base_k)
                                elif "EVO_" in k: evo_activos.append(base_k)
                                
                            if val and cita:
                                quotes[k] = cita
                                
                    st.session_state.ui_af_sel = [k for k, v in af_dict.items() if v in af_activos]
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
        
        # 1. Invertimos los diccionarios locales para traducir de Español a Inglés
        inv_af = {v: k for k, v in af_dict.items()}
        inv_cro = {v: k for k, v in cro_dict.items()}
        inv_ing = {v: k for k, v in ing_dict.items()}
        inv_evo = {v: k for k, v in evo_dict.items()}
        
        for var, quote in st.session_state.nlp_quotes.items():
            # 2. Limpiamos los prefijos técnicos
            var_clean = var.replace("LLM_AF_", "").replace("LLM_", "").replace("ING_", "").replace("EVO_", "")
            
            # 3. Asignamos el contexto clínico y traducimos usando los diccionarios
            if var.startswith("LLM_AF_"):
                nombre_base = inv_af.get(var_clean, var_clean.replace('_', ' ').title())
                var_traducida = f"Family History: {nombre_base}"
                
            elif var.startswith("LLM_"):
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
            
            # 4. Renderizamos en la interfaz
            st.markdown(f"**{var_traducida}:**\n> *\"{quote}\"*")

st.sidebar.markdown("---")

# --- BLOQUE 3: REVISIÓN HUMANA (AUDITORÍA) ---
st.sidebar.subheader("3. Patient Background (Review)")
af_seleccionados_ui = st.sidebar.multiselect("Family History (LLM_AF):", list(af_dict.keys()), key="ui_af_sel")
af_seleccionados = [af_dict[k] for k in af_seleccionados_ui]

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
for col in columnas_modelo:
    if col in ['rango_edad', 'CIE10_MACRO', 'CIE10_SUBMACRO']:
        paciente_data[col] = "DESCONOCIDO"
    else:
        paciente_data[col] = 0.0

if 'rango_edad' in paciente_data: paciente_data['rango_edad'] = rango_edad
if 'dias_internados' in paciente_data: paciente_data['dias_internados'] = float(dias_internados)
if 'pluripatologico' in paciente_data: paciente_data['pluripatologico'] = 1.0 if es_pluripatologico else 0.0

codigo_normalizado = normalizar_cie10(cie10_input)
categoria_cie10 = mapear_cie10_macro(codigo_normalizado)
if 'CIE10_MACRO' in paciente_data: paciente_data['CIE10_MACRO'] = categoria_cie10

for af in af_seleccionados: 
    if f"LLM_AF_{af}" in paciente_data: paciente_data[f"LLM_AF_{af}"] = 1.0

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

df_paciente = pd.DataFrame([paciente_data])[columnas_modelo]

# ==========================================
# 5. INFERENCE & MAIN DASHBOARD (COCKPIT LAYOUT)
# ==========================================
riesgo = pipeline.predict_proba(df_paciente)[0][1]

col_izq, col_der = st.columns([1, 3.5])

with col_izq:
    st.subheader("Readmission Risk")
    st.metric(label="15-Day Probability", value=f"{riesgo*100:.1f}%")
    
    if riesgo > umbral:
        # Quitamos los \n\n para hacer la caja más delgada
        st.error(f"⚠️ **CLINICAL ALERT** - Risk exceeds safety threshold ({umbral*100:.1f}%).")
    else:
        st.success(f"✅ **SAFE DISCHARGE** - Risk controlled within permitted threshold.")

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
    
    st.info(f"**Mapped Diagnosis:** {categoria_cie10_ingles} (Code: {codigo_normalizado})")


with col_der:
    st.subheader("Decision Audit & Clinical Context")
    
    st.markdown("#### 🔍 Prescriptive Explanability (SHAP)")
    
    # Checkbox para alternar metodologías
    filtrar_activos = st.checkbox("🎯 Show only the impact of present conditions (Hide protective factors due to absence)", value=False)
    
    try:
        clf = pipeline.named_steps['clasificador']
        prep = pipeline.named_steps['preprocesador']
        
        X_proc = prep.transform(df_paciente)
        X_proc_dense = X_proc.toarray() if hasattr(X_proc, 'toarray') else np.array(X_proc)
        X_paciente_1d = X_proc_dense[0] 
        
        nombres_crudos = prep.get_feature_names_out()
        
        shap_ui_dict = {
            'dias_internados': 'Hospitalization Days', 'pluripatologico': 'Pluripathological',
            'ING_dolor_eva': 'Initial Pain', 'ING_gravedad_percibida': 'Initial Severity',
            'EVO_dolor_eva': 'Current Pain', 'EVO_gravedad_percibida': 'Current Severity',
            'DELTA_dolor_eva': 'Pain Delta', 'DELTA_gravedad_percibida': 'Severity Delta',
            'DELTA_alteracion_mental': 'Mental Alt. Delta', 'DELTA_dependencia_funcional': 'Func. Dep. Delta',
            'DELTA_portador_dispositivos': 'Device Bearer Delta', 'ING_alteracion_mental': 'Initial Mental Alt.',
            'ING_consultas_reiteradas': 'Initial Repeated Consults', 'ING_dependencia_funcional': 'Initial Func. Dep.',
            'ING_portador_dispositivos': 'Initial Device Bearer', 'ING_riesgo_hemorragico': 'Initial Hemorrhagic Risk',
            'EVO_aislamiento_infeccioso': 'Current Infect. Isolation', 'EVO_alteracion_mental': 'Current Mental Alt.',
            'EVO_complicacion_internacion': 'Current Hosp. Complication', 'EVO_cuidados_paliativos': 'Current Palliat. Care',
            'EVO_dependencia_funcional': 'Current Func. Dep.', 'EVO_fuga_o_alta_irregular': 'Current Irreg. Discharge',
            'EVO_portador_dispositivos': 'Current Device Bearer', 'EVO_ulceras_presion': 'Current Pressure Ulcers',
            'LLM_AF_autoinmune': 'Fam. Hist: Autoimmune', 'LLM_AF_cardiovascular_otro': 'Fam. Hist: Other CV',
            'LLM_AF_diabetes': 'Fam. Hist: Diabetes', 'LLM_AF_hipertension': 'Fam. Hist: Hypertension',
            'LLM_AF_metabolico_otro': 'Fam. Hist: Other Metabolic', 'LLM_AF_neurologico': 'Fam. Hist: Neurological',
            'LLM_AF_oncologico': 'Fam. Hist: Oncological', 'LLM_AF_psiquiatrico': 'Fam. Hist: Psychiatric',
            'LLM_AF_renal': 'Fam. Hist: Renal', 'LLM_AF_respiratorio': 'Fam. Hist: Respiratory',
            'LLM_abandono_medicacion': 'Chronic: Med. Abandonment', 'LLM_alcoholismo': 'Chronic: Alcoholism',
            'LLM_desnutricion_severa': 'Chronic: Severe Malnutrition', 'LLM_drogas_ilicitas': 'Chronic: Illicit Drugs',
            'LLM_fragilidad_geriatrica': 'Chronic: Geriatric Frailty', 'LLM_historial_caidas': 'Chronic: History of Falls',
            'LLM_oxigenodependiente': 'Chronic: Oxygen Dependent', 'LLM_polifarmacia': 'Chronic: Polypharmacy',
            'LLM_tabaquismo_activo': 'Chronic: Active Smoking'
        }

        # --- LÓGICA DE TRADUCCIÓN ROBUSTA ---
        nombres_limpios_traducidos = []
        cie10_upper = {k.upper(): v for k, v in cie10_ui_dict.items()}
        
        for nombre_crudo in nombres_crudos:
            traducido = nombre_crudo.split('__')[-1]
            
            for sufijo in ['_1.0', '_1', '_True', '_true', '_0.0', '_0', '_False', '_false']:
                if traducido.endswith(sufijo):
                    traducido = traducido[:-len(sufijo)]
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

        # --- EXTRACCIÓN SHAP ---
        try:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_proc_dense, check_additivity=False)
        except Exception:
            explainer = shap.LinearExplainer(clf, X_proc_dense) if hasattr(clf, 'coef_') else shap.Explainer(clf, X_proc_dense)
            shap_vals = explainer(X_proc_dense).values
        
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        if len(shap_vals.shape) > 2: shap_vals = shap_vals[:, :, 1]
        
        exp_val = explainer.expected_value
        exp_val = exp_val[1] if isinstance(exp_val, (list, np.ndarray)) and len(exp_val) > 1 else exp_val[0] if isinstance(exp_val, (list, np.ndarray)) else exp_val
        
        shap_vals_pct = shap_vals[0] * 100
        exp_val_pct = exp_val * 100

        # --- BIFURCACIÓN METODOLÓGICA ---
        if not filtrar_activos:
            # 1. WATERFALL PURO: Integridad matemática total
            explicacion_completa = shap.Explanation(
                values=np.array(shap_vals_pct), 
                base_values=exp_val_pct, 
                data=np.array(X_paciente_1d), 
                feature_names=nombres_limpios_traducidos
            )
            
            fig_shap, ax_shap = plt.subplots(figsize=(12, 5)) 
            shap.waterfall_plot(explicacion_completa, show=False, max_display=10) 
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
            
        else:
            # 2. AISLAMIENTO CLÍNICO: Gráfico de barras horizontal nativo
            indices_activos = []
            variables_continuas = ['Days', 'Pain', 'Severity', 'Delta', 'Consultations', 'Complexity']

            for i, (val_proc, nombre_traducido) in enumerate(zip(X_paciente_1d, nombres_limpios_traducidos)):
                nombre_crudo = nombres_crudos[i]
                nombre_base = nombre_crudo.split('__')[-1]
                
                for sufijo in ['_1.0', '_1', '_True', '_true', '_0.0', '_0', '_False', '_false']:
                    if nombre_base.endswith(sufijo):
                        nombre_base = nombre_base[:-len(sufijo)]
                        break
                
                val_real = val_proc 
                
                if nombre_base in df_paciente.columns:
                    val_real = df_paciente[nombre_base].iloc[0]
                else:
                    columnas_categoricas = ['CIE10_MACRO', 'CIE10_SUBMACRO', 'rango_edad']
                    for col_cat in columnas_categoricas:
                        if nombre_base.startswith(col_cat + '_'):
                            valor_categoria_columna = nombre_base.replace(col_cat + '_', '')
                            valor_real_paciente = str(df_paciente[col_cat].iloc[0])
                            
                            if valor_categoria_columna.strip().upper() == valor_real_paciente.strip().upper():
                                val_real = 1.0 
                            else:
                                val_real = 0.0 
                            break
                
                es_continua = any(kw in nombre_traducido for kw in variables_continuas)
                es_inactivo = False
                
                if not es_continua:
                    val_str = str(val_real).strip().upper()
                    if val_str in ['0', '0.0', 'FALSE', 'NONE', 'N/A', 'NAN', '']:
                        es_inactivo = True
                
                if abs(shap_vals_pct[i]) < 0.01:
                    es_inactivo = True

                if not es_inactivo:
                    indices_activos.append(i)

            if not indices_activos:
                st.info("No significant active clinical factors to isolate.")
            else:
                activos_vals = [shap_vals_pct[i] for i in indices_activos]
                activos_nombres = [nombres_limpios_traducidos[i] for i in indices_activos]
                
                # Ordenar para visualización de barras horizontales
                datos_ordenados = sorted(zip(activos_vals, activos_nombres), key=lambda x: abs(x[0]))
                y_vals = [x[0] for x in datos_ordenados]
                y_names = [x[1] for x in datos_ordenados]
                
                # Altura dinámica según la cantidad de variables
                fig_bar, ax_bar = plt.subplots(figsize=(10, max(4, len(y_names) * 0.4)))
                colores = ['#FF0051' if v > 0 else '#008BFB' for v in y_vals]
                
                ax_bar.barh(y_names, y_vals, color=colores)
                ax_bar.set_xlabel("Impact on Readmission Risk (%)")
                ax_bar.set_title("Isolation of Present Clinical Factors")
                
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                ax_bar.axvline(0, color='black', linewidth=1)
                
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

    except Exception as e:
        st.error("SHAP computation failed.")
        st.warning(str(e))

    st.markdown("---")

    col_traj, col_scat = st.columns(2)
    
    with col_traj:
        st.markdown("#### 📉 Dynamic Trajectory")
        try:
            df_row = df_paciente.iloc[[0]].copy()
            pares_clinicos = {
                'Pain (VAS)': ('ING_dolor_eva', 'EVO_dolor_eva'),
                'Severity': ('ING_gravedad_percibida', 'EVO_gravedad_percibida'),
                'Mental Alt.': ('ING_alteracion_mental', 'EVO_alteracion_mental'),
                'Func. Dep.': ('ING_dependencia_funcional', 'EVO_dependencia_funcional'),
                'Devices': ('ING_portador_dispositivos', 'EVO_portador_dispositivos')
            }
            datos = []
            for label, (col_ing, col_evo) in pares_clinicos.items():
                val_ing = float(df_row[col_ing].values[0]) if col_ing in df_row.columns else 0.0
                val_evo = float(df_row[col_evo].values[0]) if col_evo in df_row.columns else 0.0
                datos.append({'label': label, 'ing': val_ing, 'evo': val_evo})

            fig_slope, ax_slope = plt.subplots(figsize=(6, 5.5))
            ax_slope.set_xlim(-0.5, 1.5)
            ax_slope.set_xticks([0, 1])
            ax_slope.set_xticklabels(['Admission', 'Current'], fontsize=10, fontweight='bold')
            
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

            y_ing_coords = [d['ing'] for d in datos]
            y_evo_coords = [d['evo'] for d in datos]
            textos_ing_y = separar_superposiciones(y_ing_coords)
            textos_evo_y = separar_superposiciones(y_evo_coords)
            min_y, max_y = 0, 0

            for i, d in enumerate(datos):
                val_ing, val_evo, label = d['ing'], d['evo'], d['label']
                min_y, max_y = min(min_y, val_ing, val_evo), max(max_y, val_ing, val_evo)
                color_linea = '#00C851' if val_evo <= val_ing else '#FF4444'
                
                ax_slope.plot([0, 1], [val_ing, val_evo], color=color_linea, linewidth=2.5, marker='o', markersize=6, zorder=3)
                
                ax_slope.text(-0.06, textos_ing_y[i], f"{label} ({val_ing:.1f})", ha='right', va='center', fontsize=8, fontweight='bold', color='#444444')
                ax_slope.text(1.06, textos_evo_y[i], f"({val_evo:.1f}) {label}", ha='left', va='center', fontsize=8, fontweight='bold', color=color_linea)

            ax_slope.set_ylim(min_y - 1.5, max_y + 1.5)
            ax_slope.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            ax_slope.get_yaxis().set_visible(False)
            ax_slope.axvline(x=0, color='#E5E5E5', linestyle='--', linewidth=1.2, zorder=1)
            ax_slope.axvline(x=1, color='#E5E5E5', linestyle='--', linewidth=1.2, zorder=1)

            fig_slope.tight_layout()
            st.pyplot(fig_slope)
            plt.close(fig_slope)
        except Exception as e:
            st.error("Trajectory unavailable.")


# ==========================================
# 6. THERAPEUTIC NAVIGATOR (DiCE - DUAL COORDINATED XAI)
# ==========================================
st.markdown("---")
st.subheader("Therapeutic Navigator (Prescriptive AI)")

class ModeloSincronizado:
    def __init__(self, pipeline_original, columnas_modelo):
        self.pipeline = pipeline_original
        self.columnas_modelo = columnas_modelo
        
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X_sync = pd.DataFrame(X, columns=self.columnas_modelo)
        else:
            X_sync = X.copy()
        
        def sync_delta(df, col_delta, col_evo, col_ing):
            if col_delta in df.columns and col_evo in df.columns and col_ing in df.columns:
                df[col_delta] = df[col_evo] - df[col_ing]
                
        sync_delta(X_sync, 'DELTA_dolor_eva', 'EVO_dolor_eva', 'ING_dolor_eva')
        sync_delta(X_sync, 'DELTA_gravedad_percibida', 'EVO_gravedad_percibida', 'ING_gravedad_percibida')
        sync_delta(X_sync, 'DELTA_alteracion_mental', 'EVO_alteracion_mental', 'ING_alteracion_mental')
        sync_delta(X_sync, 'DELTA_dependencia_funcional', 'EVO_dependencia_funcional', 'ING_dependencia_funcional')
        sync_delta(X_sync, 'DELTA_portador_dispositivos', 'EVO_portador_dispositivos', 'ING_portador_dispositivos')
        
        return self.pipeline.predict_proba(X_sync)

if riesgo <= umbral:
    st.info("The patient is in optimal condition for discharge. No stabilization targets required.")
else:
    st.warning("High risk detected. Automatically calculating clinical stabilization targets to reach the safety threshold...")
    
    # --- EJECUCIÓN AUTOMÁTICA (SIN BOTÓN) ---
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
            
            features_continuas = df_paciente.select_dtypes(include=[np.number]).columns.tolist()
            
            d = dice_ml.Data(
                dataframe=df_dice_train, 
                continuous_features=features_continuas, 
                outcome_name='target'
            )
            
            modelo_sincronizado = ModeloSincronizado(pipeline, columnas_modelo)
            m = dice_ml.Model(model=modelo_sincronizado, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            
            variables_accionables = [col for col in columnas_modelo if col.startswith('EVO_')]
            rangos_permitidos = {}
            vars_a_variar = []
            
            for col in variables_accionables:
                val_actual = df_paciente[col].iloc[0]
                if 'cuidados_paliativos' in col or 'fuga' in col: 
                    continue
                
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

            # Permitimos a DiCE variar los días de internación
            if 'dias_internados' in df_paciente.columns:
                val_dias = float(df_paciente['dias_internados'].iloc[0])
                rangos_permitidos['dias_internados'] = [val_dias, val_dias + 7.0]
                vars_a_variar.append('dias_internados')
            
            if not vars_a_variar:
                st.error("There are no modifiable clinical targets in the patient's current evolution that can improve their condition.")
            else:
                dice_exp = exp.generate_counterfactuals(
                    df_paciente, total_CFs=5, desired_class="opposite", 
                    features_to_vary=vars_a_variar, permitted_range=rangos_permitidos, random_seed=42
                )
                
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                if cf_df is not None and not cf_df.empty:
                    
                    # Filtro de redondeo
                    for col in vars_a_variar:
                        if col not in ['EVO_dolor_eva', 'EVO_gravedad_percibida', 'dias_internados']:
                            cf_df[col] = (cf_df[col] >= 0.5).astype(int)
                        else:
                            cf_df[col] = cf_df[col].round()
                            
                    # Sincronización causal del dataframe simulado
                    cf_df['DELTA_dolor_eva'] = cf_df['EVO_dolor_eva'] - df_paciente.iloc[0]['ING_dolor_eva']
                    cf_df['DELTA_gravedad_percibida'] = cf_df['EVO_gravedad_percibida'] - df_paciente.iloc[0]['ING_gravedad_percibida']
                    cf_df['DELTA_alteracion_mental'] = cf_df['EVO_alteracion_mental'] - df_paciente.iloc[0]['ING_alteracion_mental']
                    cf_df['DELTA_dependencia_funcional'] = cf_df['EVO_dependencia_funcional'] - df_paciente.iloc[0]['ING_dependencia_funcional']
                    cf_df['DELTA_portador_dispositivos'] = cf_df['EVO_portador_dispositivos'] - df_paciente.iloc[0]['ING_portador_dispositivos']
                            
                    cf_df = cf_df.drop_duplicates(subset=vars_a_variar).reset_index(drop=True)

                    st.success(f"✅ **{len(cf_df)} UNIQUE CLINICAL STABILIZATION TARGETS FOUND:**")
                    st.markdown("The medical staff can select the most feasible goal according to the ward capabilities:")
                    
                    evo_output_dict = {
                        'EVO_dolor_eva': 'Current Pain', 'EVO_gravedad_percibida': 'Current Severity',
                        'EVO_aislamiento_infeccioso': 'Infectious Isolation', 'EVO_alteracion_mental': 'Mental Alteration',
                        'EVO_complicacion_internacion': 'Hospitalization Complication', 'EVO_cuidados_paliativos': 'Palliative Care',
                        'EVO_dependencia_funcional': 'Functional Dependency', 'EVO_fuga_o_alta_irregular': 'Irregular Discharge / Escape',
                        'EVO_portador_dispositivos': 'Device Bearer', 'EVO_ulceras_presion': 'Pressure Ulcers',
                        'dias_internados': 'Additional Hospitalization Days'
                    }
                    
                    for r_idx in range(len(cf_df)):
                        with st.expander(f"➔ 🛤️ Alternative Target Route {r_idx + 1}", expanded=(r_idx == 0)):
                            cambios_detectados = 0
                            st.markdown("#### 🎯 Prescriptive Actions:")
                            
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
                                        
                            if cambios_detectados == 0:
                                st.write("This alternative suggests maintaining current parameters based on marginal risk stability.")
                            else:
                                # --- MULTIDIMENSIONAL RADAR CHART ---
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
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=val_meta_cerrados, theta=cat_cerradas,
                                    fill='toself', fillcolor='rgba(44, 160, 44, 0.25)', 
                                    line=dict(color='#2CA02C', width=2.5), name='DiCE Target'
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
                                
                                st.plotly_chart(fig_radar, use_container_width=True)
                                
                else:
                    st.error("No mathematically viable target routes were found.")
                    
        except Exception as e:
            st.error("Counterfactual engine is currently unavailable.")
            st.warning(f"Technical Context: {str(e)}")

# ==========================================
# 7. INTERACTIVE CLINICAL SANDBOX (LIME WHAT-IF)
# ==========================================
st.markdown("---")
st.subheader("🧪 Clinical Hypothesis Simulator (SandBox)")
st.markdown("Modify stay duration or toggle evolution complications to observe how the local risk boundaries and interaction weights (LIME) react to different clinical outcomes. **Admission parameters are locked to preserve temporal causality.**")

# --- HYPOTHESIS CONTROL PANEL ---
with st.expander("🛠️ Configure Stabilization Scenario", expanded=True):
    # Control de Tiempo
    dias_base = int(df_paciente['dias_internados'].iloc[0] if 'dias_internados' in df_paciente.columns else 1)
    dias_sim = st.slider("Hospitalization Stay (Days):", min_value=1, max_value=60, value=dias_base, step=1)
    
    st.markdown("---")
    st.markdown("**Evolution Status (EVO)** - *Toggle acquired complications or resolved states*")
    
    # Mapa completo de complicaciones de evolución
    sim_evo_map = {
        'Mental Alteration': 'EVO_alteracion_mental',
        'Functional Dependency': 'EVO_dependencia_funcional',
        'Medical Devices': 'EVO_portador_dispositivos',
        'Infectious Isolation': 'EVO_aislamiento_infeccioso',
        'Hosp. Complication': 'EVO_complicacion_internacion',
        'Pressure Ulcers': 'EVO_ulceras_presion',
        'Palliative Care': 'EVO_cuidados_paliativos',
        'Irregular Discharge': 'EVO_fuga_o_alta_irregular'
    }
    
    # Cuadrícula de 4 columnas para organizar los 8 toggles limpiamente
    cols_evo = st.columns(4)
    status_evo_sim = {}
    
    for i, (label, col) in enumerate(sim_evo_map.items()):
        # divmod reparte los toggles equitativamente entre las 4 columnas
        with cols_evo[i % 4]:
            val_init = bool(df_paciente[col].iloc[0] if col in df_paciente.columns else 0)
            status_evo_sim[col] = st.toggle(label, value=val_init, key=f"sim_{col}")

# --- MOTOR DE CÁLCULO DINÁMICO ---
try:
    with st.spinner("Analyzing combinatorial impacts of simulated trajectory..."):
        # 1. Crear el clon simulado del paciente (Pasado Inmutable)
        df_sim = df_paciente.copy()
        
        # Inyectar las simulaciones de futuro/presente
        df_sim['dias_internados'] = float(dias_sim)
        for col, val in status_evo_sim.items(): 
            df_sim[col] = 1.0 if val else 0.0
        
        # 2. SINCRONIZACIÓN CAUSAL DE DELTAS
        # (Solo recalculamos los deltas de las variables que tienen equivalente al ingreso)
        pares_delta = {
            'DELTA_alteracion_mental': ('EVO_alteracion_mental', 'ING_alteracion_mental'),
            'DELTA_dependencia_funcional': ('EVO_dependencia_funcional', 'ING_dependencia_funcional'),
            'DELTA_portador_dispositivos': ('EVO_portador_dispositivos', 'ING_portador_dispositivos')
        }
        for col_delta, (col_evo, col_ing) in pares_delta.items():
            if col_evo in df_sim.columns and col_ing in df_sim.columns:
                df_sim[col_delta] = df_sim[col_evo] - df_sim[col_ing]

        # 3. Preprocesamiento Seguro
        prep = pipeline.named_steps['preprocesador']
        clf = pipeline.named_steps['clasificador']
        
        X_p_proc = prep.transform(df_sim)
        X_p_dense = X_p_proc.toarray()[0] if hasattr(X_p_proc, 'toarray') else np.array(X_p_proc)[0]
        
        X_t_proc = prep.transform(df_train_sample)
        X_t_dense = X_t_proc.toarray() if hasattr(X_t_proc, 'toarray') else np.array(X_t_proc)
        
        # 4. Mapeo de Nombres (Traducido al inglés para la UI)
        nombres_crudos = prep.get_feature_names_out()
        
        # Diccionario para unificar nomenclaturas en LIME
        ui_dict = {
            'DELTA_dolor_eva': 'Δ Pain', 'DELTA_gravedad_percibida': 'Δ Severity',
            'DELTA_alteracion_mental': 'Δ Mental Alt.', 'DELTA_dependencia_funcional': 'Δ Func. Dep.',
            'DELTA_portador_dispositivos': 'Δ Medical Devices',
            'EVO_aislamiento_infeccioso': 'Current Infect. Isolation',
            'EVO_complicacion_internacion': 'Current Hosp. Complication',
            'EVO_cuidados_paliativos': 'Current Palliat. Care',
            'EVO_fuga_o_alta_irregular': 'Current Irreg. Discharge',
            'EVO_ulceras_presion': 'Current Pressure Ulcers',
            # Mapeamos las evoluciones que SÍ tienen delta por si el modelo las usa aisladas
            'EVO_alteracion_mental': 'Current Mental Alt.',
            'EVO_dependencia_funcional': 'Current Func. Dep.',
            'EVO_portador_dispositivos': 'Current Device Bearer'
        }
        
        nombres_lime = [ui_dict.get(n.split('__')[-1].split('_1')[0], n.split('__')[-1]) for n in nombres_crudos]

        # 5. Inicialización de LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_t_dense,
            feature_names=nombres_lime, 
            mode='classification', 
            random_state=42
        )

        exp = explainer.explain_instance(
            data_row=X_p_dense, 
            predict_fn=clf.predict_proba, 
            num_features=len(nombres_lime) 
        )
        
        # 6. Filtrado Ampliado: Mostramos Deltas Y Complicaciones Adquiridas
        lime_list = [item for item in exp.as_list() if 'Δ' in item[0] or 'Current' in item[0]]
        
        if not lime_list:
             st.info("No significant clinical interaction impacts found for this specific configuration.")
        else:
            # Eliminar ceros matemáticos y ordenar por peso
            lime_list = [x for x in lime_list if abs(x[1]) > 0.001]
            lime_list = sorted(lime_list, key=lambda x: abs(x[1]))
            
            fig_l = go.Figure(go.Bar(
                x=[x[1]*100 for x in lime_list], 
                y=[x[0] for x in lime_list],
                orientation='h', 
                marker_color=['#D62728' if x[1]>0 else '#2CA02C' for x in lime_list],
                text=[f"+{x[1]*100:.1f}%" if x[1]>0 else f"{x[1]*100:.1f}%" for x in lime_list],
                textposition='outside', 
                textfont=dict(size=12)
            ))
            
            fig_l.update_layout(
                title=f"Scenario Impact Analysis (Simulated Trajectory)",
                xaxis_title="Impact on Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=40, t=40, b=10),
                height=max(300, len(lime_list) * 40), # Ajuste dinámico de altura
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(128,128,128,0.2)', 
                    zeroline=True, zerolinecolor='rgba(128,128,128,0.6)'
                )
            )
            
            col_l1, col_l2 = st.columns([1, 3])
            with col_l2: 
                st.plotly_chart(fig_l, use_container_width=True)
            with col_l1:
                st.info("**Simulation Insight**")
                st.caption("This graph shows how resolving or acquiring specific complications dynamically changes the model's 'reasoning' for *this specific patient*.")
                
                if dias_sim > dias_base:
                    st.warning("⚠️ **Future Projection:** Simulating extended stay. Observe if protective factors degrade over time.")
                elif dias_sim < dias_base:
                    st.caption("⏪ **Retrospective View:** Simulating premature discharge conditions.")

except Exception as e:
    st.error("Simulation engine failed to initialize.")
    st.warning(str(e))


# ==========================================
# 8. CLINICAL SIMILARITY NETWORK (ARCHEGO ADVANCED UI)
# ==========================================
TRANSLATION_DICT = {
    'dias_internados': 'Length of Stay (Days)', 'rango_edad': 'Age Range', 'pluripatologico': 'Multimorbidity',
    'CIE10_MACRO': 'Primary Diagnosis (ICD-10)', 'LLM_tabaquismo_activo': 'Active Smoking', 'LLM_alcoholismo': 'Alcoholism',
    'LLM_drogas_ilicitas': 'Illicit Drug Use', 'LLM_fragilidad_geriatrica': 'Geriatric Frailty',
    'LLM_polifarmacia': 'Polypharmacy', 'LLM_desnutricion_severa': 'Severe Malnutrition', 'LLM_oxigenodependiente': 'Oxygen Dependent',
    'LLM_historial_caidas': 'History of Falls', 'LLM_abandono_medicacion': 'Medication Non-adherence',
    'LLM_AF_diabetes': 'FHx: Diabetes', 'LLM_AF_hipertension': 'FHx: Hypertension', 'LLM_AF_cardiovascular_otro': 'FHx: Other Cardiovascular',
    'LLM_AF_oncologico': 'FHx: Oncology', 'LLM_AF_metabolico_otro': 'FHx: Other Metabolic', 'LLM_AF_neurologico': 'FHx: Neurology',
    'LLM_AF_psiquiatrico': 'FHx: Psychiatry', 'LLM_AF_respiratorio': 'FHx: Respiratory', 'LLM_AF_renal': 'FHx: Renal',
    'LLM_AF_autoinmune': 'FHx: Autoimmune', 'ING_dolor_eva': 'Admission: Pain (VAS)', 'ING_gravedad_percibida': 'Admission: Perceived Severity',
    'ING_alteracion_mental': 'Admission: Altered Mental Status', 'ING_dependencia_funcional': 'Admission: Functional Dependence',
    'ING_portador_dispositivos': 'Admission: Medical Devices', 'ING_consultas_reiteradas': 'Admission: Repeated Consultations',
    'ING_riesgo_hemorragico': 'Admission: Hemorrhagic Risk', 'EVO_dolor_eva': 'Evolution: Pain (VAS)',
    'EVO_gravedad_percibida': 'Evolution: Perceived Severity', 'EVO_alteracion_mental': 'Evolution: Altered Mental Status',
    'EVO_dependencia_funcional': 'Evolution: Functional Dependence', 'EVO_portador_dispositivos': 'Evolution: Medical Devices',
    'EVO_complicacion_internacion': 'Evolution: Hospital Complication', 'EVO_fuga_o_alta_irregular': 'Evolution: Irregular Discharge (AMA)',
    'EVO_cuidados_paliativos': 'Evolution: Palliative Care', 'EVO_ulceras_presion': 'Evolution: Pressure Ulcers',
    'EVO_aislamiento_infeccioso': 'Evolution: Infectious Isolation', 'DELTA_dolor_eva': 'Δ Pain (VAS)',
    'DELTA_gravedad_percibida': 'Δ Perceived Severity', 'DELTA_alteracion_mental': 'Δ Altered Mental Status',
    'DELTA_dependencia_funcional': 'Δ Functional Dependence', 'DELTA_portador_dispositivos': 'Δ Medical Devices'
}

def format_clinical_value(key_es, value):
    val_str = str(value).strip().upper()
    if key_es == 'rango_edad':
        traducciones_edad = {'ADULTO DE MEDIANA EDAD': 'Middle-Aged Adult', 'ADULTO MAYOR': 'Senior Adult', 'ADULTO JOVEN': 'Young Adult', 'ANCIANO': 'Elderly'}
        return traducciones_edad.get(val_str, value)
    if key_es == 'CIE10_MACRO':
        traducciones_cie = {'CARDIOPATÍA ISQUÉMICA': 'Ischemic Heart Disease', 'HIPERTENSIÓN': 'Hypertension', 'DIABETES': 'Diabetes', 'ENFERMEDAD CARDIOPULMONAR': 'Cardiopulmonary Disease'}
        return traducciones_cie.get(val_str, value)
    
    bool_suffixes = ('_mental', '_funcional', '_dispositivos', '_reiteradas', '_hemorragico', '_internacion', '_irregular', '_paliativos', '_presion', '_infeccioso')
    if key_es.startswith('LLM_') or key_es == 'pluripatologico' or (key_es.endswith(bool_suffixes) and not key_es.startswith('DELTA_')):
        try:
            return "Yes" if float(value) == 1.0 else "No"
        except ValueError:
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

def renderizar_notas_gemelo(texto_evolucion, citas_llm, lista_enfermedades):
    if not isinstance(texto_evolucion, str) or not texto_evolucion:
        return "No narrative context available."
        
    texto_resaltado = texto_evolucion
    
    # --- MODIFICADO: Marcadores adaptados para modo oscuro ---
    if isinstance(citas_llm, dict):
        for cita, variable in citas_llm.items():
            if isinstance(cita, str) and cita.strip():
                cita_escapada = re.escape(cita)
                # FIX: Se fuerza 'color: #000000;' dentro del mark para que la letra no se ponga blanca con el fondo amarillo
                marcador = f"<mark style='background-color: #FFF2CC; color: #000000; border-radius: 3px; padding: 2px 4px;'><b>{cita}</b> <span style='font-size: 0.7em; background-color: #FFD966; padding: 2px 5px; border-radius: 8px; color: #594000; margin-left: 4px; display: inline-block; vertical-align: middle; line-height: 1;'>{variable}</span></mark>"
                texto_resaltado = re.sub(cita_escapada, marcador, texto_resaltado, flags=re.IGNORECASE)
                
    if isinstance(lista_enfermedades, list):
        for enfermedad in lista_enfermedades:
            if isinstance(enfermedad, str) and enfermedad.strip():
                patron = rf"\b({re.escape(enfermedad)})\b"
                # FIX: Fuerza letra negra en la marca de enfermedad
                marcador = r"<mark style='background-color: #FFCCCC; color: #000000; border-radius: 3px; padding: 0px 2px;'>\1</mark>"
                texto_resaltado = re.sub(patron, marcador, texto_resaltado, flags=re.IGNORECASE)
                
    # FIX PRINCIPAL: Uso de var(--secondary-background-color) y var(--text-color) nativos de Streamlit
    return f"""
    <div style='
        line-height: 1.8; 
        font-size: 14px; 
        padding: 15px; 
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1)); 
        color: var(--text-color, inherit); 
        border-radius: 8px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    '>
        {texto_resaltado}
    </div>
    """



st.markdown("---")
st.subheader("Clinical Similarity Network & Topology")
st.markdown("Topological visualization using K-NN and Harmonic Centrality. Identifies the Archetypal Patient within the cluster.")

# 🌟 FIX: Cambiamos a cache_data con TTL (Time To Live) para que no se quede pegado para siempre, 
# o simplemente mantenemos resource pero sabiendo que hay que limpiar la caché.
@st.cache_resource
def load_similarity_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_x = os.path.join(BASE_DIR, 'X_train_proc_llm.npy')
    ruta_ext = os.path.join(BASE_DIR, 'matriz_extended_display_llm.npy')
    ruta_cols = os.path.join(BASE_DIR, 'columnas_display_llm.npy')
    
    if not all(os.path.exists(p) for p in [ruta_x, ruta_ext, ruta_cols]):
        raise FileNotFoundError("Similarity assets missing.")
        
    X_train_proc = np.load(ruta_x)
    matriz_ext = np.load(ruta_ext, allow_pickle=True)
    nombres_columnas = np.load(ruta_cols, allow_pickle=True)
    
    knn_engine = NearestNeighbors(n_neighbors=20, metric='cosine')
    knn_engine.fit(X_train_proc)
    
    return knn_engine, matriz_ext, nombres_columnas, X_train_proc

if 'mostrar_grafo' not in st.session_state:
    st.session_state.mostrar_grafo = False

if st.button("Generate Archetype Graph", key="btn_gen_arch_graph"):
    st.session_state.mostrar_grafo = True

if st.session_state.mostrar_grafo:
    plt.close('all') 
    
    try:
        with st.spinner("Calculating topological metrics and harmonic centrality..."):
            knn, matriz_extended, nombres_columnas, X_train_proc = load_similarity_assets()
            
            prep = pipeline.named_steps['preprocesador']
            X_paciente_proc = prep.transform(df_paciente)
            
            distancias, indices = knn.kneighbors(X_paciente_proc)
            vecinos_idx = indices[0]
            distancias_vecinos = distancias[0]
            
            col_idx = {col: i for i, col in enumerate(nombres_columnas)}
            prefijos_nlp = ('LLM_', 'ING_', 'EVO_', 'DELTA_', 'rango_', 'pluripatologico', 'dias_', 'CIE10_MACRO')
            columnas_comunes_dinamicas = [col for col in nombres_columnas if str(col).startswith(prefijos_nlp)]
            
            COLOR_NEW_PATIENT = '#87CEEB' 
            COLOR_HIST_READMIT = '#FF4444' 
            COLOR_HIST_SAFE = '#00C851'    
            COLOR_ARCHETYPE = '#FFD700' 
            
            SIZE_NEW_PATIENT = 2500 
            
            similitudes_brutas = [max(0, (1 - d)) * 100 for d in distancias_vecinos]
            
            # --- NUEVA LÓGICA DE ESCALADO DE TAMAÑOS (MIN-MAX SCALING) ---
            min_sim = min(similitudes_brutas)
            max_sim = max(similitudes_brutas)
            rango_sim = max_sim - min_sim if max_sim != min_sim else 1.0 # Evitar división por cero
            
            G = nx.Graph()
            nodo_paciente = "Current\nPatient"
            G.add_node(nodo_paciente, color=COLOR_NEW_PATIENT, size=SIZE_NEW_PATIENT, edge_color='black', line_width=3)
            
            nodos_gemelos = []
            info_inspeccion = {}
            
            # --- LISTA DE MARCADORES INVÁLIDOS (Definir justo antes del bucle) ---
            invalid_markers = ["N/A", "MISSING_DATA", "NONE", "NAN", ""]
            
            for i, (idx, similitud_pct) in enumerate(zip(vecinos_idx, similitudes_brutas)):
                reingreso_real = float(matriz_extended[idx, col_idx['target']])
                color_nodo = COLOR_HIST_READMIT if reingreso_real == 1.0 else COLOR_HIST_SAFE
                
                # --- NUEVO: DETECCIÓN ANTICIPADA DE TEXTO ---
                raw_ing = str(matriz_extended[idx, col_idx.get('texto_anamnesis_ingreso', -1)] if 'texto_anamnesis_ingreso' in col_idx else "")
                raw_evo = str(matriz_extended[idx, col_idx.get('texto_evolucion_internacion', -1)] if 'texto_evolucion_internacion' in col_idx else "")
                
                tiene_texto = (raw_ing.upper().strip() not in invalid_markers) or (raw_evo.upper().strip() not in invalid_markers)
                icono_texto = " [TXT]" if tiene_texto else ""
                # --------------------------------------------
                
                # Agregamos el ícono a la etiqueta (esto actualiza el grafo y el selectbox a la vez)
                label_grafo = f"Patient {i+1}{icono_texto}\n({similitud_pct:.1f}%)"
                nodos_gemelos.append(label_grafo)
                
                # Escalamos el tamaño dinámicamente entre 400 (mínimo) y 1800 (máximo)
                norm_sim = (similitud_pct - min_sim) / rango_sim
                tamaño_dinamico = 400 + (norm_sim * 1400)
                
                G.add_node(label_grafo, color=color_nodo, size=tamaño_dinamico, edge_color='white', line_width=1.5)
                G.add_edge(nodo_paciente, label_grafo, weight=similitud_pct/10) 
                
                datos_gemelo = {
                    "similitud": similitud_pct,
                    "outcome_text": "Readmitted" if reingreso_real == 1.0 else "Safe Discharge",
                    "area": matriz_extended[idx, col_idx.get('Area', -1)] if 'Area' in col_idx else "N/A",
                    "sexo": matriz_extended[idx, col_idx.get('sexo', -1)] if 'sexo' in col_idx else "N/A",
                    "interconsultas": matriz_extended[idx, col_idx.get('cantidad_interconsultas', -1)] if 'cantidad_interconsultas' in col_idx else "N/A",
                    "guardia": matriz_extended[idx, col_idx.get('visitas_guardia_6meses_previos', -1)] if 'visitas_guardia_6meses_previos' in col_idx else "N/A",
                    "complejidad": matriz_extended[idx, col_idx.get('IN_COMPLEJIDAD', -1)] if 'IN_COMPLEJIDAD' in col_idx else "N/A",
                    "farmacos": matriz_extended[idx, col_idx.get('FARMACOS_TEXTO', -1)] if 'FARMACOS_TEXTO' in col_idx else "N/A",
                    "diagsec": matriz_extended[idx, col_idx.get('DIAGNOSTICOS_SEC_ACTIVOS', -1)] if 'DIAGNOSTICOS_SEC_ACTIVOS' in col_idx else "N/A",
                    "datos_comunes": {}
                }
                for col_comun in columnas_comunes_dinamicas:
                    datos_gemelo["datos_comunes"][col_comun] = matriz_extended[idx, col_idx[col_comun]]
                info_inspeccion[label_grafo] = datos_gemelo

            X_gemelos = X_train_proc[vecinos_idx]
            dist_gemelos = pairwise_distances(X_gemelos, metric='cosine')
            umbral_conexion = np.percentile(dist_gemelos, 30) 
            
            for i in range(len(vecinos_idx)):
                for j in range(i + 1, len(vecinos_idx)):
                    if dist_gemelos[i, j] < umbral_conexion:
                        peso_interno = max(0.1, 1 - dist_gemelos[i, j])
                        G.add_edge(nodos_gemelos[i], nodos_gemelos[j], weight=peso_interno * 2)
            
            centrality = nx.harmonic_centrality(G)
            centrality.pop(nodo_paciente, None) 
            arquetipo_label = max(centrality, key=centrality.get)
            
            # --- CORRECCIÓN DEL ARQUETIPO ---
            # ELIMINAMOS la línea que modificaba el tamaño: G.nodes[arquetipo_label]['size'] = 1800
            # Solo mantenemos el resaltado del borde dorado y grueso
            G.nodes[arquetipo_label]['edge_color'] = COLOR_ARCHETYPE
            G.nodes[arquetipo_label]['line_width'] = 4.5
            info_inspeccion[arquetipo_label]["is_archetype"] = True

            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42, k=0.85)
            
            node_colors = [data['color'] for node, data in G.nodes(data=True)]
            node_sizes = [data['size'] for node, data in G.nodes(data=True)]
            edge_colors = [data.get('edge_color', 'white') for node, data in G.nodes(data=True)]
            line_widths = [data.get('line_width', 1) for node, data in G.nodes(data=True)]
            
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='#A0A0A0')
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, 
                                   edgecolors=edge_colors, linewidths=line_widths, alpha=0.95)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold', font_color='black')
            ax.axis('off')
            
            col_grafo, col_panel = st.columns([2, 1])
            
            with col_grafo:
                st.pyplot(fig)
                plt.close(fig)
                st.caption("🌟 The node highlighted in gold is the **Archetypal Patient** (highest Harmonic Centrality within the local cohort).")
                
            with col_panel:
                st.markdown("### 🔍 Case Inspector")
                
                lista_nodos = list(info_inspeccion.keys())
                seleccion = st.selectbox("Inspect Patient Twin:", lista_nodos, index=lista_nodos.index(arquetipo_label))
                
                if seleccion:
                    idx_gemelo_matriz = vecinos_idx[lista_nodos.index(seleccion)]
                    data = info_inspeccion[seleccion]
                    
                    if data.get("is_archetype"):
                        st.warning("⭐ **Archetypal Patient (Cluster Hub)**")
                        
                    st.metric(label="Clinical Match", value=f"{data['similitud']:.1f}%")
                    
                    if data['outcome_text'] == "Readmitted":
                        st.error(f"**Outcome:** {data['outcome_text']}")
                    else:
                        st.success(f"**Outcome:** {data['outcome_text']}")
                    
                    st.markdown("---")
                    st.markdown("#### 🤝 Shared Clinical Profile")
                    with st.expander("View Core Matching Variables", expanded=True):
                        for nombre_var_es, valor_var in data['datos_comunes'].items():
                            nombre_en = TRANSLATION_DICT.get(nombre_var_es, nombre_var_es.replace('_', ' ').title())
                            valor_en = format_clinical_value(nombre_var_es, valor_var)
                            st.markdown(f"**{nombre_en}:** {valor_en}")
                    
                    st.markdown("---")
                    
                    traduccion_sexo = {'MASCULINO': 'Male', 'FEMENINO': 'Female'}
                    traduccion_area = {'CIRUGIA': 'Surgery', 'CLINICA MEDICA': 'Internal Medicine', 'TERAPIA INTENSIVA': 'ICU', 'UNIDAD CORONARIA': 'CCU', 'GUARDIA': 'ER', 'TRAUMATOLOGIA': 'Traumatology', 'PEDIATRIA': 'Pediatrics'}
                    traduccion_complejidad = {'ALTA': 'High', 'MEDIA': 'Medium', 'BAJA': 'Low'}
                    
                    def traducir_ninguno(texto):
                        texto_str = str(texto).strip()
                        if texto_str.upper() == 'NINGUNO': return 'None'
                        elif texto_str == '': return 'Unknown'
                        return texto_str

                    sexo_en = traduccion_sexo.get(str(data['sexo']).strip().upper(), data['sexo'])
                    area_en = traduccion_area.get(str(data['area']).strip().upper(), data['area'])
                    complejidad_en = traduccion_complejidad.get(str(data['complejidad']).strip().upper(), data['complejidad'])
                    
                    diagsec_en = traducir_ninguno(data['diagsec'])
                    farmacos_raw = data['farmacos']
                    
                    st.markdown("#### 🏥 Retrospective Details")
                    st.markdown(f"**Sex:** {sexo_en} | **Area:** {area_en}")
                    st.markdown(f"**Complexity:** {complejidad_en}")
                    st.markdown(f"**Prior ER Visits (6m):** {safe_int(data['guardia'])}")
                    st.markdown(f"**Consultations:** {safe_int(data['interconsultas'])}")
                    
                    # --- BLOQUE DE FENOTIPO NARRATIVO LIMPIO ---
                    st.markdown("#### 📜 Narrative Phenotype (Notes)")
                    
                    raw_ing = str(matriz_extended[idx_gemelo_matriz, col_idx.get('texto_anamnesis_ingreso', -1)] if 'texto_anamnesis_ingreso' in col_idx else "")
                    raw_evo = str(matriz_extended[idx_gemelo_matriz, col_idx.get('texto_evolucion_internacion', -1)] if 'texto_evolucion_internacion' in col_idx else "")
                    
                    invalid_markers = ["N/A", "MISSING_DATA", "NONE", "NAN", ""]
                    texto_ing = "" if raw_ing.upper().strip() in invalid_markers else raw_ing
                    texto_evo = "" if raw_evo.upper().strip() in invalid_markers else raw_evo
                    
                    if not texto_ing and not texto_evo:
                        st.info("ℹ️ No narrative clinical notes available for this patient.")
                    else:
                        texto_completo = ""
                        if texto_ing: texto_completo += f"**Admission:**\n{texto_ing}\n\n"
                        if texto_evo: texto_completo += f"**Evolution:**\n{texto_evo}"
                        
                        # --- MODIFICADO: Ahora extraemos un DICCIONARIO ---
                        citas_gemelo = {} 
                        for col_nombre in nombres_columnas:
                            if col_nombre.startswith("TX_"):
                                cita_val = str(matriz_extended[idx_gemelo_matriz, col_idx[col_nombre]])
                                if cita_val and cita_val.strip() not in ["nan", "None", "", "N/A"]:
                                    var_original = col_nombre.replace("TX_", "")
                                    var_traducida = TRANSLATION_DICT.get(var_original, var_original.replace('_', ' ').title())
                                    citas_gemelo[cita_val.strip()] = var_traducida
                        
                            enfermedades_a_resaltar = [
                            # --- Base original ---
                            "diabetes", "hipertensión", "epoc", "neumonía", "tuberculosis", "iam", "acv", "cáncer",
                            "trombosis", "celulitis", "plaquetopenia", "fa", "fibrilación auricular", "insuficiencia cardíaca",
                            "sepsis", "infarto", "arritmia", "infección", "sme compartimental",
                        
                            # --- Cardiovasculares y Hemodinámicas ---
                            "isquemia", "angina", "miocardiopatía", "endocarditis", "pericarditis", "shock",
                            "aneurisma", "taponamiento cardíaco", "tvp", "tromboembolismo", "tep",
                            "hipertensión arterial", "hta", "hipotensión", "bradicardia", "taquicardia", "ic",
                            "sca", "síndrome coronario agudo", "insuficiencia venosa",
                        
                            # --- Respiratorias ---
                            "asma", "bronquitis", "derrame pleural", "edema agudo de pulmón", "eap",
                            "insuficiencia respiratoria", "sdra", "neumotórax", "fibrosis pulmonar", "broncoespasmo",
                        
                            # --- Renales y Urológicas ---
                            "itu", "infección urinaria", "insuficiencia renal", "ira", "irc", "pielonefritis",
                            "litiasis", "nefropatía", "retención aguda de orina", "rao", "hematuria",
                        
                            # --- Metabólicas y Endocrinas ---
                            "hipotiroidismo", "hipertiroidismo", "cetoacidosis", "hipoglucemia", "hiperglucemia",
                            "dislipidemia", "obesidad", "desnutrición", "sme metabólico", "acidosis",
                        
                            # --- Neurológicas y Psiquiátricas ---
                            "convulsión", "epilepsia", "demencia", "alzheimer", "parkinson", "delirium",
                            "encefalopatía", "meningitis", "isquemia cerebral", "hemorragia subaracnoidea",
                            "ataque isquémico transitorio", "ait", "sme confusional", "delirio",
                        
                            # --- Digestivas y Hepáticas ---
                            "cirrosis", "hepatitis", "pancreatitis", "colecistitis", "apendicitis", "peritonitis",
                            "hemorragia digestiva", "hda", "hdb", "úlcera", "úlceras", "úlcera gástrica", 
                            "obstrucción intestinal", "íleo", "isquemia mesentérica", "ascitis", 
                            "insuficiencia hepática", "gastroenteritis", "colitis", "colitis ulcerosa", 
                            "enfermedad de crohn", "diverticulitis", "celiaquía",
                        
                            # --- Hematológicas y Oncológicas ---
                            "anemia", "leucemia", "linfoma", "neutropenia", "coagulopatía", "metástasis",
                            "tumor", "neoplasia", "leucocitosis", "pancitopenia", "mieloma",
                        
                            # --- Infecciosas y Sistémicas ---
                            "bacteriemia", "shock séptico", "covid", "osteomielitis", "fascitis",
                            "candidiasis", "aspergilosis", "vih", "sida", "dengue", "bacteraemia", "sir", "tbc",
                        
                            # --- Traumatológicas, Piel y Quirúrgicas ---
                            "fractura", "luxación", "artrosis", "artritis", "sme de aplastamiento",
                            "úlcera por presión", "úlceras por presión", "escara", "escaras", 
                            "herida quirúrgica", "evisceración", "dehiscencia", "osteomielitis", "necrosis", "gangrena"
                        ]
                        texto_html = renderizar_notas_gemelo(texto_completo, citas_gemelo, enfermedades_a_resaltar)
                        
                        with st.expander("🔍 Inspect Original Clinical Notes", expanded=False):
                            st.caption("🟡 **Yellow:** Extracted Phenotype Evidence | 🔴 **Red:** Disease Mention")
                            st.markdown(texto_html, unsafe_allow_html=True)

                    st.markdown("#### Clinical Background")
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
                            
    except Exception as e:
        st.error("Error generating similarity topology graph.")
        st.warning(f"Technical Detail: {str(e)}")
