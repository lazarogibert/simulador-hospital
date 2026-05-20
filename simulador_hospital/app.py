import streamlit as st
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

import pandas as pd

def mapear_cie10_macro(cod):
    # Si viene nulo del normalizador previo, salimos rápido
    if pd.isna(cod):
        return "Desconocido"

    # Extracción directa, confiando en el normalizador
    letra = cod[0]
    num = int(cod[1:3])

    # --- INFECCIOSAS Y PARASITARIAS CRÓNICAS (A y B) ---
    if letra == "A":
        if 15 <= num <= 19:
            return "Tuberculosis"
        elif 30 <= num <= 30:
            return "Lepra"
        elif 50 <= num <= 53:
            return "Sífilis"
        else:
            return "Otras infecciosas (A)"

    if letra == "B":
        if 15 <= num <= 19:
            return "Hepatitis viral"
        elif 20 <= num <= 24:
            return "Enfermedad por VIH"
        elif 57 <= num <= 57:
            return "Enfermedad de Chagas"
        elif 58 <= num <= 58:
            return "Toxoplasmosis"
        elif 67 <= num <= 67: # 🌟 PARCHE 3
            return "Equinococosis / Hidatidosis"
        elif 90 <= num <= 94: 
            return "Secuelas de enfermedades infecciosas"
        else:
            return "Otras infecciosas (B)"

    # --- ONCOLOGÍA Y HEMATOLOGÍA (C y D) ---
    if letra == "C":
        if 0 <= num <= 14:
            return "Cáncer de labio / boca / faringe"
        elif 15 <= num <= 26:
            return "Cáncer digestivo"
        elif 30 <= num <= 39:
            return "Cáncer respiratorio / intratorácico"
        elif 40 <= num <= 41:
            return "Cáncer de hueso / cartílago"
        elif 43 <= num <= 44:
            return "Melanoma / Cáncer de piel"
        elif 50 <= num <= 50:
            return "Cáncer de mama"
        elif 51 <= num <= 58:
            return "Cáncer genital femenino"
        elif 60 <= num <= 63:
            return "Cáncer genital masculino"
        elif 64 <= num <= 68:
            return "Cáncer de vías urinarias"
        elif 69 <= num <= 72:
            return "Cáncer de sistema nervioso central"
        elif 81 <= num <= 96:
            return "Cáncer linfoide / hematopoyético"
        else:
            return "Otros tumores malignos"

    if letra == "D":
        if 0 <= num <= 48:
            return "Tumores in situ o benignos"
        elif 50 <= num <= 53:
            return "Anemias nutricionales"
        elif 55 <= num <= 59:
            return "Anemias hemolíticas"
        elif 60 <= num <= 64:
            return "Aplasias y otras anemias"
        elif 65 <= num <= 69:
            return "Defectos de coagulación / púrpura"
        elif 80 <= num <= 89:
            return "Trastornos de inmunodeficiencia"
        else:
            return "Otros trastornos de la sangre"

    # --- ENDOCRINAS Y METABÓLICAS (E) ---
    if letra == "E":
        if 0 <= num <= 7:
            return "Tiroides"
        elif 8 <= num <= 13:
            return "Diabetes"
        elif 15 <= num <= 16:
            return "Glucosa / hipoglucemia"
        elif 20 <= num <= 35:
            return "Otros endocrinos y metabólicos"
        elif 65 <= num <= 68:
            return "Obesidad y trastornos de hiperalimentación"
        elif 70 <= num <= 90: 
            if num == 78:
                return "Dislipidemia"
            elif num == 84: # 🌟 PARCHE 3
                return "Fibrosis quística"
            return "Trastornos metabólicos"
        else:
            return "Otros metabólicos / nutricionales"

    # --- PSIQUIATRÍA Y SALUD MENTAL (F) ---
    if letra == "F":
        if 0 <= num <= 9:
            return "Trastornos mentales orgánicos (Demencias)"
        elif 10 <= num <= 19:
            return "Trastornos por uso de sustancias"
        elif 20 <= num <= 29:
            return "Esquizofrenia y trastornos psicóticos"
        elif 30 <= num <= 39:
            return "Trastornos del humor (Afectivos)"
        elif 40 <= num <= 48:
            return "Trastornos neuróticos y de ansiedad"
        elif 50 <= num <= 59:
            return "Trastornos de la conducta alimentaria / sueño"
        elif 60 <= num <= 69:
            return "Trastornos de la personalidad"
        elif 70 <= num <= 79: # 🌟 PARCHE 1
            return "Discapacidad intelectual"
        elif 80 <= num <= 89: # 🌟 PARCHE 1
            return "Trastornos del desarrollo psicobiológico (Autismo)"
        else:
            return "Otros trastornos mentales"

    # --- NEUROLOGÍA (G) ---
    if letra == "G":
        if 10 <= num <= 14:
            return "Atrofias sistémicas del SNC"
        elif 20 <= num <= 26:
            return "Trastornos extrapiramidales y del movimiento (Parkinson)"
        elif 30 <= num <= 32:
            return "Enfermedades degenerativas (Alzheimer)"
        elif 35 <= num <= 37:
            return "Enfermedades desmielinizantes (Esclerosis Múltiple)"
        elif 40 <= num <= 47:
            return "Trastornos episódicos y paroxísticos (Epilepsia, Migraña)"
        elif 50 <= num <= 59:
            return "Trastornos de nervios y plexos"
        elif 60 <= num <= 64:
            return "Polineuropatías"
        elif 70 <= num <= 73:
            return "Enfermedades de la unión neuromuscular (Miastenia)"
        elif 80 <= num <= 83: # 🌟 PARCHE 1
            return "Parálisis cerebral y síndromes paralíticos"
        else:
            return "Otros trastornos neurológicos"

    # --- SENTIDOS (H) ---
    if letra == "H":
        if 0 <= num <= 59:
            return "Ojo"
        elif 60 <= num <= 95:
            return "Oído"
        else:
            return "Otros órganos de los sentidos"

    # --- CARDIOVASCULAR (I) ---
    if letra == "I":
        if 10 <= num <= 15:
            return "Hipertensión"
        elif 20 <= num <= 25:
            return "Cardiopatía isquémica"
        elif 26 <= num <= 28:
            return "Enfermedad cardiopulmonar"
        elif 30 <= num <= 52:
            return "Otras enfermedades del corazón (Insuficiencia Cardíaca)"
        elif 60 <= num <= 69:
            return "Cerebrovascular"
        elif 70 <= num <= 79:
            return "Enfermedades de arterias y capilares"
        elif 80 <= num <= 89:
            return "Enfermedades de venas y vasos linfáticos"
        else:
            return "Otros circulatorios"

    # --- RESPIRATORIO (J) ---
    if letra == "J":
        if 0 <= num <= 6:
            return "Vías respiratorias altas"
        elif 9 <= num <= 18:
            return "Infecciones agudas / neumonía / influenza"
        elif 20 <= num <= 22:
            return "Infecciones respiratorias bajas"
        elif 30 <= num <= 39:
            return "Enfermedades de vías respiratorias superiores"
        elif 40 <= num <= 47:
            return "Asma / EPOC / bronquitis"
        elif 60 <= num <= 70:
            return "Enfermedades del pulmón por agentes externos (Neumoconiosis)"
        elif 80 <= num <= 84:
            return "Enfermedades pulmonares intersticiales"
        else:
            return "Otros respiratorios"

    # --- DIGESTIVO (K) ---
    if letra == "K":
        if 0 <= num <= 14:
            return "Boca / dientes / faringe"
        elif 20 <= num <= 31:
            return "Esófago / estómago / duodeno"
        elif 35 <= num <= 38:
            return "Apendicitis"
        elif 40 <= num <= 46:
            return "Hernias"
        elif 50 <= num <= 52:
            return "Enfermedad de Crohn y colitis"
        elif 55 <= num <= 63:
            return "Otras enfermedades de los intestinos"
        elif 70 <= num <= 77:
            return "Hígado"
        elif 80 <= num <= 87:
            return "Vesícula / vías biliares / páncreas"
        else:
            return "Otros digestivos"

    # --- DERMATOLOGÍA (L) ---
    if letra == "L":
        if 20 <= num <= 30:
            return "Dermatitis y eczema"
        elif 40 <= num <= 45:
            return "Trastornos papuloescamosos (Psoriasis)"
        elif 50 <= num <= 54:
            return "Urticaria y eritema"
        elif 80 <= num <= 99:
            return "Trastornos de las faneras / Otros trastornos de piel"
        else:
            return "Otras enfermedades de la piel"

    # --- OSTEOMUSCULAR (M) ---
    if letra == "M":
        if 0 <= num <= 25:
            return "Artropatías"
        elif 30 <= num <= 36:
            return "Tejido conectivo (Lupus, etc.)"
        elif 40 <= num <= 54:
            return "Dorsopatías"
        elif 60 <= num <= 79:
            return "Tejidos blandos"
        elif 80 <= num <= 94:
            return "Osteopatías y condropatías (Osteoporosis)"
        else:
            return "Otros osteomusculares"

    # --- GENITOURINARIO (N) ---
    if letra == "N":
        if 0 <= num <= 29:
            return "Riñón (Insuficiencia Renal Crónica)"
        elif 30 <= num <= 39:
            return "Vías urinarias bajas"
        elif 40 <= num <= 51:
            return "Genital masculino (Hiperplasia Prostática)"
        elif 60 <= num <= 64:
            return "Mama"
        elif 70 <= num <= 98:
            return "Genital femenino (Endometriosis, etc.)"
        else:
            return "Otros genitourinarios"
            
    # --- CONGÉNITAS (Q) ---
    if letra == "Q":
        if 0 <= num <= 7: # 🌟 PARCHE 1
            return "Malformaciones del sistema nervioso (Espina bífida)"
        elif 20 <= num <= 28:
            return "Malformaciones cardíacas congénitas"
        elif 90 <= num <= 99: # 🌟 PARCHE 1
            return "Anomalías cromosómicas (Síndrome de Down)"
        else:
            return "Otras malformaciones congénitas"

    # --- CONDICIONES PERINATALES CRÓNICAS (P) --- 🌟 PARCHE 3
    if letra == "P":
        if num == 27:
            return "Enfermedad respiratoria crónica perinatal"
        else:
            return pd.NA

    # --- SECUELAS DE TRAUMATISMOS (T) --- 🌟 PARCHE 2
    if letra == "T":
        if 90 <= num <= 98:
            return "Secuelas crónicas de traumatismos"
        else:
            return pd.NA
            
    # --- CONDICIONES ESPECIALES Y POST-COVID (U) --- 
    if letra == "U":
        if num == 9:
            return "Síndrome Post-COVID (Long COVID)"
        else:
            return "Otras condiciones especiales (U)"

    # --- ESTADO DE SALUD Y DISPOSITIVOS (Z) ---
    if letra == "Z":
        if 85 <= num <= 87:
            return "Historia personal de tumores / enfermedades"
        elif 89 <= num <= 90: # 🌟 PARCHE 2
            return "Ausencia adquirida de miembros / órganos"
        elif 93 <= num <= 93: # 🌟 PARCHE 2
            return "Aberturas artificiales (Ostomías)"
        elif 94 <= num <= 94:
            return "Estado de órgano trasplantado"
        elif 95 <= num <= 95:
            return "Presencia de implantes cardíacos / vasculares"
        elif 99 <= num <= 99:
            return "Dependencia de máquinas (diálisis, oxígeno)"
        else:
            return "Otros factores de salud"

    # Casos residuales estrictamente agudos o no clasificables (R, S, V, W, X, Y)
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
# 3. INTERFACE CAPTURE (ZERO FRICTION)
# ==========================================
st.sidebar.header("🩺 Phenotype Loading")
st.sidebar.markdown("Select only the present conditions.")

# --- CLINICAL-DEMOGRAPHIC BLOCK ---
st.sidebar.subheader("1. Baseline Parameters & Admission")
cie10_input = st.sidebar.text_input("Reason for admission (ICD-10 Code):", value="I10", help="Example: I10, E11, J44")
dias_internados = st.sidebar.number_input("Number of days hospitalized:", min_value=1, max_value=150, value=5)

# UI English mappings to Spanish backend categories
opciones_edad_dict = {
    "Minor": "Menor de edad", 
    "Young Adult": "Adulto Joven", 
    "Middle-aged Adult": "Adulto de mediana edad", 
    "Older Adult": "Adulto mayor", 
    "Elderly": "Anciano"
}
rango_edad_ui = st.sidebar.selectbox("Patient Age Range:", list(opciones_edad_dict.keys()))
rango_edad = opciones_edad_dict[rango_edad_ui].upper()

es_pluripatologico = st.sidebar.checkbox("Is the patient Pluripathological?", value=False)

# --- LLM BLOCK (History & Chronic) ---
st.sidebar.subheader("2. Patient Background (History)")

af_dict = {
    'Autoimmune': 'autoinmune', 'Other Cardiovascular': 'cardiovascular_otro', 
    'Diabetes': 'diabetes', 'Hypertension': 'hipertension', 
    'Other Metabolic': 'metabolico_otro', 'Neurological': 'neurologico', 
    'Oncological': 'oncologico', 'Psychiatric': 'psiquiatrico', 
    'Renal': 'renal', 'Respiratory': 'respiratorio'
}
af_seleccionados_ui = st.sidebar.multiselect("Family History (LLM_AF):", list(af_dict.keys()))
af_seleccionados = [af_dict[k] for k in af_seleccionados_ui]

cro_dict = {
    'Medication Abandonment': 'abandono_medicacion', 'Alcoholism': 'alcoholismo', 
    'Severe Malnutrition': 'desnutricion_severa', 'Illicit Drugs': 'drogas_ilicitas', 
    'Geriatric Frailty': 'fragilidad_geriatrica', 'History of Falls': 'historial_caidas', 
    'Oxygen Dependent': 'oxigenodependiente', 'Polypharmacy': 'polifarmacia', 
    'Active Smoking': 'tabaquismo_activo'
}
cronicos_seleccionados_ui = st.sidebar.multiselect("Chronic Conditions & Habits (LLM):", list(cro_dict.keys()))
cronicos_seleccionados = [cro_dict[k] for k in cronicos_seleccionados_ui]

# --- ING vs EVO BLOCK ---
st.sidebar.subheader("3. Clinical Evolution")
c_ing, c_evo = st.sidebar.columns(2)

with c_ing:
    st.markdown("**At Admission (ING)**")
    ing_dolor = st.slider("Initial Pain", 0, 10, 0)
    ing_grav = st.slider("Initial Severity", 1, 10, 5)
    
    ing_dict = {
        'Mental Alteration': 'alteracion_mental', 'Repeated Consultations': 'consultas_reiteradas', 
        'Functional Dependency': 'dependencia_funcional', 'Device Bearer': 'portador_dispositivos', 
        'Hemorrhagic Risk': 'riesgo_hemorragico'
    }
    ing_sel_ui = st.multiselect("Complications (ING):", list(ing_dict.keys()))
    ing_sel = [ing_dict[k] for k in ing_sel_ui]

with c_evo:
    st.markdown("**At Discharge (EVO)**")
    evo_dolor = st.slider("Current Pain", 0, 10, 0)
    evo_grav = st.slider("Current Severity", 1, 10, 5)
    
    evo_dict = {
        'Infectious Isolation': 'aislamiento_infeccioso', 'Mental Alteration': 'alteracion_mental', 
        'Hospitalization Complication': 'complicacion_internacion', 'Palliative Care': 'cuidados_paliativos', 
        'Functional Dependency': 'dependencia_funcional', 'Irregular Discharge / Escape': 'fuga_o_alta_irregular', 
        'Device Bearer': 'portador_dispositivos', 'Pressure Ulcers': 'ulceras_presion'
    }
    evo_sel_ui = st.multiselect("Complications (EVO):", list(evo_dict.keys()))
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
# 5. INFERENCE & MAIN DASHBOARD
# ==========================================
riesgo = pipeline.predict_proba(df_paciente)[0][1]

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Readmission Risk")
    st.metric(label="15-Day Probability", value=f"{riesgo*100:.1f}%")
    
    if riesgo > umbral:
        st.error(
            f"⚠️ **CLINICAL ALERT**\n\n"
            f"The patient exceeds the strict safety threshold ({umbral*100:.1f}%)."
        )
    else:
        st.success(
            "✅ **SAFE DISCHARGE**\n\n"
            "Risk controlled within the permitted threshold."
        )

    # UI Translation Dictionary for CIE-10
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
        "Asma / EPOC / bronquitis": "Asthma / COPD / bronchitis", "Enfermedades del pulmón por agentes externos (Neumoconiosis)": "Lung diseases due to external agents (Neumoconiosis)", 
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
    st.subheader("Decision Audit (SHAP)")

    clf = pipeline.named_steps["clasificador"]
    prep = pipeline.named_steps["preprocesador"]

    # Tomar solo el paciente actual
    df_row = df_paciente.iloc[[0]].copy()

    # Transformación
    X_proc = prep.transform(df_row)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

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

    def limpiar_nombre(nombre):
        n = nombre.replace("num__", "").replace("cat__", "")
        if "CIE10_MACRO_" in n:
            cat_val = n.replace("CIE10_MACRO_", "")
            return f"Diagnosis: {cie10_ui_dict.get(cat_val, cat_val)}"
        if "rango_edad_" in n:
            cat_val = n.replace("rango_edad_", "")
            try:
                trad = next((k for k, v in opciones_edad_dict.items() if v.upper() == cat_val.upper()), cat_val)
            except Exception:
                trad = cat_val
            return f"Age: {trad}"
        return shap_ui_dict.get(n, n)

    def agrupar_feature(nombre):
        n = nombre.replace("num__", "").replace("cat__", "")
        if n.startswith("CIE10_MACRO_"):
            return "Diagnosis"
        if n.startswith("rango_edad_"):
            return "Age Group"
        if n.startswith("ING_"):
            return "Initial status"
        if n.startswith("EVO_"):
            return "Current status"
        if n.startswith("DELTA_"):
            return "Evolution / delta"
        if n.startswith("LLM_"):
            return "Long-term history"
        return "Other clinical variables"

    # SHAP
try:
    explainer = shap.TreeExplainer(clf)
    shap_raw = explainer.shap_values(X_proc, check_additivity=False)
except Exception:
    explainer = shap.Explainer(clf, X_proc)
    shap_raw = explainer(X_proc)

# Extraer valores de clase positiva en binario
if isinstance(shap_raw, list):
    shap_values_paciente = shap_raw[1][0]
elif hasattr(shap_raw, "values"):
    vals = shap_raw.values
    if vals.ndim == 3:
        shap_values_paciente = vals[0, :, 1]
    else:
        shap_values_paciente = vals[0]
else:
    shap_values_paciente = shap_raw[0]

# Forzar vector 1D
shap_values_paciente = np.asarray(shap_values_paciente).ravel()
nombres_crudos = np.asarray(nombres_crudos).ravel()

# Si por alguna razón no coincide el número de features, cortar al mínimo
n_min = min(len(nombres_crudos), len(shap_values_paciente))
nombres_crudos = nombres_crudos[:n_min]
shap_values_paciente = shap_values_paciente[:n_min]

# DataFrame SHAP
df_shap = pd.DataFrame({
    "Feature": [limpiar_nombre(n) for n in nombres_crudos],
    "Group": [agrupar_feature(n) for n in nombres_crudos],
    "SHAP_Value": shap_values_paciente
})

    # Agregar por grupo para evitar que salga una lista larga de diagnósticos/dummies
    df_group = (
        df_shap.groupby("Group", as_index=False)["SHAP_Value"]
        .sum()
    )
    df_group["Abs"] = df_group["SHAP_Value"].abs()

    # Top contribuciones
    df_top = (
        df_group.sort_values("Abs", ascending=False)
        .head(8)
        .sort_values("SHAP_Value", ascending=True)
    )

    # Gráfico custom
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9, 4.8))

    colores = ["#FF0051" if v > 0 else "#008BFB" for v in df_top["SHAP_Value"]]
    ax.barh(
        y=df_top["Group"],
        width=df_top["SHAP_Value"],
        color=colores,
        edgecolor="white",
        height=0.6
    )
    ax.axvline(x=0, color="black", linewidth=1.2)

    ax.set_xlabel("Relative Impact on Risk (SHAP)", fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")

    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        "📌 **Note:** Bar size represents the clinical weight of the variable in the model's decision. "
        "🔴 **Red** pushes risk higher (towards readmission), 🔵 **Blue** pushes risk lower (towards safe discharge)."
    )

# ==========================================
# 6. THERAPEUTIC NAVIGATOR (DiCE - HIGH SECURITY & METRICALLY SOUND)
# ==========================================
import os
import numpy as np
import pandas as pd
import dice_ml

st.markdown("---")
st.subheader("Therapeutic Navigator (Prescriptive AI)")

# --- MODELO BLINDADO CONTRA ARRAYS DE NUMPY Y DESINCRONIZACIÓN ---
class ModeloSincronizado:
    def __init__(self, pipeline_original, columnas_modelo):
        self.pipeline = pipeline_original
        self.columnas_modelo = columnas_modelo
        
    def predict_proba(self, X):
        # SOPORTE DUAL: Si DiCE envía un array de NumPy, lo convertimos a DataFrame sobre la marcha
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
    st.info("The patient is in optimal condition for discharge. No counterfactuals required.")
else:
    st.warning("High risk detected. Click the button to calculate clinical stabilization routes that allow crossing the safety threshold.")
    
    if st.button("Generate Prescription (DiCE)", type="primary"):
        with st.spinner("Calculating multiple clinically viable alternatives from historical matrix..."):
            try:
                # 1. RUTAS ABSOLUTAS CONTROLADAS
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                ruta_ext = os.path.join(BASE_DIR, 'matriz_extended_display.npy')
                ruta_cols = os.path.join(BASE_DIR, 'columnas_display.npy')
                
                matriz_extended = np.load(ruta_ext, allow_pickle=True)
                nombres_columnas = np.load(ruta_cols, allow_pickle=True)
                
                # 2. RECONSTRUCCIÓN CON DESENLACES HISTÓRICOS REALES
                df_background_raw = pd.DataFrame(matriz_extended, columns=nombres_columnas)
                
                columnas_modelo = df_paciente.columns.tolist()
                # Rescatamos las variables del modelo + el target real histórico
                df_dice_train = df_background_raw[columnas_modelo].copy()
                df_dice_train['target'] = df_background_raw['target'].astype(int)
                
                # 3. ALINEACIÓN ESTRICTA DE DTYPES (Fix contra residuos de texto en NumPy)
                for col in columnas_modelo:
                    if df_paciente[col].dtype == object:
                        df_dice_train[col] = df_background_raw[col].astype(str).str.strip().str.upper()
                    else:
                        # Forzamos conversión limpia a números para eliminar formatos como '0.0' en texto
                        df_dice_train[col] = pd.to_numeric(df_background_raw[col], errors='coerce')
                
                # Downsampling estratégico para conservar la fluidez del servidor
                df_dice_train = df_dice_train.sample(n=min(1000, len(df_dice_train)), random_state=42)
                
                # Inyectamos al paciente actual en el subset estructural de DiCE
                df_paciente_para_dice = df_paciente.copy()
                df_paciente_para_dice['target'] = 1 # Frontera de riesgo para el paciente actual
                df_dice_train = pd.concat([df_dice_train, df_paciente_para_dice], ignore_index=True)
                
                # 4. SEGREGACIÓN METODOLÓGICA BLINDADA
                # Metemos TODAS las variables numéricas (incluyendo binarias) como continuas para DiCE
                features_continuas = df_paciente.select_dtypes(include=[np.number]).columns.tolist()
                features_categoricas = [col for col in columnas_modelo if col not in features_continuas]
                
                # 5. INICIALIZACIÓN DEL ENTORNO DiCE CON APOYO DUAL
                d = dice_ml.Data(
                    dataframe=df_dice_train, 
                    continuous_features=features_continuas, 
                    outcome_name='target'
                )
                
                # Pasamos las columnas explícitamente a nuestro sincronizador
                modelo_sincronizado = ModeloSincronizado(pipeline, columnas_modelo)
                m = dice_ml.Model(model=modelo_sincronizado, backend="sklearn")
                exp = dice_ml.Dice(d, m, method="random")
                
                # 6. LIMITADORES CLÍNICOS Y DIRECCIONALIDAD ACCIONABLE
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
                        # Si la complicación está activa (1.0), le permitimos a DiCE apagarla (0)
                        if val_actual == 1.0:
                            rangos_permitidos[col] = [0, 1]
                            vars_a_variar.append(col)
                
                # 7. EJECUCIÓN Y RENDERIZADO VISUAL
                if not vars_a_variar:
                    st.error("There are no modifiable clinical variables in the patient's evolution that can improve their condition.")
                else:
                    dice_exp = exp.generate_counterfactuals(
                        df_paciente, total_CFs=5, desired_class="opposite", 
                        features_to_vary=vars_a_variar, permitted_range=rangos_permitidos, random_seed=42
                    )
                    
                    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                    if cf_df is not None and not cf_df.empty:
                        st.success(f"✅ **{len(cf_df)} ALTERNATIVE CLINICAL ROUTES FOUND:**")
                        st.markdown("The medical staff can select the most feasible option according to the ward scenario:")
                        
                        evo_output_dict = {
                            'EVO_dolor_eva': 'Current Pain',
                            'EVO_gravedad_percibida': 'Current Severity',
                            'EVO_aislamiento_infeccioso': 'Infectious Isolation',
                            'EVO_alteracion_mental': 'Mental Alteration',
                            'EVO_complicacion_internacion': 'Hospitalization Complication',
                            'EVO_cuidados_paliativos': 'Palliative Care',
                            'EVO_dependencia_funcional': 'Functional Dependency',
                            'EVO_fuga_o_alta_irregular': 'Irregular Discharge / Escape',
                            'EVO_portador_dispositivos': 'Device Bearer',
                            'EVO_ulceras_presion': 'Pressure Ulcers'
                        }
                        
                        for r_idx in range(len(cf_df)):
                            with st.expander(f"➔ 🛤️ Alternative Therapeutic Option {r_idx + 1}"):
                                cambios_detectados = 0
                                for col in vars_a_variar:
                                    val_orig = df_paciente.iloc[0][col]
                                    val_cf = cf_df.iloc[r_idx][col]
                                    
                                    # 🚨 CAMBIO 1: Verificamos el tipo numérico entero en el df_paciente nativo
                                    if col in features_continuas and df_paciente[col].dtype in [np.int64, np.int32, int]:
                                        val_cf = round(val_cf)
                                    
                                    if val_orig != val_cf:
                                        cambios_detectados += 1
                                        col_en = evo_output_dict.get(col, col)
                                        
                                        if 'dolor' in col or 'gravedad' in col:
                                            st.write(f"- 💊 **{col_en}**: Reduce from [{val_orig:.0f}] ➔ Target: **[{val_cf:.0f}]**")
                                        else:
                                            # 🚨 CAMBIO 2: Evaluamos dinámicamente el estado binario objetivo
                                            status_en = "Absent" if val_cf == 0 else "Present"
                                            st.write(f"- 💊 **{col_en}**: Target status ➔ **[{status_en}]**")
                                            
                                if cambios_detectados == 0:
                                    st.write("This alternative suggests maintaining current parameters based on marginal risk stability.")
                    
                    
                    else:
                        st.error("No mathematically viable routes were found using only clinical modifications.")
                        
            except Exception as e:
                st.error("The required stabilization exceeds the clinically permitted modifications with the current parameters.")
                st.warning(f"🔍 Technical Context: {str(e)}")

# ==========================================
# 7. GLOBAL INTERPRETABILITY (PDP GRID) - HIGH SECURITY & PERSISTENT
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("---")
st.subheader("Global Interpretability (Partial Dependence Plots)")
st.markdown("Analysis of the model's behavior across all evolution metrics simultaneously against the cohort distribution.")

# Configuración de variables fijas para la grilla
binary_deltas = [
    'DELTA_alteracion_mental', 
    'DELTA_dependencia_funcional', 
    'DELTA_portador_dispositivos'
]

delta_ui_dict = {
    'DELTA_dolor_eva': 'Pain Delta', 
    'DELTA_gravedad_percibida': 'Severity Delta',
    'DELTA_alteracion_mental': 'Mental Alteration Delta', 
    'DELTA_dependencia_funcional': 'Functional Dependency Delta',
    'DELTA_portador_dispositivos': 'Device Bearer Delta'
}

# --- CONTROL DE PERSISTENCIA (UI FIX) ---
if 'mostrar_pdp' not in st.session_state:
    st.session_state.mostrar_pdp = False

col_btn, _ = st.columns([1, 3])
with col_btn:
    if st.button("Generate All PDPs", type="secondary"):
        st.session_state.mostrar_pdp = True

# El bloque se renderiza de forma persistente si el estado es True
if st.session_state.mostrar_pdp:
    plt.close('all')
    
    try:
        from sklearn.inspection import PartialDependenceDisplay
        
        with st.spinner("Reconstructing cohort alignment and generating visualization grid..."):
            # 1. CARGA SEGURA MEDIANTE RUTAS ABSOLUTAS
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            ruta_ext = os.path.join(BASE_DIR, 'matriz_extended_display.npy')
            ruta_cols = os.path.join(BASE_DIR, 'columnas_display.npy')
            
            matriz_extended = np.load(ruta_ext, allow_pickle=True)
            nombres_columnas = np.load(ruta_cols, allow_pickle=True)
            
            # 2. ALINEACIÓN MAESTRA DEL DATAFRAME DE TRASFONDO
            df_background_raw = pd.DataFrame(matriz_extended, columns=nombres_columnas)
            columnas_modelo = df_paciente.columns.tolist()
            
            df_pdp_train = pd.DataFrame(index=range(len(df_background_raw)))
            
            # 3. BLINDAJE DE DTYPES Y MAPEO DEFENSIVO DE NULOS
            for col in columnas_modelo:
                if df_paciente[col].dtype == object:
                    # Si la variable es categórica, estandarizamos texto pero preservamos NaNs reales para el Imputer
                    df_pdp_train[col] = df_background_raw[col].apply(
                        lambda x: np.nan if pd.isna(x) or str(x).strip().upper() in ['NAN', 'NONE', 'NULL', ''] 
                        else str(x).strip().upper()
                    )
                else:
                    # Si es numérica, forzamos casteo numérico limpio
                    df_pdp_train[col] = pd.to_numeric(df_background_raw[col], errors='coerce')
            
            # 4. GENERACIÓN DE LA GRILLA PDP (Layout optimizado de 3 columnas)
            features_to_plot = list(delta_ui_dict.keys())
            
            disp = PartialDependenceDisplay.from_estimator(
                pipeline, 
                df_pdp_train, 
                features=features_to_plot,
                n_cols=3,
                kind='average',
                subsample=150, # Muestra representativa segura para la CPU del servidor
                random_state=42
            )
            
            fig = disp.figure_
            fig.set_size_inches(15, 9)
            axes_flat = disp.axes_.flatten()
            
            # 5. PERSONALIZACIÓN INDIVIDUAL DE SUBPLOTS
            for idx, var in enumerate(features_to_plot):
                ax_real = axes_flat[idx]
                
                # Rescatamos el valor actual de este paciente específico
                valor_paciente = float(df_paciente[var].iloc[0])
                
                # Inyección de la directriz del paciente (Línea roja discontinua frontal)
                ax_real.axvline(x=valor_paciente, color='#FF0000', linestyle='--', linewidth=2.5, 
                                label=f'Patient: {valor_paciente:.1f}', zorder=10)
                
                # Ajuste adaptativo de los límites de los ejes (Especial para deltas binarios)
                if var in binary_deltas:
                    ax_real.set_xlim(-1.1, 1.1)
                    ax_real.set_xticks([-1, 0, 1])
                else:
                    curr_min, curr_max = ax_real.get_xlim()
                    ax_real.set_xlim(min(curr_min, valor_paciente - 1), max(curr_max, valor_paciente + 1))
                
                # Formateo estético y traducción de etiquetas
                ax_real.set_title(f"{delta_ui_dict[var]} vs Risk", fontsize=11, fontweight='bold')
                ax_real.set_xlabel(delta_ui_dict[var], fontsize=9)
                ax_real.set_ylabel("Partial Dependence (Risk)", fontsize=9)
                ax_real.grid(True, linestyle='--', alpha=0.5, zorder=0)
                ax_real.legend(loc='best', frameon=True, fontsize=8)
            
            # 6. BLINDAJE VISUAL: Ocultamos de forma segura los cuadrantes sobrantes de la cuadrícula
            for ax_sobrante in axes_flat[len(features_to_plot):]:
                if ax_sobrante is not None:
                    ax_sobrante.axis('off')
                
            fig.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error generating parallel global interpretability suite: {str(e)}")

# ==========================================
# 8. CLINICAL SIMILARITY NETWORK (ARCHEGO ADVANCED UI)
# ==========================================
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors

# --- DICCIONARIO DE TRADUCCIÓN CLÍNICA (ES -> EN) ---
TRANSLATION_DICT = {
    'dias_internados': 'Length of Stay (Days)',
    'rango_edad': 'Age Range',
    'pluripatologico': 'Multimorbidity',
    'CIE10_MACRO': 'Primary Diagnosis (ICD-10)',
    # LLM (Antecedentes y Riesgos)
    'LLM_tabaquismo_activo': 'Active Smoking',
    'LLM_alcoholismo': 'Alcoholism',
    'LLM_drogas_ilicitas': 'Illicit Drug Use',
    'LLM_fragilidad_geriatrica': 'Geriatric Frailty',
    'LLM_polifarmacia': 'Polypharmacy',
    'LLM_desnutricion_severa': 'Severe Malnutrition',
    'LLM_oxigenodependiente': 'Oxygen Dependent',
    'LLM_historial_caidas': 'History of Falls',
    'LLM_abandono_medicacion': 'Medication Non-adherence',
    'LLM_AF_diabetes': 'FHx: Diabetes',
    'LLM_AF_hipertension': 'FHx: Hypertension',
    'LLM_AF_cardiovascular_otro': 'FHx: Other Cardiovascular',
    'LLM_AF_oncologico': 'FHx: Oncology',
    'LLM_AF_metabolico_otro': 'FHx: Other Metabolic',
    'LLM_AF_neurologico': 'FHx: Neurology',
    'LLM_AF_psiquiatrico': 'FHx: Psychiatry',
    'LLM_AF_respiratorio': 'FHx: Respiratory',
    'LLM_AF_renal': 'FHx: Renal',
    'LLM_AF_autoinmune': 'FHx: Autoimmune',
    # ING (Ingreso)
    'ING_dolor_eva': 'Admission: Pain (VAS)',
    'ING_gravedad_percibida': 'Admission: Perceived Severity',
    'ING_alteracion_mental': 'Admission: Altered Mental Status',
    'ING_dependencia_funcional': 'Admission: Functional Dependence',
    'ING_portador_dispositivos': 'Admission: Medical Devices',
    'ING_consultas_reiteradas': 'Admission: Repeated Consultations',
    'ING_riesgo_hemorragico': 'Admission: Hemorrhagic Risk',
    # EVO (Evolución)
    'EVO_dolor_eva': 'Evolution: Pain (VAS)',
    'EVO_gravedad_percibida': 'Evolution: Perceived Severity',
    'EVO_alteracion_mental': 'Evolution: Altered Mental Status',
    'EVO_dependencia_funcional': 'Evolution: Functional Dependence',
    'EVO_portador_dispositivos': 'Evolution: Medical Devices',
    'EVO_complicacion_internacion': 'Evolution: Hospital Complication',
    'EVO_fuga_o_alta_irregular': 'Evolution: Irregular Discharge (AMA)',
    'EVO_cuidados_paliativos': 'Evolution: Palliative Care',
    'EVO_ulceras_presion': 'Evolution: Pressure Ulcers',
    'EVO_aislamiento_infeccioso': 'Evolution: Infectious Isolation',
    # DELTA (Variación)
    'DELTA_dolor_eva': 'Δ Pain (VAS)',
    'DELTA_gravedad_percibida': 'Δ Perceived Severity',
    'DELTA_alteracion_mental': 'Δ Altered Mental Status',
    'DELTA_dependencia_funcional': 'Δ Functional Dependence',
    'DELTA_portador_dispositivos': 'Δ Medical Devices'
}

def format_clinical_value(key_es, value):
    """Traduce y formatea los valores numéricos/categóricos según su tipo."""
    val_str = str(value).strip().upper()
    
    # 1. Traducción de Categorías Clínicas de Texto
    if key_es == 'rango_edad':
        traducciones_edad = {
            'ADULTO DE MEDIANA EDAD': 'Middle-Aged Adult',
            'ADULTO MAYOR': 'Senior Adult',
            'ADULTO JOVEN': 'Young Adult',
            'ANCIANO': 'Elderly'
        }
        return traducciones_edad.get(val_str, value)
        
    if key_es == 'CIE10_MACRO':
        traducciones_cie = {
            'CARDIOPATÍA ISQUÉMICA': 'Ischemic Heart Disease',
            'HIPERTENSIÓN': 'Hypertension',
            'DIABETES': 'Diabetes',
            'ENFERMEDAD CARDIOPULMONAR': 'Cardiopulmonary Disease'
            # Puedes añadir más aquí según vayan apareciendo en tu dataset
        }
        return traducciones_cie.get(val_str, value)

    # 2. Manejo de Booleanos (Convertir 1/0 a Yes/No)
    # Identificamos variables booleanas por su nombre para no alterar escalas como 'dolor_eva'
    bool_suffixes = ('_mental', '_funcional', '_dispositivos', '_reiteradas', 
                     '_hemorragico', '_internacion', '_irregular', '_paliativos', 
                     '_presion', '_infeccioso')
    
    if key_es.startswith('LLM_') or key_es == 'pluripatologico' or \
       (key_es.endswith(bool_suffixes) and not key_es.startswith('DELTA_')):
        try:
            return "Yes" if float(value) == 1.0 else "No"
        except ValueError:
            pass
            
    # 3. Limpieza de Numéricos (Quitar .0 a los enteros)
    try:
        f_val = float(value)
        if f_val.is_integer():
            return str(int(f_val))
    except ValueError:
        pass

    return value

st.markdown("---")
st.subheader("Clinical Similarity Network")
st.markdown("Topological visualization of historical cases. Nodes are sized by clinical similarity.")

# --- FUNCIÓN SEGURA DE CONVERSIÓN ---
def safe_int(value, default="N/A"):
    """Evita que la app se rompa si hay NaNs o strings vacíos en los datos numéricos."""
    try:
        if pd.isna(value) or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

# --- CARGA SEGURA DE ACTIVOS DINÁMICOS (Caché y Rutas Absolutas) ---
@st.cache_resource
def load_similarity_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_x = os.path.join(BASE_DIR, 'X_train_proc.npy')
    ruta_ext = os.path.join(BASE_DIR, 'matriz_extended_display.npy')
    ruta_cols = os.path.join(BASE_DIR, 'columnas_display.npy')
    
    X_train_proc = np.load(ruta_x)
    matriz_ext = np.load(ruta_ext, allow_pickle=True)
    nombres_columnas = np.load(ruta_cols, allow_pickle=True)
    
    # Búsqueda ampliada a 20 vecinos
    knn_engine = NearestNeighbors(n_neighbors=20, metric='cosine')
    knn_engine.fit(X_train_proc)
    
    return knn_engine, matriz_ext, nombres_columnas

# --- GESTIÓN DE ESTADO DE STREAMLIT (UI FIX) ---
if 'mostrar_grafo' not in st.session_state:
    st.session_state.mostrar_grafo = False

if st.button("Generate Advanced Graph"):
    st.session_state.mostrar_grafo = True

# Todo el bloque se ejecuta si el estado es True, protegiendo el panel lateral
if st.session_state.mostrar_grafo:
    plt.close('all')
    
    try:
        # 1. Cargamos el motor y los datos dinámicos
        knn, matriz_extended, nombres_columnas = load_similarity_assets()
        
        # Transformamos al paciente actual (Asumiendo que df_paciente y pipeline están en scope)
        prep = pipeline.named_steps['preprocesador']
        X_paciente_proc = prep.transform(df_paciente)
        
        # Ejecutamos la búsqueda
        distancias, indices = knn.kneighbors(X_paciente_proc)
        vecinos_idx = indices[0]
        distancias_vecinos = distancias[0]
        
        # 2. MAPEO DINÁMICO DE ÍNDICES
        col_idx = {col: i for i, col in enumerate(nombres_columnas)}
        
        # Identificamos dinámicamente las columnas en común usando tu lista blanca
        prefijos_nlp = ('LLM_', 'ING_', 'EVO_', 'DELTA_', 'rango_', 'pluripatologico', 'dias_', 'CIE10_MACRO')
        columnas_comunes_dinamicas = [col for col in nombres_columnas if str(col).startswith(prefijos_nlp)]
        
        # 3. CONFIGURACIÓN VISUAL Y ESCALADO
        COLOR_NEW_PATIENT = '#87CEEB' 
        COLOR_HIST_READMIT = '#FF0000' # Rojo para reingresos
        COLOR_HIST_SAFE = '#228B22'    # Verde para altas seguras
        
        SIZE_NEW_PATIENT = 2000 
        SIZE_MAX_TWIN = 1400    
        SIZE_MIN_TWIN = 150     

        # Pre-calculamos las similitudes corregidas (evitando negativos) y buscamos límites
        similitudes_brutas = [max(0, (1 - d)) * 100 for d in distancias_vecinos]
        sim_max = max(similitudes_brutas)
        sim_min = min(similitudes_brutas)

        # 4. CONSTRUCCIÓN DEL GRAFO
        G = nx.Graph()
        G.add_node("Current\nPatient", color=COLOR_NEW_PATIENT, size=SIZE_NEW_PATIENT)
        
        info_inspeccion = {}
        
        for i, (idx, similitud_pct) in enumerate(zip(vecinos_idx, similitudes_brutas)):
            reingreso_real = float(matriz_extended[idx, col_idx['target']])
            color_nodo = COLOR_HIST_READMIT if reingreso_real == 1.0 else COLOR_HIST_SAFE
            
            # Escalado Dinámico Exagerado
            if sim_max > sim_min:
                factor_exagerado = ((similitud_pct - sim_min) / (sim_max - sim_min)) ** 1.5 
            else:
                factor_exagerado = 1.0
                
            scaled_size = SIZE_MIN_TWIN + factor_exagerado * (SIZE_MAX_TWIN - SIZE_MIN_TWIN)
            label_grafo = f"Twin {i+1}\n({similitud_pct:.1f}%)"
            
            G.add_node(label_grafo, color=color_nodo, size=scaled_size)
            G.add_edge("Current\nPatient", label_grafo, weight=(0.5 + (factor_exagerado * 2.5)))
            
            # 5. EXTRACCIÓN DE DATOS PARA EL PANEL
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
                
            info_inspeccion[f"Twin {i+1}"] = datos_gemelo

        # 6. RENDERIZADO DEL GRAFO (Ajustado para 20 nodos)
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=0.9) # k=0.9 separa más los nodos
        
        node_colors = [data['color'] for node, data in G.nodes(data=True)]
        node_sizes = [data['size'] for node, data in G.nodes(data=True)]
        edge_weights = [G[u][v]['weight'] for u,v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, alpha=0.4, edge_color='#999999')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, 
                               edgecolors='white', linewidths=2, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold', font_color='black')
        ax.axis('off')
        
        # 7. INTERFAZ DEL PANEL LATERAL
        col_grafo, col_panel = st.columns([2, 1])
        
        with col_grafo:
            st.pyplot(fig)
            
        with col_panel:
            st.markdown("### 🔍 Case Inspector")
            seleccion = st.selectbox("Inspect Patient Twin:", list(info_inspeccion.keys()))
            
            if seleccion:
                data = info_inspeccion[seleccion]
                st.metric(label="Clinical Match", value=f"{data['similitud']:.1f}%")
                
                if data['outcome_text'] == "Readmitted":
                    st.error(f"**Outcome:** {data['outcome_text']}")
                else:
                    st.success(f"**Outcome:** {data['outcome_text']}")
                
                st.markdown("---")
                
                # 🤝 SECCIÓN 1: LO QUE TIENEN EN COMÚN (Dinámico y Traducido)
                st.markdown("#### 🤝 Shared Matching Criteria")
                with st.expander("View Clinical Profile Core", expanded=True):
                    for nombre_var_es, valor_var in data['datos_comunes'].items():
                        
                        # 1. Obtenemos el nombre en inglés del diccionario (si no existe, usa el original limpio)
                        nombre_en = TRANSLATION_DICT.get(nombre_var_es, nombre_var_es.replace('_', ' ').title())
                        
                        # 2. Formateamos y traducimos el valor
                        valor_en = format_clinical_value(nombre_var_es, valor_var)
                        
                        # 3. Renderizamos en la UI
                        st.markdown(f"**{nombre_en}:** {valor_en}")
                
                st.markdown("---")
                
                # -----------------------------------------------------------
                # DICCIONARIOS DE TRADUCCIÓN PARA VARIABLES EXTRA
                # -----------------------------------------------------------
                traduccion_sexo = {
                    'MASCULINO': 'Male',
                    'FEMENINO': 'Female'
                }
                
                traduccion_area = {
                    'CIRUGIA': 'Surgery',
                    'CLINICA MEDICA': 'Internal Medicine',
                    'TERAPIA INTENSIVA': 'Intensive Care Unit (ICU)',
                    'UNIDAD CORONARIA': 'Coronary Care Unit (CCU)',
                    'GUARDIA': 'Emergency Room (ER)',
                    'TRAUMATOLOGIA': 'Traumatology',
                    'PEDIATRIA': 'Pediatrics'
                }
                
                traduccion_complejidad = {
                    'ALTA': 'High',
                    'MEDIA': 'Medium',
                    'BAJA': 'Low'
                }
                
                # Función rápida para traducir "Ninguno"
                def traducir_ninguno(texto):
                    texto_str = str(texto).strip()
                    if texto_str.upper() == 'NINGUNO':
                        return 'None'
                    elif texto_str == '':
                        return 'Unknown'
                    return texto_str

                # Aplicamos las traducciones de forma segura
                sexo_en = traduccion_sexo.get(str(data['sexo']).strip().upper(), data['sexo'])
                area_en = traduccion_area.get(str(data['area']).strip().upper(), data['area'])
                complejidad_en = traduccion_complejidad.get(str(data['complejidad']).strip().upper(), data['complejidad'])
                
                diagsec_en = traducir_ninguno(data['diagsec'])
                farmacos_en = traducir_ninguno(data['farmacos'])
                
                # -----------------------------------------------------------
                # 🏥 SECCIÓN 2: LA INFORMACIÓN ADICIONAL (UI Traducida)
                # -----------------------------------------------------------
                st.markdown("#### 🏥 Retrospective Extra Details")
                st.markdown(f"**Sex:** {sexo_en} | **Area:** {area_en}")
                st.markdown(f"**Complexity:** {complejidad_en}")
                st.markdown(f"**Prior ER Visits (6m):** {safe_int(data['guardia'])}")
                st.markdown(f"**Consultations:** {safe_int(data['interconsultas'])}")
                
                st.markdown("#### Clinical Profile")
                st.markdown(f"**Secondary Diagnoses:**\n{diagsec_en}")
                # --- TRADUCCIÓN DINÁMICA DE LA LISTA DE FÁRMACOS (CASE-INSENSITIVE) ---
                farmacos_raw = data['farmacos']
                if str(farmacos_raw).strip().upper() in ('NINGUNO', '', 'NONE'):
                    farmacos_en = 'None'
                else:
                    # Dividimos el string por la coma para traducir cada familia por separado
                    lista_farmacos = [f.strip() for f in str(farmacos_raw).split(',')]
                    
                    # 🚨 FIX: Creamos un diccionario temporal con las claves en mayúsculas para emparejar
                    dict_farmacos_upper = {k.upper(): v for k, v in FARMACOS_TRANSLATION_DICT.items()}
                    
                    # Buscamos forzando mayúsculas, si no se encuentra preservamos el formato original limpio
                    lista_traducida = [dict_farmacos_upper.get(f.upper(), f.strip().title()) for f in lista_farmacos]
                    
                    # Unimos de nuevo en un string con saltos de línea para que quede ordenado en la interfaz
                    farmacos_en = "\n".join([f"{f}" for f in lista_traducida])

                # --- RENDERIZADO FINAL EN LA UI ---
                st.markdown("#### Clinical Profile")
                st.markdown(f"**Secondary Diagnoses:**\n{diagsec_en}")
                st.markdown(f"**Medications:**")
                # Usamos st.markdown o st.write para renderizar la lista limpia
                if farmacos_en == 'None':
                    st.markdown("None")
                else:
                    for f in lista_traducida:
                        st.markdown(f"- {f}")
    except Exception as e:
        st.error(f"Error generating similarity graph: {str(e)}")
