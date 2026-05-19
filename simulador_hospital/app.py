import streamlit as st
import pandas as pd
import numpy as np
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
        st.error(f"⚠️ **CLINICAL ALERT**\n\nThe patient exceeds the strict safety threshold ({umbral*100:.1f}%).")
    else:
        st.success(f"✅ **SAFE DISCHARGE**\n\nRisk controlled within the permitted threshold.")

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
    clf = pipeline.named_steps['clasificador']
    prep = pipeline.named_steps['preprocesador']
    
    X_proc = prep.transform(df_paciente)
    nombres_crudos = prep.get_feature_names_out()
    nombres_limpios = [nombre.replace('num__', '').replace('cat__', '') for nombre in nombres_crudos]
    
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

    nombres_limpios_traducidos = []
    for nombre in nombres_limpios:
        if "CIE10_MACRO" in nombre:
            cat_val = nombre.replace("CIE10_MACRO_", "")
            nombres_limpios_traducidos.append(f"Diagnosis: {cie10_ui_dict.get(cat_val, cat_val)}")
        elif "rango_edad" in nombre:
            cat_val = nombre.replace("rango_edad_", "")
            trad = next((k for k, v in opciones_edad_dict.items() if v.upper() == cat_val.upper()), cat_val)
            nombres_limpios_traducidos.append(f"Age: {trad}")
        else:
            nombres_limpios_traducidos.append(shap_ui_dict.get(nombre, nombre))

    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_proc)
    except:
        explainer = shap.LinearExplainer(clf, X_proc) if hasattr(clf, 'coef_') else shap.Explainer(clf, X_proc)
        shap_vals = explainer(X_proc).values
    
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    if len(shap_vals.shape) > 2: shap_vals = shap_vals[:, :, 1]
    
    # 🌟 CORRECCIÓN DE TIPO: Forzamos el valor base a float escalar único
    raw_exp = explainer.expected_value
    base_val = float(raw_exp[1]) if isinstance(raw_exp, (list, np.ndarray)) else float(raw_exp)
    
    # 🌟 LIMPIEZA DE MEMORIA: Previene el texto superpuesto eliminando cachés previas
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Renderizado nativo: No manipulamos texto, garantizamos integridad
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[0], 
        base_values=base_val, 
        data=X_proc[0], 
        feature_names=nombres_limpios_traducidos), 
        show=False, max_display=8
    )
    
    st.pyplot(fig)
    st.caption("📌 **Note:** SHAP values represent the impact on readmission risk (e.g., 0.19 = 19% increase).")
    
# ==========================================
# 6. THERAPEUTIC NAVIGATOR (DiCE)
# ==========================================
st.markdown("---")
st.subheader("Therapeutic Navigator (Prescriptive AI)")

class ModeloSincronizado:
    def __init__(self, pipeline_original):
        self.pipeline = pipeline_original
        
    def predict_proba(self, X):
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
        with st.spinner("Calculating multiple clinically viable alternatives..."):
            
            df_dice_train = df_train_sample.copy()
            df_dice_train['target'] = 0 
            
            df_paciente_para_dice = df_paciente.copy()
            df_paciente_para_dice['target'] = 0
            df_dice_train = pd.concat([df_dice_train, df_paciente_para_dice], ignore_index=True)
            
            cols_numericas = df_dice_train.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in cols_numericas:
                cols_numericas.remove('target') 
            
            d = dice_ml.Data(dataframe=df_dice_train, continuous_features=cols_numericas, outcome_name='target')
            
            modelo_blindado = ModeloSincronizado(pipeline)
            m = dice_ml.Model(model=modelo_blindado, backend="sklearn")
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
            
            try:
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
                        
                        # DICCIONARIO DE TRADUCCIÓN PARA LA SALIDA DE DiCE
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
                                    
                                    if rangos_permitidos[col] == [0, 1]: 
                                        val_cf = round(val_cf)
                                    
                                    if val_orig != val_cf:
                                        cambios_detectados += 1
                                        col_en = evo_output_dict.get(col, col)
                                        
                                        if 'dolor' in col or 'gravedad' in col:
                                            st.write(f"- 💊 **{col_en}**: Reduce from [{val_orig:.0f}] ➔ Target: **[{val_cf:.0f}]**")
                                        else:
                                            st.write(f"- 💊 **{col_en}**: Resolve complication ➔ **[Absent]**")
                                
                                if cambios_detectados == 0:
                                    st.write("This alternative suggests maintaining current parameters based on marginal risk stability.")
                    else:
                        st.error("No mathematically viable routes were found using only clinical modifications.")
            except Exception as e:
                st.error("The required stabilization exceeds the clinically permitted modifications with the current parameters.")
                st.warning(f"🔍 Mathematical Debug: {str(e)}")


# ==========================================
# 7. GLOBAL INTERPRETABILITY (PDP)
# ==========================================
st.markdown("---")
st.subheader("Global Interpretability (Partial Dependence Plots)")
st.markdown("Analyze how the Delta variables influence the average readmission risk. **The green line marks your current patient's position.**")

# Definimos las variables Delta
delta_vars = [
    'DELTA_dolor_eva', 'DELTA_gravedad_percibida', 'DELTA_alteracion_mental', 
    'DELTA_dependencia_funcional', 'DELTA_portador_dispositivos'
]

delta_ui_dict = {
    'DELTA_dolor_eva': 'Pain Delta', 'DELTA_gravedad_percibida': 'Severity Delta',
    'DELTA_alteracion_mental': 'Mental Alteration Delta', 
    'DELTA_dependencia_funcional': 'Functional Dependency Delta',
    'DELTA_portador_dispositivos': 'Device Bearer Delta'
}

var_a_graficar = st.selectbox("Select variable for PDP analysis:", list(delta_ui_dict.keys()), format_func=lambda x: delta_ui_dict[x])

if st.button("Generate PDP"):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    try:
        from sklearn.inspection import PartialDependenceDisplay
        
        # 1. Graficamos el PDP global
        PartialDependenceDisplay.from_estimator(
            pipeline, 
            df_train_sample, 
            features=[var_a_graficar],
            ax=ax,
            kind='average',
            subsample=100
        )
        
        # 2. Obtenemos el valor del paciente
        valor_paciente = float(df_paciente[var_a_graficar].iloc[0])
        
        # 3. Dibujamos el marcador solo si está dentro de los límites visuales
        xlim = ax.get_xlim()
        if xlim[0] <= valor_paciente <= xlim[1]:
            # Línea roja vertical gruesa
            ax.axvline(x=valor_paciente, color='red', linestyle='--', linewidth=2.5, label=f'Patient: {valor_paciente:.1f}')
            
            # Etiqueta de texto para que sea obvio
            ylim = ax.get_ylim()
            ax.text(valor_paciente, ylim[1]*0.9, ' Patient', color='red', fontweight='bold', fontsize=10, rotation=90)
            ax.legend(loc='upper right')
        else:
            st.warning(f"⚠️ Patient value ({valor_paciente:.1f}) is outside the training data range (Limits: {xlim[0]:.1f} to {xlim[1]:.1f}).")
        
        # Estilo final
        ax.set_title(f"PDP: {delta_ui_dict[var_a_graficar]} vs Risk")
        ax.set_ylabel("Partial Dependence (Risk)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Could not generate PDP: {str(e)}")
