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
# ==========================================
def normalizar_cie10(codigo):
    if pd.isna(codigo) or not codigo: 
        return "DESCONOCIDO"
    codigo = str(codigo).strip().upper().replace(".", "").replace(" ", "")
    m = re.match(r'^([A-Z])(\d{2})', codigo)
    if not m: 
        return "DESCONOCIDO"
    return f"{m.group(1)}{int(m.group(2)):02d}"

def mapear_cie10_macro(cod):
    if pd.isna(cod) or cod == "DESCONOCIDO":
        return "DESCONOCIDO"

    letra = cod[0]
    try:
        num = int(cod[1:3])
    except ValueError:
        return "DESCONOCIDO"

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
            
    if letra == "Q":
        if 0 <= num <= 7: return "Malformaciones del sistema nervioso (Espina bífida)"
        elif 20 <= num <= 28: return "Malformaciones cardíacas congénitas"
        elif 90 <= num <= 99: return "Anomalías cromosómicas (Síndrome de Down)"
        else: return "Otras malformaciones congénitas"

    if letra == "P":
        if num == 27: return "Enfermedad respiratoria crónica perinatal"
        else: return "DESCONOCIDO"

    if letra == "T":
        if 90 <= num <= 98: return "Secuelas crónicas de traumatismos"
        else: return "DESCONOCIDO"
            
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

    return "DESCONOCIDO"

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador de Alta Segura", layout="wide")
st.title("🏥 Simulador Clínico de Alta Segura (15 Días)")
st.markdown("Herramienta de soporte a la decisión basada en fenotipos narrativos y explicabilidad prescriptiva.")

# ==========================================
# 2. CARGA DE MODELO Y DATOS QUIMERA (CACHÉ)
# ==========================================
@st.cache_resource
def cargar_entorno():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(directorio_actual, 'modelo_reingreso_nlp_41vars.pkl')
    ruta_datos = os.path.join(directorio_actual, 'train_sample_quimera.csv')
    
    paquete = joblib.load(ruta_modelo)
    pipeline = paquete['pipeline']
    umbral = paquete['umbral']
    cols_modelo = paquete['nombres_columnas']
    
    try:
        df_train_sample = pd.read_csv(ruta_datos)
    except FileNotFoundError:
        st.error(f"⚠️ Falta el archivo en: {ruta_datos}")
        df_train_sample = pd.DataFrame(np.zeros((50, len(cols_modelo))), columns=cols_modelo)
    
    return pipeline, umbral, cols_modelo, df_train_sample

pipeline, umbral, columnas_modelo, df_train_sample = cargar_entorno()

# ==========================================
# 3. INTERFAZ DE CAPTURA (FRICCIÓN CERO)
# ==========================================
st.sidebar.header("🩺 Carga de Fenotipo")
st.sidebar.markdown("Seleccione únicamente las condiciones presentes.")

# --- BLOQUE CLÍNICO-DEMOGRÁFICO ---
st.sidebar.subheader("1. Parámetros Base e Ingreso")
cie10_input = st.sidebar.text_input("Motivo de internación (Código CIE-10):", value="I10", help="Ejemplo: I10, E11, J44")
dias_internados = st.sidebar.number_input("Cantidad de días internado:", min_value=1, max_value=150, value=5)

# Opciones completas y mapeo a mayúsculas para la compatibilidad con el pipeline
opciones_edad = ["Menor de edad", "Adulto Joven", "Adulto de mediana edad", "Adulto mayor", "Anciano"]
rango_edad_ui = st.sidebar.selectbox("Rango de Edad del Paciente:", opciones_edad)
rango_edad = rango_edad_ui.upper()

es_pluripatologico = st.sidebar.checkbox("¿El paciente es Pluripatológico?", value=False)

# --- BLOQUE LLM (Historial y Crónicos) ---
st.sidebar.subheader("2. Terreno del Paciente (Historial)")
opciones_af = [
    'autoinmune', 'cardiovascular_otro', 'diabetes', 'hipertension', 
    'metabolico_otro', 'neurologico', 'oncologico', 'psiquiatrico', 'renal', 'respiratorio'
]
af_seleccionados = st.sidebar.multiselect("Antecedentes Familiares (LLM_AF):", opciones_af)

opciones_cronicos = [
    'abandono_medicacion', 'alcoholismo', 'desnutricion_severa', 'drogas_ilicitas', 
    'fragilidad_geriatrica', 'historial_caidas', 'oxigenodependiente', 'polifarmacia', 'tabaquismo_activo'
]
cronicos_seleccionados = st.sidebar.multiselect("Crónicos y Hábitos (LLM):", opciones_cronicos)

# --- BLOQUE ING vs EVO ---
st.sidebar.subheader("3. Evolución Clínica")
c_ing, c_evo = st.sidebar.columns(2)

with c_ing:
    st.markdown("**Al Ingreso (ING)**")
    ing_dolor = st.slider("Dolor Inicial", 0, 10, 0)
    ing_grav = st.slider("Gravedad Inicial", 1, 10, 5)
    
    opc_ing = ['alteracion_mental', 'consultas_reiteradas', 'dependencia_funcional', 'portador_dispositivos', 'riesgo_hemorragico']
    ing_sel = st.multiselect("Complicaciones (ING):", opc_ing)

with c_evo:
    st.markdown("**Al Alta (EVO)**")
    evo_dolor = st.slider("Dolor Actual", 0, 10, 0)
    evo_grav = st.slider("Gravedad Actual", 1, 10, 5)
    
    opc_evo = [
        'aislamiento_infeccioso', 'alteracion_mental', 'complicacion_internacion', 
        'cuidados_paliativos', 'dependencia_funcional', 'fuga_o_alta_irregular', 
        'portador_dispositivos', 'ulceras_presion'
    ]
    evo_sel = st.multiselect("Complicaciones (EVO):", opc_evo)

# ==========================================
# 4. MOTOR DE ENSAMBLADO MATEMÁTICO (BLINDADO)
# ==========================================
# 1. Inicialización ultra-segura basada SOLO en las columnas del modelo entrenado
paciente_data = {}
for col in columnas_modelo:
    if col in ['rango_edad', 'CIE10_MACRO', 'CIE10_SUBMACRO']:
        paciente_data[col] = "DESCONOCIDO"
    else:
        paciente_data[col] = 0.0

# 2. Asignación de nuevas variables (Solo si existen en el modelo)
if 'rango_edad' in paciente_data: paciente_data['rango_edad'] = rango_edad
if 'dias_internados' in paciente_data: paciente_data['dias_internados'] = float(dias_internados)
if 'pluripatologico' in paciente_data: paciente_data['pluripatologico'] = 1.0 if es_pluripatologico else 0.0

# 3. Tratamiento CIE-10
codigo_normalizado = normalizar_cie10(cie10_input)
categoria_cie10 = mapear_cie10_macro(codigo_normalizado)
if 'CIE10_MACRO' in paciente_data: paciente_data['CIE10_MACRO'] = categoria_cie10

# 4. Asignación de variables NLP Crónicas (Solo cambia a 1.0 lo que el médico seleccionó y existe en el modelo)
for af in af_seleccionados: 
    if f"LLM_AF_{af}" in paciente_data: paciente_data[f"LLM_AF_{af}"] = 1.0

for cro in cronicos_seleccionados: 
    if f"LLM_{cro}" in paciente_data: paciente_data[f"LLM_{cro}"] = 1.0

# 5. Escala y Complicaciones (ING y EVO)
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

# 6. Cálculo Seguro de Deltas (Solo se calculan si la columna DELTA existe en el modelo entrenado)
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

# 7. Construcción final del DataFrame
df_paciente = pd.DataFrame([paciente_data])[columnas_modelo]

# ==========================================
# 5. INFERENCIA Y DASHBOARD PRINCIPAL
# ==========================================
riesgo = pipeline.predict_proba(df_paciente)[0][1]

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Riesgo de Reingreso")
    st.metric(label="Probabilidad a 15 Días", value=f"{riesgo*100:.1f}%")
    
    if riesgo > umbral:
        st.error(f"⚠️ **ALERTA CLÍNICA**\n\nEl paciente supera el umbral de seguridad estricto ({umbral*100:.1f}%).")
    else:
        st.success(f"✅ **ALTA SEGURA**\n\nRiesgo controlado dentro del umbral permitido.")

    st.info(f"**Diagnóstico Mapeado:** {categoria_cie10} (Código: {codigo_normalizado})")

with col_der:
    st.subheader("Auditoría de Decisión (SHAP)")
    clf = pipeline.named_steps['clasificador']
    prep = pipeline.named_steps['preprocesador']
    
    # 1. Transformamos los datos (Esto genera la matriz ancha)
    X_proc = prep.transform(df_paciente)
    
    # 2. Extraemos los nombres reales con los que entrena el algoritmo y los limpiamos
    nombres_crudos = prep.get_feature_names_out()
    nombres_limpios = [nombre.replace('num__', '').replace('cat__', '') for nombre in nombres_crudos]
    
    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_proc)
    except Exception:
        explainer = shap.LinearExplainer(clf, X_proc) if hasattr(clf, 'coef_') else shap.Explainer(clf, X_proc)
        shap_vals = explainer(X_proc).values
    
    # Control de dimensiones dependiendo del tipo de modelo (RF, CatBoost, XGBoost)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    if len(shap_vals.shape) > 2: shap_vals = shap_vals[:, :, 1]
    
    # Extracción segura del Expected Value (Base Value)
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = exp_val[1] if len(exp_val) > 1 else exp_val[0]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # 🌟 BLINDAJE DIMENSIONAL: Ahora values, data y feature_names tienen exactamente la misma longitud
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[0], 
        base_values=exp_val, 
        data=X_proc[0], 
        feature_names=nombres_limpios), 
        show=False, max_display=8
    )
    st.pyplot(fig)

# ==========================================
# 6. NAVEGADOR TERAPÉUTICO (DiCE)
# ==========================================
st.markdown("---")
st.subheader("Navegador Terapéutico (IA Prescriptiva)")

class ModeloSincronizado:
    def __init__(self, pipeline_original):
        self.pipeline = pipeline_original
        
    def predict_proba(self, X):
        X_sync = X.copy()
        
        # Recalcular deltas dinámicos SOLO si las columnas están en el modelo
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
    st.info("El paciente se encuentra en condiciones óptimas para el egreso. No se requieren contrafactuales.")
else:
    st.warning("El riesgo es alto. Presione el botón para calcular una ruta clínica de estabilización que permita cruzar el umbral.")
    if st.button("Generar Prescripción (DiCE)", type="primary"):
        with st.spinner("Calculando contrafactuales clínicamente viables..."):
            
            # 1. Preparación del DataFrame base para DiCE
            df_dice_train = df_train_sample.copy()
            df_dice_train['target'] = 0 
            
            # 🌟 BLINDAJE DE TIPOS DE DATOS PARA DiCE
            # Extraemos automáticamente solo las columnas numéricas verdaderas (int, float)
            cols_numericas = df_dice_train.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in cols_numericas:
                cols_numericas.remove('target') # Quitamos el target de la lista de features continuos
            
            # Instanciamos DiCE pasándole únicamente las columnas numéricas
            d = dice_ml.Data(dataframe=df_dice_train, continuous_features=cols_numericas, outcome_name='target')
            
            # 2. Instanciación del Modelo Sincronizado
            modelo_blindado = ModeloSincronizado(pipeline)
            m = dice_ml.Model(model=modelo_blindado, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            
            # 3. Definición del espacio de búsqueda clínico
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
            
            # 4. Ejecución del motor contrafactual
            try:
                if not vars_a_variar:
                    st.error("No hay variables clínicas modificables en la evolución que puedan mejorar el estado del paciente.")
                else:
                    dice_exp = exp.generate_counterfactuals(
                        df_paciente, total_CFs=1, desired_class="opposite", 
                        features_to_vary=vars_a_variar, permitted_range=rangos_permitidos, random_seed=42
                    )
                    
                    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                    if cf_df is not None and not cf_df.empty:
                        st.success("✅ **RUTA CLÍNICA ENCONTRADA:**")
                        for col in vars_a_variar:
                            val_orig = df_paciente.iloc[0][col]
                            val_cf = cf_df.iloc[0][col]
                            
                            if rangos_permitidos[col] == [0, 1]: 
                                val_cf = round(val_cf)
                            
                            if val_orig != val_cf:
                                if 'dolor' in col or 'gravedad' in col:
                                    st.write(f"- 💊 **{col}**: Reducir de [{val_orig:.0f}] ➔ Meta: **[{val_cf:.0f}]**")
                                else:
                                    st.write(f"- 💊 **{col}**: Resolver complicación ➔ **[Ausente]**")
                    else:
                        st.error("No se encontró una ruta matemáticamente viable solo con modificaciones clínicas.")
            except Exception as e:
                st.error("La estabilización requerida excede las modificaciones clínicamente permitidas con los parámetros actuales.")
