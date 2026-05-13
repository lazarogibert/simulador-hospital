import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import dice_ml
import matplotlib.pyplot as plt

# Evitar warnings en la consola del servidor
import warnings
warnings.filterwarnings("ignore")

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
    # 1. Cargar el modelo
    paquete = joblib.load('modelo_reingreso_nlp_41vars.pkl')
    pipeline = paquete['pipeline']
    umbral = paquete['umbral']
    cols_modelo = paquete['nombres_columnas']
    
    # 2. Cargar el Dataset Quimera (Permutado para mantener MAD sin violar privacidad)
    try:
        df_train_sample = pd.read_csv('train_sample_quimera.csv')
    except FileNotFoundError:
        st.error("⚠️ Falta el archivo 'train_sample_quimera.csv'. DiCE no podrá ejecutarse correctamente.")
        # Backup de emergencia (evita que la app crashee, pero DiCE perderá precisión)
        df_train_sample = pd.DataFrame(np.zeros((50, len(cols_modelo))), columns=cols_modelo)
    
    return pipeline, umbral, cols_modelo, df_train_sample

pipeline, umbral, columnas_modelo, df_train_sample = cargar_entorno()

# ==========================================
# 3. INTERFAZ DE CAPTURA (FRICCIÓN CERO)
# ==========================================
st.sidebar.header("🩺 Carga de Fenotipo")
st.sidebar.markdown("Seleccione únicamente las condiciones presentes.")

# --- BLOQUE LLM (Historial y Crónicos) ---
st.sidebar.subheader("1. Terreno del Paciente (Historial)")
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
st.sidebar.subheader("2. Evolución Clínica")
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
# 4. MOTOR DE ENSAMBLADO MATEMÁTICO
# ==========================================
# Inicializar diccionario seguro en 0
paciente_data = {col: 0.0 for col in columnas_modelo}

# Mapear listas de interfaz a variables
for af in af_seleccionados: paciente_data[f"LLM_AF_{af}"] = 1.0
for cro in cronicos_seleccionados: paciente_data[f"LLM_{cro}"] = 1.0

paciente_data['ING_dolor_eva'] = float(ing_dolor)
paciente_data['ING_gravedad_percibida'] = float(ing_grav)
for ing in ing_sel: paciente_data[f"ING_{ing}"] = 1.0

paciente_data['EVO_dolor_eva'] = float(evo_dolor)
paciente_data['EVO_gravedad_percibida'] = float(evo_grav)
for evo in evo_sel: paciente_data[f"EVO_{evo}"] = 1.0

# 🌟 CÁLCULO INVISIBLE DE DELTAS
paciente_data['DELTA_dolor_eva'] = paciente_data['EVO_dolor_eva'] - paciente_data['ING_dolor_eva']
paciente_data['DELTA_gravedad_percibida'] = paciente_data['EVO_gravedad_percibida'] - paciente_data['ING_gravedad_percibida']
paciente_data['DELTA_alteracion_mental'] = paciente_data['EVO_alteracion_mental'] - paciente_data['ING_alteracion_mental']
paciente_data['DELTA_dependencia_funcional'] = paciente_data['EVO_dependencia_funcional'] - paciente_data['ING_dependencia_funcional']
paciente_data['DELTA_portador_dispositivos'] = paciente_data['EVO_portador_dispositivos'] - paciente_data['ING_portador_dispositivos']

# Construir DataFrame final
df_paciente = pd.DataFrame([paciente_data])[columnas_modelo]

# ==========================================
# 5. INFERENCIA Y DASHBOARD PRINCIPAL
# ==========================================
riesgo = pipeline.predict_proba(df_paciente)[0][1]

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Riesgo de Reingreso")
    st.metric(label="Probabilidad a 30 Días", value=f"{riesgo*100:.1f}%")
    
    if riesgo > umbral:
        st.error(f"⚠️ **ALERTA CLÍNICA**\n\nEl paciente supera el umbral de seguridad estricto ({umbral*100:.1f}%).")
    else:
        st.success(f"✅ **ALTA SEGURA**\n\nRiesgo controlado dentro del umbral permitido.")

with col_der:
    st.subheader("Auditoría de Decisión (SHAP)")
    # Extracción segura de etapas del pipeline
    clf = pipeline.named_steps['clasificador']
    prep = pipeline.named_steps['preprocesador']
    X_proc = prep.transform(df_paciente)
    
    # Reemplaza la inicialización de SHAP en la Sección 5 por esto:
    try:
        # Intenta primero con el optimizador de árboles (muy rápido)
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_proc)
    except Exception:
        # Si el modelo es lineal o distinto, usa el explicador genérico
        explainer = shap.LinearExplainer(clf, X_proc) if hasattr(clf, 'coef_') else shap.Explainer(clf, X_proc)
        shap_vals = explainer(X_proc).values
    
    # Control de dimensiones dependiendo del tipo de modelo (RF, CatBoost, XGBoost)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    if len(shap_vals.shape) > 2: shap_vals = shap_vals[:, :, 1]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[0], 
        base_values=explainer.expected_value, 
        data=df_paciente.iloc[0], 
        feature_names=columnas_modelo), 
        show=False, max_display=8
    )
    st.pyplot(fig)

# ==========================================
# 6. NAVEGADOR TERAPÉUTICO (DiCE)
# ==========================================
st.markdown("---")
st.subheader("Navegador Terapéutico (IA Prescriptiva)")

# 🌟 NUEVO: Creamos un Wrapper para sincronizar los DELTAS durante la optimización de DiCE
class ModeloSincronizado:
    def __init__(self, pipeline_original):
        self.pipeline = pipeline_original
        
    def predict_proba(self, X):
        X_sync = X.copy()
        # Forzamos a DiCE a recalcular la matemática si cambia el EVO
        X_sync['DELTA_dolor_eva'] = X_sync['EVO_dolor_eva'] - X_sync['ING_dolor_eva']
        X_sync['DELTA_gravedad_percibida'] = X_sync['EVO_gravedad_percibida'] - X_sync['ING_gravedad_percibida']
        X_sync['DELTA_alteracion_mental'] = X_sync['EVO_alteracion_mental'] - X_sync['ING_alteracion_mental']
        X_sync['DELTA_dependencia_funcional'] = X_sync['EVO_dependencia_funcional'] - X_sync['ING_dependencia_funcional']
        X_sync['DELTA_portador_dispositivos'] = X_sync['EVO_portador_dispositivos'] - X_sync['ING_portador_dispositivos']
        
        return self.pipeline.predict_proba(X_sync)

if riesgo <= umbral:
    st.info("El paciente se encuentra en condiciones óptimas para el egreso. No se requieren contrafactuales.")
else:
    st.warning("El riesgo es alto. Presione el botón para calcular una ruta clínica de estabilización que permita cruzar el umbral.")
    if st.button("Generar Prescripción (DiCE)", type="primary"):
        with st.spinner("Calculando contrafactuales clínicamente viables..."):
            
            # 1. Configurar DiCE con el Modelo Sincronizado
            df_dice_train = df_train_sample.copy()
            df_dice_train['target'] = 0 
            
            d = dice_ml.Data(dataframe=df_dice_train, continuous_features=columnas_modelo, outcome_name='target')
            
            # 🌟 Instanciamos el wrapper en lugar del pipeline crudo
            modelo_blindado = ModeloSincronizado(pipeline)
            m = dice_ml.Model(model=modelo_blindado, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            
            # 2. Definir espacio de búsqueda estricto (Solo EVO y respetando lógica médica)
            variables_accionables = [col for col in columnas_modelo if col.startswith('EVO_')]
            rangos_permitidos = {}
            vars_a_variar = []
            
            for col in variables_accionables:
                val_actual = df_paciente[col].iloc[0]
                
                # Ignorar variables inmutables al momento del alta
                if 'cuidados_paliativos' in col or 'fuga' in col: 
                    continue
                
                # Direccionalidad clínica estricta
                if 'gravedad' in col:
                    if val_actual > 1.0:
                        rangos_permitidos[col] = [1.0, float(val_actual)] # Piso en 1
                        vars_a_variar.append(col)
                elif 'dolor' in col:
                    if val_actual > 0.0: # Para dolor el piso sí es 0
                        rangos_permitidos[col] = [0.0, float(val_actual)]
                        vars_a_variar.append(col)
                else:
                    if val_actual == 1.0:
                        rangos_permitidos[col] = [0, 1]
                        vars_a_variar.append(col)
            
            # 3. Ejecutar contrafactuales
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
                        st.error("No se encontró una ruta matemáticamente viable solo con mejoras clínicas.")
            except Exception as e:
                # Si DiCE falla matemáticamente, capturamos el error sin que la aplicación colapse
                st.error("La estabilización requerida excede las modificaciones clínicamente permitidas con los parámetros actuales.")