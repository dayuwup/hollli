"""Aplicación del modelo (nivel 1 de la sección F6B).

Lee modelo.joblib, el paquete que guarda el cuaderno 12, y arma sola el
formulario: un campo por cada pista, con el rango que el modelo vio al
entrenar. Para su proyecto solo se cambia el bloque TEXTOS.
"""
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- TEXTOS: lo único que cambian
TITULO = "OrganizaApp · Predicción de pizarras creadas"
DESCRIPCION = "Ingrese los datos del usuario o equipo para predecir cuántas pizarras creará."
RESULTADO = "Pizarras creadas estimadas"                # qué significa el número que se muestra
SI_SUPERA = ""                                         # no aplica en regresión
NO_SUPERA = ""                                         # no aplica en regresión
ETIQUETAS = {                                          # opcional: cómo se pregunta cada pista
    "plan": "Plan del usuario",
    "región": "Región",
    "region": "Región",
    "tareas_creadas": "Tareas creadas",
    "miembros_equipo": "Miembros en el equipo",
    "sesiones_semana": "Sesiones por semana",
}
OPCIONES = {                                           # opcional: cómo se muestran las opciones de una lista
    "plan": {
        "free": "Gratuito",
        "pro": "Pro",
        "enterprise": "Empresa",
    },
}
GRUPOS = {}                                            # agrupamiento: {0: "Nombre del grupo 0", ...}
AVISO = "Ejemplo de clase con datos de la app de organización. Los casos atípicos los revisa el equipo."
# ---------------------------------------------------------------- de aquí para abajo no se cambia

st.set_page_config(page_title=TITULO)


@st.cache_resource                                     # el paquete se abre una sola vez y queda en memoria
def cargar_paquete():
    return joblib.load("modelo.joblib")


paquete = cargar_paquete()
modelo = paquete["modelo"]
TIPO = paquete.get("tipo", "clasificacion")
COLUMNAS = paquete["columnas"]                         # las columnas que vio el modelo, en su orden
PISTAS = paquete.get("pistas", COLUMNAS)               # las columnas del CSV original
CAMPOS = paquete.get("campos", {})                     # cómo es cada pista: número u opciones

st.title(TITULO)
st.write(DESCRIPCION)

with st.form("caso"):                                  # solo calcula al pulsar el botón
    valores = {}
    for pista in PISTAS:
        campo = CAMPOS.get(pista, {"tipo": "numero"})
        texto = ETIQUETAS.get(pista, pista.replace("_", " ").capitalize())
        if campo["tipo"] == "opciones":                # texto o sí/no: una lista para elegir
            valores[pista] = st.selectbox(texto, campo["opciones"],
                                          format_func=lambda v, p=pista: str(OPCIONES.get(p, {}).get(v, v)))
        elif campo.get("entero", True):                # números enteros, dentro del rango que vio el modelo
            valores[pista] = st.number_input(texto, min_value=int(campo.get("min", 0)),
                                             max_value=int(campo.get("max", 10**9)),
                                             value=int(campo.get("valor", campo.get("min", 0))), step=1)
        else:                                          # números con decimales
            valores[pista] = st.number_input(texto, min_value=float(campo["min"]),
                                             max_value=float(campo["max"]), value=float(campo["valor"]))
    enviar = st.form_submit_button("Calcular")

if enviar:
    caso = pd.get_dummies(pd.DataFrame([valores]), dtype=int)   # el texto se vuelve 0 y 1, como en Colab
    caso = caso.reindex(columns=COLUMNAS, fill_value=0)         # mismas columnas y mismo orden que en Colab

    if TIPO == "clasificacion":
        UMBRAL = paquete["umbral"]
        p = float(modelo.predict_proba(caso)[0, 1])            # probabilidad de la clase que se quiere detectar
        st.metric(RESULTADO, f"{p:.0%}")
        if p > UMBRAL:
            st.warning(f"{SI_SUPERA} (supera el umbral de {UMBRAL:.0%}).")
        else:
            st.success(f"{NO_SUPERA} (no supera el umbral de {UMBRAL:.0%}).")
    elif TIPO == "regresion":
        valor = float(modelo.predict(caso)[0])                 # un número: predict, sin umbral
        st.metric(RESULTADO, f"{valor:,.2f}")
    else:
        grupo = int(modelo.predict(caso)[0])                   # el grupo más cercano
        st.metric(RESULTADO, GRUPOS.get(grupo, f"Grupo {grupo}"))

    st.caption(AVISO)
