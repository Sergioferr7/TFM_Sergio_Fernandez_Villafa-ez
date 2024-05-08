import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

#Se configura el layout
st.set_page_config(layout="wide")

#Se cargan los datos
df = pd.read_csv(r"https://raw.githubusercontent.com/Sergioferr7/TFM_Sergio_Fernandez_Villafa-ez/main/df_saved.csv")

#Se seleccionan las variables numericas
features = df.select_dtypes(include=[np.number])

#Se escalan los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(features)

#Se define la función para transformar la variable value
def format_value(value):
    return f'{value:.0f} €' if pd.notnull(value) else None

#Se define la función para transformar la variables skill
def format_skill(skill):
    return f'{skill:.0f}' if pd.notnull(skill) else None

#Se inician las sesiones
if "player_search_results" not in st.session_state:
    st.session_state["player_search_results"] = None

if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None

if "player_search_results2" not in st.session_state:
    st.session_state["player_search_results2"] = None

#Se da título a la aplicación
st.title("FutbolMetrics: Análisis Avanzado de Jugadores de Fútbol")

#Se crea un contenedor
with st.container():
    st.header("Búsqueda de jugadores similares")
    st.write("Introduce el nombre de un jugador para encontrar los más similares.(Por ejemplo: cristiano ronaldo, l. messi, ederson, t. courtois, sergio ramos, d. alaba, pedri)")
    
    #Se pide al usuario que introduzca el nombre del jugador a estudiar
    selected_player_name = st.text_input("Nombre del jugador.",key="input1").strip().lower()
    
    #Se crea un botón para iniciar la búsqueda
    if st.button("Buscar jugadores similares", key="btn1"):
        player_df = df[df["Name"] == selected_player_name].nlargest(1, "Overall_2022")
        
        if player_df.empty:
            st.session_state["player_search_results"] = None
            st.error("Escribió un nombre erróneo, revise su ortografía y recuerde ponerlo todo en minúsculas.")
        else:
            #Se calcula la similitud
            selected_player_id = player_df["ID"].iloc[0]
            player_index = df.index[df["ID"] == selected_player_id].tolist() 
            cosine_similarities = cosine_similarity(df_scaled, df_scaled[player_index])
                        
            #Se añade la similitud calculada al dataframe
            df["Similarity"] = cosine_similarities
                        
            #Se coge el top 5 más similar 
            top_5_similar = df[df["ID"] != selected_player_id].nlargest(5, "Similarity")
                        
            #Se definen las variables que s emostraran de los jugadores
            additional_columns = ["ID", "Name","Age_2022", "Nationality", "Club_2022", "Value_2022",
                                            "Overall_2022", "Preferred Foot", "Contract Valid Until_2022",
                                            "Finishing_2022", "ShortPassing_2022", "Dribbling_2022",
                                            "Acceleration_2022", "Stamina_2022", "Interceptions_2022",
                                            "GKDiving_2022", "GKReflexes_2022", "Similarity"]
                        
            #Se añade el jugador a buscar al inicio
            selected_player_row = df[df["ID"] == selected_player_id][additional_columns]
            selected_player_row["Similarity"] = 1.0  
            top_players = pd.concat([selected_player_row, top_5_similar[additional_columns]])
                        
                        
            #Se aplican los formatos
            top_players["Value_2022"] = top_players["Value_2022"].apply(format_value)
            skill_columns = ["Finishing_2022", "ShortPassing_2022", "Dribbling_2022", 
                                "Acceleration_2022", "Stamina_2022", "Interceptions_2022", 
                                "GKDiving_2022", "GKReflexes_2022","Contract Valid Until_2022"]
            for column in skill_columns:
                top_players[column] = top_players[column].apply(format_skill)
                        
            #Se elimina ID
            top_players_final = top_players.drop(columns="ID")
                        
            #Se renombra las columnas quitando el sufijo '_2022'
            rename_columns = {
                            "Name": "Nombre",
                            "Club_2022": "Club",
                            "Age_2022": "Edad",
                            "Nationality": "Nacionalidad",
                            "Value_2022": "Valor",
                            "Overall_2022": "Puntuación",
                            "Preferred Foot": "Pie Preferido",
                            "Contract Valid Until_2022": "Último año contrato",
                            "Finishing_2022": "Finalización",
                            "ShortPassing_2022": "Pase Corto",
                            "Dribbling_2022": "Regate",
                            "Acceleration_2022": "Aceleración",
                            "Stamina_2022": "Stamina",
                            "Interceptions_2022": "Intercepciones",
                            "GKDiving_2022": "Estiradas Portero",
                            "GKReflexes_2022": "Reflejos Portero",
                            "Similarity": "Similitud"
            }
                        
            #Se aplica
            top_players_final = top_players_final.rename(columns=rename_columns)

            top_players_final_html = top_players_final.to_html(index=False)

             #SE alinean los resultados de la tabla
            top_players_final_html = top_players_final_html.replace('<th>', '<th style="text-align: center;">').replace('<td>', '<td style="text-align: center;">')
    
            #Se guarda el DataFrame en session_state
            st.session_state["player_search_results"] = top_players_final_html

#Se muestran los resultados
if st.session_state["player_search_results"] is not None:
    st.write("Jugadores similares encontrados:")
    st.markdown(st.session_state["player_search_results"], unsafe_allow_html=True)

    
#Se implementa el segundo algoritmo creado. Se ha elegido el árbol de decisión para hallar la puntuación del jugador:
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Se definen las habilidades sobre las que se ejecutara el algoritmo 2
features2 = ["Finishing", "ShortPassing", "Dribbling", "Acceleration", "Stamina", "Interceptions", "GKDiving", "GKReflexes"]

#Se generan los nombres de columnas para múltiples años
feature_columns = [f'{feature}_{year}' for year in range(2020, 2023) for feature in features2]

#Titulo del 2 algoritmo
st.header("Predicción de Overall para el año 2023")

#Se pide al usuario que introduzca el nombre del jugador a estudiar
selected_player_name2 = st.text_input("Nombre del jugador para predicción de Overall",key="input2").strip().lower()

#Se crea un botón para iniciar la búsqueda
if st.button("Predecir Overall 2023", key="btn2"):
    player_data = df[df["Name"].str.lower() == selected_player_name2.lower()]
    
    if player_data.empty:
        st.session_state["prediction_results"] = None
        st.error("Escribió un nombre erróneo, revise su ortografía y recuerde ponerlo todo en minúsculas.")
    else:
        player_features = player_data[feature_columns].iloc[0]
        #Se definen X e y
        X = df[feature_columns]
        y = df["Overall_2022"]
        
        #Se divide en un 20% para test y un 80% para entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Se crea y entrena en modelo de árboles de decisión en este caso
        dt = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5)
        dt.fit(X_train, y_train)

        player_features_df = pd.DataFrame([player_features.values], columns=feature_columns)
    
        #Se implementa en el modelo de prueba y el final
        y_pred = dt.predict(X_test)
        predicted_overall_2023 = dt.predict(player_features_df)
    
        #Se calculan las métricas para medir el error
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        #Se almacena el Overall de los años anteriores
        overall_2020 = player_data["Overall_2020"].iloc[0]
        overall_2021 = player_data["Overall_2021"].iloc[0]
        overall_2022 = player_data["Overall_2022"].iloc[0]

        #Se guardan los resultados en la sesion
        st.session_state["prediction_results"] = {
        "selected_player_name2": selected_player_name2,
        "overall_2020": overall_2020,
        "overall_2021": overall_2021,
        "overall_2022": overall_2022,
        "predicted_overall_2023": predicted_overall_2023
        }

#Se muestran
if st.session_state["prediction_results"] is not None:
    resultados = st.session_state["prediction_results"]
    st.write(f'Jugador estudiado: {resultados["selected_player_name2"]}')
    st.write(f'Overall en 2020: {resultados["overall_2020"]}')
    st.write(f'Overall en 2021: {resultados["overall_2021"]}')
    st.write(f'Overall en 2022: {resultados["overall_2022"]}')
    st.write(f'Predicción de Overall para el año 2023: {resultados["predicted_overall_2023"]}')

#Algoritmo 3: busqueda del mejor jugador por equipo y posición
from tabulate import tabulate

#Se normalizan las variables input
df["Club_2022"] = df["Club_2022"].str.strip().str.lower()
df["Def_POSITION"] = df["Def_POSITION"].str.strip().str.lower()

#Algoritmo 3: busqueda del mejor jugador por equipo y posición
st.header("Búsqueda del mejor jugador por equipo y posición")

#Se solicita al usuario los valores de entrada
team_name = st.text_input("Por favor, introduzca el nombre del equipo:(ej. real madrid cf, fc barcelona, atlético de madrid, manchester united, ac milan, paris saint-germain ) ",key="input3").strip().lower()
position = st.text_input("Por favor, introduzca la posición a estudiar (DELANTERO, MEDIOCENTRO, DEFENSA, PORTERO): ",key="input4").strip().lower()

#Se realiza el filtrado por lo valores introducidos por el usuario
filtered_df = df[(df["Club_2022"] == team_name) & (df["Def_POSITION"] == position)]

#Se crea un botón para iniciar la búsqueda
if st.button("Encontrar al mejor jugador", key="btn3"):
    #Se comprueba si lo datos introducidos son correctos
    if filtered_df.empty:
        st.session_state["player_search_results2"] = None
        st.error("No se encontraron jugadores que coincidan con los criterios especificados.")
    else:
        #Se filtra por el jugador con mayor puntuación
        max_overall = filtered_df["Overall_2022"].max()
        
        #Si hay más de uno con la misma puntuación lo recogemos
        best_players = filtered_df[filtered_df["Overall_2022"] == max_overall]
    
        #Se crea la tabla con la salida
        html_table = best_players[["Name", "Overall_2022", "Age_2022", "Preferred Foot"]].rename(columns={
            "Name": "Nombre",
            "Overall_2022": "Puntuación",
            "Age_2022": "Edad",
            "Preferred Foot": "Pie Preferido"
            }).to_html(index=False, classes="table table-striped", border=0)

        #Se aplica el estilo
        html_table = html_table.replace('<th>', '<th style="text-align: center;">').replace('<td>', '<td style="text-align: center;">')


        #Se almacena en la sesion correspondiente
        st.session_state["player_search_results2"] = html_table

#Se muestra
if st.session_state["player_search_results2"] is not None:
    st.write(f'Mejores jugadores en la posición de {position} para el equipo {team_name} en 2022:')
    st.markdown(st.session_state["player_search_results2"], unsafe_allow_html=True)
    

