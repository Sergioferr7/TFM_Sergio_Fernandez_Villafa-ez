import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Se carga el conjunto de datos
df = pd.read_csv(r'https://raw.githubusercontent.com/Sergioferr7/TFM_Sergio_Fernandez_Villafa-ez/main/df_saved.csv')


#Se seleccionan las variables numericas
features = df.select_dtypes(include=[np.number])

#Se escalan los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(features)

# Define una función para formatear los valores de 'Value'
def format_value(value):
    return f'{value:.0f} €' if pd.notnull(value) else None

# Define una función para formatear los valores de habilidades
def format_skill(skill):
    return f'{skill:.0f}' if pd.notnull(skill) else None


# Configura el layout y los márgenes de la aplicación
st.set_page_config(layout="wide")

with st.container():
    # Agrega un título y una descripción a tu app
    st.title('Herramienta de búsqueda de jugadores similares')
    st.write('Introduce el nombre de un jugador para encontrar los más similares.')
    
    # Solicita al usuario que introduzca el nombre del jugador
    selected_player_name = st.text_input('Nombre del jugador')
    
# Botón para realizar la búsqueda
if st.button('Buscar jugadores similares'):
    # Proceso de búsqueda y visualización de resultados
    player_df = df[df['Name'] == selected_player_name].nlargest(1, 'Overall_2022')
    
    if player_df.empty:
        st.error("Escribió un nombre erróneo, revise su ortografía y recuerde ponerlo todo en minúsculas.")
    else:
        #Se calcula la similitud
        selected_player_id = player_df['ID'].iloc[0]
        player_index = df.index[df['ID'] == selected_player_id].tolist() 
        cosine_similarities = cosine_similarity(df_scaled, df_scaled[player_index])
                    
        #Se añade la similitud calculada al dataframe
        df['Similarity'] = cosine_similarities
                    
        #Se coge el top 5 más similar 
        top_5_similar = df[df['ID'] != selected_player_id].nlargest(5, 'Similarity')
                    
        #Se definen las variables que s emostraran de los jugadores
        additional_columns = ["ID", "Name","Age_2022", "Nationality", "Club_2022", "Value_2022",
                                        "Overall_2022", "Preferred Foot", "Contract Valid Until_2022",
                                        'Finishing_2022', 'ShortPassing_2022', 'Dribbling_2022',
                                        'Acceleration_2022', 'Stamina_2022', 'Interceptions_2022',
                                        'GKDiving_2022', 'GKReflexes_2022', 'Similarity']
                    
        # Añade la fila del jugador seleccionado al principio del DataFrame top_5_similar
        selected_player_row = df[df['ID'] == selected_player_id][additional_columns]
        selected_player_row['Similarity'] = 1.0  #La similitud de un jugador con él mismo es 1
        top_players = pd.concat([selected_player_row, top_5_similar[additional_columns]])
                    
                    
        #Se aplican los formatos
        top_players['Value_2022'] = top_players['Value_2022'].apply(format_value)
        skill_columns = ['Finishing_2022', 'ShortPassing_2022', 'Dribbling_2022', 
                                    'Acceleration_2022', 'Stamina_2022', 'Interceptions_2022', 
                                    'GKDiving_2022', 'GKReflexes_2022','Contract Valid Until_2022']
        for column in skill_columns:
            top_players[column] = top_players[column].apply(format_skill)
                    
        #Se elimina ID
        top_players_final = top_players.drop(columns='ID')
                    
        #Se renombra las columnas quitando el sufijo '_2022'
        rename_columns = {
                        "Club_2022": "Club",
                        "Age_2022": "Age",
                        "Value_2022": "Value",
                        "Overall_2022": "Overall",
                        "Contract Valid Until_2022": "Contract Valid Until",
                        "Finishing_2022": "Finishing",
                        "ShortPassing_2022": "ShortPassing",
                        "Dribbling_2022": "Dribbling",
                        "Acceleration_2022": "Acceleration",
                        "Stamina_2022": "Stamina",
                        "Interceptions_2022": "Interceptions",
                        "GKDiving_2022": "GKDiving",
                        "GKReflexes_2022": "GKReflexes"
        }
                    
        #Se aplica
        top_players_final = top_players_final.rename(columns=rename_columns)
    
        # Convertir el DataFrame a HTML sin el índice
        html = top_players_final.to_html(index=False)
            

        #Se resetea el indice para que no salga
        #top_players_final.reset_index(drop=True, inplace=True)
                  
        #Se muestra la tabla en Streamlit
        #st.table(top_players_final)


        # Mostrar el HTML en Streamlit
        st.markdown(html, unsafe_allow_html=True)

