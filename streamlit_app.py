import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Funktion zum Laden der Excel-Datei aus dem lokalen Verzeichnis
def load_excel_file(filename):
    try:
        file_path = os.path.join("data", filename)
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Fehler beim Lesen der Excel-Datei {filename}: {e}")
    return None

def process_datetime(df, date_column):
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        return df
    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung des Datums: {e}")
        return None

def merge_dataframes(df1, df2, date_column):
    merged_df = pd.merge(df1, df2, on=date_column, how='outer')
    merged_df = merged_df.sort_values(by=date_column)
    return merged_df

def analyze_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    return corr_matrix, pca_result, pca.explained_variance_ratio_

def main():
    st.title("Qualitätsdatenanalyse für Lackieranlage")
    
    # Laden der Dateien aus dem lokalen Verzeichnis
    env_file = "DecTod_Hum.csv"
    quality_file = "PP_53_Jan_Feb.xlsx"
    
    env_df = load_excel_file(env_file)
    quality_df = load_excel_file(quality_file)
    
    if env_df is not None and quality_df is not None:
        st.success("Beide Dateien erfolgreich geladen!")
        
        # Datumsverarbeitung
        date_column = st.selectbox("Wählen Sie die Datums-/Zeitspalte", env_df.columns)
        env_df = process_datetime(env_df, date_column)
        quality_df = process_datetime(quality_df, date_column)
        
        if env_df is not None and quality_df is not None:
            # Zusammenführen der Dataframes
            merged_df = merge_dataframes(env_df, quality_df, date_column)
            
            st.subheader("Zusammengeführte Datenübersicht")
            st.write(merged_df.head())
            st.write(merged_df.describe())
            
            # Datenanalyse
            corr_matrix, pca_result, explained_variance = analyze_data(merged_df)
            
            # Korrelationsmatrix visualisieren
            st.subheader("Korrelationsmatrix")
            fig_corr = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr)
            
            # PCA-Ergebnisse visualisieren
            st.subheader("PCA-Analyse")
            fig_pca = px.scatter(x=pca_result[:,0], y=pca_result[:,1],
                                 labels={'x': 'Erste Hauptkomponente', 'y': 'Zweite Hauptkomponente'},
                                 title="PCA der zusammengeführten Daten")
            st.plotly_chart(fig_pca)
            
            st.write("Erklärte Varianz der ersten beiden Hauptkomponenten:", 
                     sum(explained_variance[:2]))
            
            # Zeitreihenvisualisierung
            st.subheader("Zeitreihenanalyse")
            numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
            selected_column = st.selectbox("Wählen Sie eine Spalte für die Zeitreihenanalyse", numeric_columns)
            
            fig_time = px.line(merged_df, x=date_column, y=selected_column, 
                               title=f'Zeitreihe: {selected_column}')
            st.plotly_chart(fig_time)
            
            # Streudiagramm
            st.subheader("Streudiagramm")
            x_axis = st.selectbox("Wählen Sie die X-Achse", numeric_columns, key='x_axis')
            y_axis = st.selectbox("Wählen Sie die Y-Achse", numeric_columns, key='y_axis')
            
            fig_scatter = px.scatter(merged_df, x=x_axis, y=y_axis, 
                                     title=f'Streudiagramm: {x_axis} vs {y_axis}')
            st.plotly_chart(fig_scatter)

if __name__ == "__main__":
    main()