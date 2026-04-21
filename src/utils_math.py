
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def elbow_method(x_data):
    # Determinamos el número óptimo de clusters (Método del Codo)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x_data)
        wcss.append([i, kmeans.inertia_])

    wcss_df = pd.DataFrame(wcss, columns=['n_cluster', 'wcss'])

    sns.lineplot(data=wcss_df, x='n_cluster', y='wcss', marker='o')

    # Determinacion de punto de cambio de pendiente (codo)
    wcss_df['dwcss'] = wcss_df['wcss'].diff()
    wcss_df['dn_cluster'] = wcss_df['n_cluster'].diff()
    wcss_df['slope'] = wcss_df['dwcss']/wcss_df['dn_cluster']

    wcss_df['d_slope'] = wcss_df['slope'].diff()

    elbow_index = wcss_df['d_slope'].idxmax()
    elbow_point = wcss_df.loc[elbow_index]

    print(wcss_df)

    sns.scatterplot(x=[elbow_point['n_cluster']], y=[elbow_point['wcss']], color='red', s=80, zorder=5)

    for x, y in zip(wcss_df["n_cluster"], wcss_df["wcss"]):
        plt.annotate(f'{x, int(y)}', (x, y), textcoords="offset points", xytext=(15,10), ha='center', fontsize=9)

    plt.title('Método del Codo para detectar Clusters')
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS (Inercia)')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig('./graphs/training/codo_metodo_nclusters.png')
    plt.show()


def run_silhouette_score(data: pd.DataFrame, n_clusters: int, n_rows: int = 3, n_cols: int = 3) -> Tuple[np.ndarray[Any, dtype[float64]], int]:

    silhouette_data: np.ndarray[Any, np.dtype[np.float64]] = np.zeros([n_clusters, 2])

    i_row: int = 0
    i_col: int = 0

    fig, ax_1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))

    for n_clusters_opt in range(2, n_clusters+2):
        # Crear el subplot
        ax_1[i_row, i_col].set_xlim([-0.1, 1])
        
        # Entrenar el modelo
        clusterer: KMeans = KMeans(n_clusters=n_clusters_opt, random_state=10)
        cluster_labels: np.ndarray[Any, Any] = clusterer.fit_predict(data)

        # Calcular el promedio de silueta (la línea punteada)
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"Para n_clusters = {n_clusters_opt}, el promedio es: {silhouette_avg:.3f}")

        # Calcular el coeficiente de cada muestra
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters_opt):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters_opt)
            ax_1[i_row, i_col].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            y_lower = y_upper + 10 

        ax_1[i_row, i_col].axvline(x=silhouette_avg, color="red", linestyle="--")
        ax_1[i_row, i_col].set_title(f"Análisis de silueta para k = {n_clusters_opt}, cs={np.round(silhouette_avg,4)}")
        ax_1[i_row, i_col].set_xlabel('Coeficiente silueta')

        silhouette_data[n_clusters_opt-2, 0] = n_clusters_opt
        silhouette_data[n_clusters_opt-2, 1] = silhouette_avg

        # Actualizando contadores
        if i_col < n_cols-1:
            i_col += 1
        else:
            i_col = 0
            if i_row < n_rows-1:
                i_row += 1
            else:
                i_row = 0

    plt.tight_layout()
    plt.savefig(f'./graphs/training/graf_silueta.png')
    plt.show()

    silhouette_df = pd.DataFrame(silhouette_data, columns=['n_cluster', 'cs'])

    sns.lineplot(data=silhouette_df, x='n_cluster', y='cs', marker='o')
    sns.scatterplot(x=[4], y=[0.4034], color='red', s=80, zorder=5)

    for x, y in zip(silhouette_df["n_cluster"], silhouette_df["cs"]):
        plt.annotate(f'{x, np.round(y, 4)}', (x, y), textcoords="offset points", xytext=(15,10), ha='center', fontsize=9)

    plt.title('Análisis de coeficiente de silueta')
    plt.xlabel('n_clusters')
    plt.ylabel('Coef. silueta promedio')
    plt.grid(linestyle='--', alpha=0.5)
    # plt.ylim([0.35, 0.415])
    # plt.xlim([1, 11])
    plt.savefig(f'./graphs/training/coef_silueta_line.png')

    n_clusters_opt = int(silhouette_data[np.argmax(silhouette_data[:, 1]), 0])
    print(f'El n_cluster optimo {n_clusters_opt}')

    return silhouette_data, n_clusters_opt



def empty_directory(directory_path):
    path = Path(directory_path)
    if path.is_dir():
        return not any(path.iterdir())
    else:
        raise ValueError("This path is not a directory")

def graph_dist_clusters(data, column_names, pca_n_cols, pca_n_rows, column_analysed, output_path):

    # Verificando que el directorio para generar las graficas este vacios

    directory_path = Path(output_path).parent
    
    try:
        if empty_directory(directory_path):
            print("Directory is empty. ok")
            pca_i_row = 0
            pca_i_col = 0

            for pca_column_name_1 in column_names:

                fig_pca, ax_pca = plt.subplots(pca_n_rows, pca_n_cols, figsize=(5*pca_n_cols, 3*pca_n_rows))
                
                for pca_column_name_2 in column_names:
                    
                    sns.scatterplot(
                        data=data, 
                        x=pca_column_name_1,
                        y=pca_column_name_2,
                        hue=column_analysed,
                        palette='viridis',
                        s=15,
                        alpha=1,
                        edgecolor='w',
                        ax=ax_pca[pca_i_row, pca_i_col]
                    )

                    # Personalización de títulos (basado en tu interpretación de loadings
                    ax_pca[pca_i_row, pca_i_col].set_title(f'{pca_column_name_1} vs {pca_column_name_2}')
                    ax_pca[pca_i_row, pca_i_col].set_xlabel(f'{pca_column_name_1}')
                    ax_pca[pca_i_row, pca_i_col].set_ylabel(f'{pca_column_name_2}')
                    ax_pca[pca_i_row, pca_i_col].grid(True, linestyle='--', alpha=0.5)

                    # Actualizacón de contadores
                    if pca_i_row < pca_n_rows-1:
                        pca_i_row += 1
                    else:
                        pca_i_row = 0
                    
                    if pca_i_col < pca_n_cols-1:
                        pca_i_col += 1
                    else:
                        pca_i_col = 0
                
                plt.tight_layout()
                plt.savefig(output_path+f'{pca_column_name_1}.png')
                plt.show()
        else:
            print(f"This directory {directory_path} is not empty. Choose another one.")
    except ValueError as e:
        print(e)