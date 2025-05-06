# # Simulación de Robot en el Entorno Aloha con LeRobot

Este repositorio contiene un script de Python que demuestra cómo simular un robot en el entorno Aloha (`gym_aloha/AlohaTransferCube-v0`) utilizando el framework [LeRobot](https://github.com/huggingface/lerobot).

El script carga un modelo de política ACT preentrenado, ejecuta episodios de simulación, evalúa el rendimiento (tasa de éxito, recompensas) y guarda un video del episodio. También incluye un pequeño experimento para evaluar la robustez del modelo ante diferentes condiciones iniciales.

## Características Principales

*   Simulación de robot en el entorno Aloha.
*   Uso del framework LeRobot para cargar políticas y ejecutar simulaciones.
*   Carga de un modelo ACT preentrenado (`lerobot/act_aloha_sim_transfer_cube_human`).
*   Ejecución de episodios individuales y múltiples.
*   Recopilación y visualización de métricas de rendimiento (recompensa, tasa de éxito).
*   Guardado de video de la simulación.

## Instalación

1.  **Clona este repositorio:**
    ```bash
    git clone <URL_DE_TU_REPOSITORIO>
    cd <NOMBRE_DE_TU_REPOSITORIO>
    ```

2.  **Instala LeRobot y otras dependencias:**
    Sigue los pasos del script (se recomienda usar un entorno Conda):
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    conda create -y -n lerobot python=3.10
    conda activate lerobot
    pip install -e ".[aloha]"
    # Vuelve al directorio del script
    cd ..
    pip install matplotlib imageio
    ```
    *Nota: Asegúrate de estar en el entorno `lerobot` activado cuando instales `matplotlib` e `imageio` si no lo hiciste dentro del directorio `lerobot`.*

## Uso

Una vez que las dependencias estén instaladas y el entorno Conda (`lerobot`) esté activado:

1.  **Ejecuta el script:**
    ```bash
    python nombre_del_script.py
    ```
    (Reemplaza `nombre_del_script.py` con el nombre real de tu archivo Python, por ejemplo, `simulate_aloha.py`)

2.  **Resultados:**
    *   Se imprimirán mensajes en la consola mostrando el progreso y los resultados.
    *   Se generará un video del episodio en la carpeta `outputs/aloha_simulation/episodio_aloha.mp4`.
    *   Se mostrarán gráficos de Matplotlib con la evolución de la recompensa y los resultados del experimento.

## Ejemplo de Salida

El script mostrará información como:
*   Dispositivo utilizado (CUDA/MPS/CPU).
*   Compatibilidad entre el modelo y el entorno.
*   Resultados del episodio (pasos, recompensa, éxito).
*   Resultados del experimento (tasa de éxito, recompensa promedio).

Se crearán gráficos visualizando las recompensas y un archivo de video `episodio_aloha.mp4`.
