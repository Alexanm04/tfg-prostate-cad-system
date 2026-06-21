# TFG: Desarrollo de un sistema inteligente para la clasificación de cánceres de próstata basado en imágenes

## Estructura del Repositorio

Debido a la arquitectura modular desacoplada, el repositorio se organiza en dos directorios principales:

* **`/backend`**: Scripts realizados en Python. Se divide en dos subdirectorios:
  * `/investigacion`: Código de la fase experimental (partición de datos, entrenamiento de modelos y análisis de resultados).
  * `/sistema_cad`: Implementación del sistema CAD (pipeline de procesamiento *end-to-end* desde la carga de la imagen WSI hasta la generación de inferencias).
* **`/frontend`**: Código correspondiente a la extensión desarrollada en Java para QuPath.

## 1. Requisitos Previos y Herramientas

Para que todo funcione correctamente, necesita instalar el siguiente software:
* **Python 3.9** o superior.
* **QuPath v0.7.0**. Recomendada esta versión para garantizar la compatibilidad de la extensión.

### Librerías de Python
Se recomienda utilizar un entorno virtual (conda). Las librerías principales requeridas son:
* `torch` y `torchvision` (recomendado con soporte de CUDA para aceleración por GPU)
* `tensorflow` y `keras`
* `timm` y `transformers` (Hugging Face)
* `trident` (para lectura WSI y segmentación HEST)
* `scikit-learn`, `pandas`, `numpy`, `Pillow`, `joblib`


## 2. Configuración y ejecución del Backend

### Configuración del Token de Hugging Face
El sistema utiliza modelos fundacionales de acceso restringido (como CONCH). Antes de extraer características, es obligatorio disponer de un token de Hugging Face con acceso a dichos modelos.

Abra el archivo `extrae_caracteristicas_wsi.py`, situado en `/backend/sistema_cad`, y sustituya el valor de la variable `HF_TOKEN` por su token personal de Hugging Face.

### Flujo de ejecución
1. Ejecute el script de segmentación y extracción de características (`extrae_caracteristicas_wsi.py`), situado en `/backend/sistema_cad`. Este genera los archivos `.npy` con los embeddings de cada parche.
2. Ejecute el script de inferencia (`predice_wsi.py`), también situado en `/backend/sistema_cad`. Este utiliza los embeddings del paso 1 y genera el archivo `.csv` con las coordenadas y probabilidades de predicción que posteriormente será empleado por la extensión de QuPath.


## 3. Instrucciones de uso de la extensión QuPath

### Instalación de la extensión
1. Abra QuPath.
2. Arrastre el archivo `.jar` que está ubicado en la carpeta `/frontend` y suéltelo sobre la ventana principal de QuPath.

### Preparación de los datos
1. Asegúrese de tener disponibles los archivos `.csv` generados por el módulo *backend* tras la inferencia. Si aún no los tiene, ejecute los scripts del módulo `/backend/sistema_cad` sobre sus imágenes histológicas.
2. Abra su proyecto de QuPath (`File -> Project -> Open Project...`). Si no dispone de un proyecto, créelo primero (`File -> Project -> Create Project...`) y añada las imágenes WSI que desee analizar (`File -> Project -> Add images...`).
3. Seleccione, dentro del proyecto, la imagen WSI que desea analizar.

### Configuración de rutas en QuPath
1. Seleccione el botón 📁 en la barra superior y añada la ruta donde alojó la carpeta con los archivos `.csv`. Posteriormente, pulse `OK`.

### Generación de predicciones
1. Utilice los botones 🤖, ⧉ y 📊 para ajustar adecuadamente la configuración de la ejecución (Modelo, Tipo de clasificación y Estrategia de solapamiento).
2. El sistema procesará los datos en segundo plano y dibujará el mapa de calor sobre la imagen automáticamente.
3. Utilice el botón de visibilidad (⛔ / 👁) para ocultar o visualizar los objetos añadidos.