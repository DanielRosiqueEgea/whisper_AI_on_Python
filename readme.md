# Transcriptor de Audio/Video con Whisper y Docker

Este proyecto permite seleccionar archivos de video o audio, extraer el audio, dividirlo en fragmentos, transcribirlo usando [OpenAI Whisper](https://github.com/openai/whisper), y guardar los resultados en múltiples formatos (`txt`, `srt`, `vtt`, etc.). Puede ejecutarse en local (CPU/GPU) o usando un contenedor Docker para acelerar el proceso.

---

## Requisitos

- Python 3.8+
- [Docker](https://www.docker.com/) (opcional pero recomendado)
- GPU con CUDA (opcional, para acelerar transcripción con Whisper)
- `ffmpeg` instalado y en el `PATH`

### Dependencias Python

Instálalas con:

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecuta el script principal:

    ```
    python main.py
    ```

2. Selecciona archivos de audio o video (.mp4, .mp3, .wav, etc.) usando el explorador de archivos.

3. Selecciona los formatos de salida deseados (txt, vtt, srt, etc.).

4. Uso de docker

    __*No está correctamenete testeado el uso de docker aún*__   

    >El sistema usará Docker si detecta un contenedor llamado whisper_container. Si no, utilizará Whisper localmente.

## Estructura de carpetas

Los archivos transcritos se almacenan por defecto en:

```
/tmp/audio_to_transcribe/
    └── <nombre_del_archivo>/
        ├── audio_chunk_0.json
        ├── <archivo>.txt
        └── ...
```

## Funcionalidades

    ✅ Interfaz gráfica para seleccionar archivos

    ✅ Soporte de múltiples formatos de salida

    ✅ Transcripción por fragmentos (para archivos largos)

    ✅ Detección automática de GPU o contenedor Docker

    ✅ Guardado incremental de fragmentos en .json

    🚧 Soporte multilenguaje (gettext) [pendiente]

    🚧 Limpieza automática de temporales [pendiente]
    
    🚧 Interfaz totalmente grafica [pendiente]


## Tests

Puedes encontrar funciones auxiliares y sus tests en el módulo utils.py. 

Aún faltan tests por generar y código que limpiar. 


# TODOs

- [ ] Soporte multilenguaje con gettext

- [ ] Limpieza automática de archivos temporales

- [ ] Interfaz gráfica más amigable (PyQT o PySide6)

- [ ] Añadir más tests