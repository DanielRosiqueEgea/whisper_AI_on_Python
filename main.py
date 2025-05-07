import tkinter as tk
from tqdm import tqdm
from tkinter import filedialog
import ffmpeg
import os
import docker
import shutil
import sys
import whisper
import torch
from whisper.utils import get_writer, format_timestamp
import subprocess
import math  
import json
from core.utils import generate_audio_files, transcribe_chunks, run_whisper_docker, run_whisper_local, select_formats

device = "cuda" if torch.cuda.is_available() else "cpu"


# Paso 1: Selector de archivos (videos y audios)
root = tk.Tk()
root.withdraw()
input_paths = filedialog.askopenfilenames(
    title="Selecciona archivos de video o audio",
    filetypes=[("Multimedia", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.m4a")]
)
if not input_paths:
    print("No se seleccionaron archivos.")
    exit()

# Paso 2: Definir carpeta de salida de audios
output_audio_dir = "/tmp/audio_to_transcribe"
os.makedirs(output_audio_dir, exist_ok=True)

audio_files = generate_audio_files(input_paths, output_audio_dir)

use_docker = True
try:
    client = docker.from_env()
    container = client.containers.get("whisper_container")
    if container.status != "running":
        print("Contenedor Docker no iniciado. Utilizando Whisper en CPU.")
        use_docker = False
        raise Exception("Contenedor Docker no iniciado")

    mount_point = "/audios"  # Debe coincidir con el volumen del contenedor
except Exception as e:
    print(f"No se pudo conectar al contenedor Docker: {e}")
    use_docker = False
    if not torch.cuda.is_available():
        print("CUDA no disponible. Utilizando Whisper en CPU.")
    else:
        print("CUDA disponible. Utilizando Whisper en GPU.")
    model = whisper.load_model("medium", device=device,  in_memory=True)  # Puedes cambiar a 'medium' o 'large' si quieres



print("Seleccione los formatos de salida:")
available_formats = ["txt", "vtt", "srt", "tsv", "json"]
formats = select_formats(available_formats)
print("Formatos seleccionados:", formats)


transcribed_audios = {}

for audio in (audio_pbar := tqdm(audio_files, desc="Procesando audio", leave=False)): # for audio in audio_files:
    audio_name = os.path.basename(audio)
    audio_stem = os.path.splitext(audio_name)[0]
    audio_output_host_dir = os.path.join(output_audio_dir, audio_stem)
    os.makedirs(audio_output_host_dir, exist_ok=True)

    
    if use_docker:
        audio_pbar.set_description(f"Procesando {audio} con Docker...")
        audio_output_container_dir = os.path.join(mount_point, audio_stem) 
        audio_in_container = os.path.join(mount_point, audio_name)
        result = run_whisper_docker(container, audio_in_container, audio_output_container_dir)
        continue

    all_segments = []
    audio_pbar.set_description(f"Procesando audio: {audio_name} en Local")

    chunks = split_audio_fixed_chunks(audio, chunk_duration=60)
    
    transcript_path = os.path.join(audio_output_host_dir, f"{audio_stem}.txt")
    if os.path.exists(transcript_path):
        audio_pbar.set_description(f"Transcript ya existente: {transcript_path}")
        transcribed_audios.update({audio: audio_output_host_dir}) #Mapeado de audio a output_host_dir para la limpieza posterior
    
        continue
    
    transcribe_chunks(model, adjust_segments, audio_stem, audio_output_host_dir, all_segments, chunks)


    write_files(formats, audio_stem, audio_output_host_dir, all_segments)


    if os.path.exists(transcript_path):
        audio_pbar.set_description(f"Transcript generado: {transcript_path}")
        transcribed_audios.update({audio: audio_output_host_dir}) #Mapeado de audio a output_host_dir para la limpieza posterior
    else:
        audio_pbar.set_description(f"No se encontró transcript para {audio_name}")

#TODO: añadir sistema de traducción usando gettext o similar
#TODO: LIMPIAR ARCHIVOS TEMPORALES DESPUÉS de la EJECUCIÓN
#TODO: integrar con tkinter una interfaz gráfica
# for transcribed_audio in (audio_pbar := tqdm(transcribed_audios, desc="Limpiando archivos", leave=False)): # for audio in audio_files:
#     audio, output_dir = transcribed_audio
#     # Extraer nuevo directorio de audio
#     audio_dir = os.path.dirname(audio)
#     audio_name = os.path.basename(audio)
#     new_output_dir = os.path.join(audio_dir, audio_name)
#     os.mkdirs(new_output_dir, exist_ok=True)


#     # Limpiar archivos temporales
#     shutil.rmtree(output_dir)

#     audio_name = os.path.basename(audio)
#     audio_stem = os.path.splitext(audio_name)[0]
#     audio_output_host_dir = os.path.join(output_audio_dir, audio_stem)

# print("Proceso finalizado.")
