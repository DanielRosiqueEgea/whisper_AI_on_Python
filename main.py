import tkinter as tk
from tqdm import tqdm
from tkinter import filedialog , messagebox
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
from core.utils import (generate_audio_files, transcribe_chunks, run_whisper_docker,  
                        run_whisper_local, select_formats, write_files, 
                        adjust_segments, split_audio_fixed_chunks)

import getopt, sys

available_formats = ["txt", "vtt", "srt", "tsv", "json"]
available_models = whisper._MODELS.keys()
available_devices = ["cpu", "cuda"]
def process_command_line_arguments():
    argumentList = sys.argv[1:]

    options = "hi:d:m:o:f:"
    long_options = ["help", "input=", "device=", "model_size=", "output_formats=", "fraction="]

    # verbose = None
    input_path = None
    output_formats= None
    model_size = 'medium'
    device = 'cuda'
    fraction = 1

    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    
    print("Arguments: ", arguments)
    print("Values: ", values)   
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("Usage: python main.py [-h] [-i INPUT] [-o OUTPUT] [-c CLASSIFICATION_MODE] [-m MODEL_SIZE] [-d DEVICE] [-f FRACTION]")
            print("Options:")
            print("  -h, --help           Show this help message and exit")
            # TODO: ADD Verbose mode
            # print("  -v, --verbose        Enable verbose mode.")
            # print("                       Optional value: 0 = silent, >0 = verbose level.")
            # print("                       Silent mode logs all outputs to a text file: operation_log.txt")
            # print("                       If not provided, it will be asked")
            print("  -i, --input PATH     Specify the input directory (if not provided, it will be asked)")
            print("  -d, --device 'cuda' or 'cpu'    Specify the device to try to use (default: 'cuda')")

            print("  -m, --model_size SIZE    Specify the model_size (default: 'medium')")
            print(f"                       possible values: {available_models}")
            print("  -o, --output_formats ALLOWED_FORMATS    Specify the formats to be used (default: it will be asked)")
            print(f"                       possible values: {available_formats}")
            print("  -f, --fraction [value]    Fraction of the video to transcribe (default: 1")
            print(f"                       can be negative, if so, the transcription will 'start' from the end")
            sys.exit(0)

        # if currentArgument in ("-v", "--verbose"):
        #     verbose = currentValue
        if currentArgument in ("-i", "--input"):
            
            input_path = currentValue

        if currentArgument in ("-o", "--output_formats"):
            print(f"Output formats: {currentValue}")
            tmp_formats = currentValue.split(' ')
            print(f"Output formats: {tmp_formats}")
            tmp_formats = [format for format in tmp_formats if format in available_formats]
            if len(tmp_formats) != 0:
                output_formats = tmp_formats
            else:
                print(f"Output formats {currentValue} are not available. Available formats: {available_formats}")
                output_formats = None

        if currentArgument in ("-f", "--fraction"):
            fraction = currentValue
            fraction = float(fraction)
            print(f"Fraction: {fraction}")
            

        if currentArgument in ("-d", "--device"):
            if currentValue not in available_devices:
                print(f"Device {currentValue} is not available. Available devices: {available_devices}")
                device = 'cuda'

        if currentArgument in ("-m", "--model_size"):
            if currentValue not in available_models:
                print(f"Model size {currentValue} is not available. Available models: {available_models}")
                model_size = None
       
    return input_path, output_formats, model_size, device, fraction



# Paso 1: Selector de archivos (videos y audios)
input_formats = ["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"]
def ask_for_file_paths():
    root = tk.Tk()
    root.withdraw()
    input_paths = filedialog.askopenfilenames(
    title="Selecciona archivos de video o audio",
    filetypes=[("Multimedia", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.m4a")]
)
    if not input_paths:
        print("No se seleccionaron archivos.")
        exit()
    return input_paths


def get_file_paths(input_path):
    if not os.path.exists(input_path):
        print("La ruta especificada no existe.")
        return []

    if os.path.isfile(input_path):
        input_paths = [input_path]
    elif os.path.isdir(input_path):
        input_paths = [
            os.path.join(input_path, file)
            for file in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, file))
        ]
    else:
        print("La ruta no es válida.")
        return []

    # Filtrar solo archivos con extensiones válidas
    valid_paths = [path for path in input_paths if any(path.lower().endswith(ext) for ext in input_formats)]

    if not valid_paths:
        print("No se encontraron archivos de audio o video válidos en la ruta especificada.")

    return valid_paths


input_path, output_formats, model_size, device, fraction = process_command_line_arguments()


input_paths = []
if input_path:
    input_paths = get_file_paths(input_path)

if not input_paths:
    input_paths = ask_for_file_paths()

# Paso 2: Definir carpeta de salida de audios
output_tmp_dir = "/tmp/audio_to_transcribe"
os.makedirs(output_tmp_dir, exist_ok=True)

audio_map = generate_audio_files(input_paths, output_tmp_dir,fraction=fraction, verbose=False)
audio_files = list(audio_map.keys())


if device is None:
    device = "cuda"

if device == "cuda":
    if not torch.cuda.is_available():
        print("CUDA no disponible. Utilizando Whisper en CPU.")
        device = "cpu"
    else:
        print("CUDA disponible. Utilizando Whisper en GPU.")
        device = "cuda"

if model_size is None:
    model_size = "medium"
model = whisper.load_model(model_size, device=device,  in_memory=True)  



print("Seleccione los formatos de salida:")
if output_formats:
    formats = output_formats
else:
    formats = select_formats(available_formats)
print("Formatos seleccionados:", formats)


transcribed_audios = {}

for audio in (audio_pbar := tqdm(audio_files, desc="Procesando audio", leave=False)): # for audio in audio_files:
    audio_name = os.path.basename(audio)
    audio_stem = os.path.splitext(audio_name)[0]
    audio_tmp_host_dir = os.path.join(output_tmp_dir, audio_stem)
    audio_dict = audio_map[audio]
    original_path = audio_map[audio].get("original")
    audio_output_host_dir = os.path.join(os.path.dirname(original_path),audio_stem)

    os.makedirs(audio_tmp_host_dir, exist_ok=True)
    os.makedirs(audio_output_host_dir, exist_ok=True)

    

    all_segments = []
    audio_pbar.set_description(f"Procesando audio: {audio_name} en Local")

    if fraction != 1.0:
        pass
    else:
        chunks = split_audio_fixed_chunks(audio, duration= audio_dict.get("duration"), start= audio_dict.get("start"), chunk_duration=60)

    transcript_path = os.path.join(audio_output_host_dir, f"{audio_stem}.txt")
    if os.path.exists(transcript_path):
        audio_pbar.set_description(f"Transcript ya existente: {transcript_path}")
        transcribed_audios.update({audio: audio_output_host_dir}) #Mapeado de audio a output_host_dir para la limpieza posterior
    
        continue
    
    transcribe_chunks(model, adjust_segments, audio_stem, audio_tmp_host_dir, all_segments, chunks)


    write_files(formats, audio_stem, audio_output_host_dir, all_segments)


    if os.path.exists(transcript_path):
        audio_pbar.set_description(f"Transcript generado: {transcript_path}")
        transcribed_audios.update({audio: audio_output_host_dir}) #Mapeado de audio a output_host_dir para la limpieza posterior
    else:
        audio_pbar.set_description(f"No se encontró transcript para {audio_name}")
    
messagebox.showinfo("Proceso finalizado", "El proceso de transcripción ha finalizado.")
    


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


#TODO: ADD docker support
# use_docker = True
# try:
#     client = docker.from_env()
#     container = client.containers.get("whisper_container")
#     if container.status != "running":
#         print("Contenedor Docker no iniciado. Utilizando Whisper en CPU.")
#         use_docker = False
#         raise Exception("Contenedor Docker no iniciado")

#     mount_point = "/audios"  # Debe coincidir con el volumen del contenedor
# except Exception as e:
#     print(f"No se pudo conectar al contenedor Docker: {e}")
#     use_docker = False

   # if use_docker:
    #     audio_pbar.set_description(f"Procesando {audio} con Docker...")
    #     audio_output_container_dir = os.path.join(mount_point, audio_stem) 
    #     audio_in_container = os.path.join(mount_point, audio_name)
    #     result = run_whisper_docker(container, audio_in_container, audio_output_container_dir)
    #     continue