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

# Extensiones de audio aceptadas directamente
audio_exts = [".mp3", ".wav", ".m4a"]

# Paso 3: Convertir solo los videos
audio_files = []
for path in input_paths:
    ext = os.path.splitext(path)[1].lower()
    base = os.path.splitext(os.path.basename(path))[0]

    if ext in audio_exts:
        # Es un archivo de audio, usar directamente
        print(f"Detectado audio: {path}")
        audio_files.append(path)
    else:
        # Es un video, convertirlo a .mp3
        audio_path = os.path.join(output_audio_dir, f"{base}.mp3")
        print(f"Convirtiendo video a audio: {path} → {audio_path}")
        if os.path.exists(audio_path):
            print(f"El archivo {audio_path} ya existe. Saltando...")
            audio_files.append(audio_path)
            continue
        ffmpeg.input(path).output(audio_path, acodec='libmp3lame').run(overwrite_output=True)
        audio_files.append(audio_path)

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


def run_whisper_docker(container, audio_path, output_dir, language="Spanish"):
    command_gpu = f"whisper '{audio_path}' --language {language} --output_dir '{output_dir}'"
    command_cpu = f"{command_gpu} --device cpu"
    try:
        print(f"[Whisper-Docker] Procesando {audio_path} con GPU...")
        result = container.exec_run(command_gpu, stream=True, stdout=True, stderr=True)
        output = ""
        for line in result.output:
            output += line.decode()
        if "CUDA error" in output or "RuntimeError" in output:
            raise RuntimeError("Fallo con GPU")
        return output
    except Exception as e:
        print(f"[Whisper-Docker] Error con GPU: {e}")
        print(f"[Whisper-Docker] Reintentando con CPU...")
        result = container.exec_run(command_cpu, stream=True, stdout=True, stderr=True)
        output = ""
        for line in result.output:
            output += line.decode()
        return output
    
# Función para ejecución local con Whisper en Python
def run_whisper_local(model, audio_path, output_dir, formats = None):
    print(f"[Whisper-Local] Procesando {audio_path}...")
    result = model.transcribe(audio_path, language="Spanish", verbose=False)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(audio_path))[0] 
    # output_file = os.path.join(output_dir,base_filename + ".txt")
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write(result["text"])

    # Generar distintos formatos
    if isinstance(formats, list):
        for format in formats:
            writer = get_writer(format, output_dir)
            with open(os.path.join(output_dir, f"{base_filename}.{format}"), "w", encoding="utf-8") as f:
                writer.write_result(result, file=f)
    elif formats is not None:
        writer = get_writer(formats, output_dir)
        with open(os.path.join(output_dir, f"{base_filename}.{formats}"), "w", encoding="utf-8") as f:
            writer.write_result(result, file=f)

    return result["text"]



def split_audio_fixed_chunks(audio_path, chunk_duration=60):
    probe = ffmpeg.probe(audio_path)
    duration = float(probe['format']['duration'])
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = os.path.join("/tmp", base_name + "_chunks")
    os.makedirs(out_dir, exist_ok=True)

    chunk_paths = []
    
    total_chunks = math.ceil(duration / chunk_duration)

    for i in tqdm(range(total_chunks), desc="Dividiendo audio en chunks", leave=False):
        start = i * chunk_duration
        output_chunk = os.path.join(out_dir, f"{base_name}_chunk_{i}.mp3")

        if os.path.exists(output_chunk): 
            chunk_paths.append((output_chunk, start))  # Se salta el chunk y se guarda el offset
            continue

        ffmpeg.input(audio_path, ss=start, t=chunk_duration) \
            .output(output_chunk, acodec="copy") \
            .run(overwrite_output=True, quiet=True)
        
        chunk_paths.append((output_chunk, start))  # Guardar también el offset
    return chunk_paths

def adjust_segments(segments, offset):
    for segment in segments:
        segment["start"] += offset
        segment["end"] += offset
    return segments


def select_formats(available_formats):
    window = tk.Tk()
    window.title("Seleccione los formatos de salida")
    window.geometry("300x200")

    label = tk.Label(window, text="Seleccione uno o más formatos:")
    label.pack(pady=5)

    formats_list = tk.Listbox(window, selectmode="multiple", height=len(available_formats))
    for fmt in available_formats:
        formats_list.insert(tk.END, fmt)
    formats_list.pack(padx=10, pady=5)

    selected_formats = []

    def select():
        for i in formats_list.curselection():
            selected_formats.append(formats_list.get(i))
        window.quit()
        window.destroy()

    button = tk.Button(window, text="Seleccionar", command=select)
    button.pack(pady=10)

    window.mainloop()
    return selected_formats


print("Seleccione los formatos de salida:")
available_formats = ["txt", "vtt", "srt", "tsv", "json"]
formats = select_formats(available_formats)
print("Formatos seleccionados:", formats)




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
        continue
    
    for chunk_path, offset in (chunk_pbar := tqdm(chunks, desc="Procesando fragmento", leave=False)):
        chunk_pbar.set_description(f"Procesando fragmento desde {offset}s: {chunk_path}")
        
        json_path = os.path.join(audio_output_host_dir, f"{audio_stem}_chunk_{offset}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                segments = json.load(jf)
            all_segments.extend(segments)
            continue


        result = model.transcribe(chunk_path, language="Spanish", verbose=None)
        segments = result["segments"]
        segments = adjust_segments(segments, offset)
        all_segments.extend(segments)

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(segments, jf, ensure_ascii=False, indent=2)


    combined_text = "".join(s["text"] for s in all_segments)
    for fmt in formats:
        output_file = os.path.join(audio_output_host_dir, f"{audio_stem}.{fmt}")
        writer = get_writer(fmt, audio_output_host_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            writer.write_result({"segments": all_segments, "text": combined_text}, f)


    if os.path.exists(transcript_path):
        audio_pbar.set_description(f"Transcript generado: {transcript_path}")
    else:
        audio_pbar.set_description(f"No se encontró transcript para {audio_name}")

print("Proceso finalizado.")
