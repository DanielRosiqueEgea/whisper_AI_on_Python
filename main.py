import tkinter as tk
from tkinter import filedialog
import ffmpeg
import os
import docker
import shutil
import sys
import whisper
import torch
from whisper.utils import WriteSRT, WriteVTT, get_writer
  
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

# Paso 4: Ejecutar whisper para cada audio
def select_formats(available_formats):
    window = tk.Tk()
    window.title("Seleccione los formatos de salida")
    window.geometry("300x100")

    selected_formats = tk.StringVar()
    selected_formats.set("txt")  # seleccionar txt por defecto

    formats = tk.StringVar()
    formats.set(" ".join(available_formats))
    formats_list = tk.Listbox(window, listvariable=formats, selectmode="multiple")
    formats_list.pack(padx=10, pady=10)

    def select():
        selected = [formats_list.get(i) for i in formats_list.curselection()]
        selected_formats.set(" ".join(selected))
        window.destroy()

    button = tk.Button(window, text="Seleccionar", command=select)
    button.pack()

    window.mainloop()

    return selected_formats.get().split()

available_formats = ["txt", "vtt", "srt", "tsv", "json"]
formats = select_formats(available_formats)




for audio in audio_files:
    audio_name = os.path.basename(audio)
    audio_stem = os.path.splitext(audio_name)[0]
    audio_output_host_dir = os.path.join(output_audio_dir, audio_stem)

    if use_docker:
        audio_output_container_dir = os.path.join(mount_point, audio_stem) 
        audio_in_container = os.path.join(mount_point, audio_name)
        result = run_whisper_docker(container, audio_in_container, audio_output_container_dir)
    else:
        result = run_whisper_local(model, audio, audio_output_host_dir)

    transcript_path = os.path.join(audio_output_host_dir, f"{audio_stem}.txt")
   
    if os.path.exists(transcript_path):
        print(f"Transcript generado: {transcript_path}")
    else:
        print(f"No se encontró transcript para {audio_name}")

print("Proceso finalizado.")