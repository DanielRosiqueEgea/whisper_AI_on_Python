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
import hashlib
import math  
import json

# Extensiones de audio aceptadas directamente
def generate_audio_files(input_paths, output_audio_dir, fraction = 1.0, verbose = False):
    #TODO: Añadir tqdm
    audio_exts = [".mp3"]
    audio_exts_not_supported = [ ".wav", ".m4a"]
    audio_map = {}  # audio convertido o directo → original
    # Paso 3: Convertir solo los videos
    
    for path in input_paths:
        ext = os.path.splitext(path)[1].lower()
        base = os.path.splitext(os.path.basename(path))[0]

        probe = ffmpeg.probe(path)
        duration = float(probe['format']['duration'])

        if fraction >= 0.0:
            start_time = 0
            new_duration = duration * fraction
        else:
            new_duration = duration * abs(fraction)
            start_time = duration - new_duration


        if ext in audio_exts:
        # Es un archivo de audio, usar directamente
            print(f"Detectado audio: {path=}")

         
            audio_map[path] = {
                "original": path,
                "start": start_time,
                "duration": new_duration
            }
            continue
        
        # Es un video, convertirlo a .mp3ç
        path_hash = hashlib.sha1(path.encode()).hexdigest()[:8]
        safe_name = f"{base}_{path_hash}.mp3"
        audio_path = os.path.join(output_audio_dir, safe_name)
        print(f"Convirtiendo video a audio: {path} ->  {audio_path}")
        
        if os.path.exists(audio_path):
            print(f"El archivo convertido {audio_path} ya existe. Saltando...")
            probe = ffmpeg.probe(audio_path)
            converted_duration = float(probe['format']['duration'])
            if abs(converted_duration - new_duration) > 0.1:
                print(f"El archivo convertido {audio_path} tiene una duración diferente a la esperada. Saltando...")
                
                pass
            audio_map[audio_path] = {
                "original": path,
                "start": start_time,
                "duration": duration
            }
            continue

        stream = ffmpeg.input(path).output(
            audio_path,
             ss=start_time, t=new_duration,
              acodec='libmp3lame'
            
        )
        if verbose:
            stream.run(overwrite_output=True)
        else:
            stream.run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)

    
        audio_map[audio_path] = {
            "original": path,
            "start": start_time,
            "duration": new_duration
        }
    return audio_map


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



def split_audio_fixed_chunks(audio_path, chunk_duration=60, duration = None,sift = 0, fraction =1.0, verbose = False):

    if duration is None:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
    if start is None:
        start = 0

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = os.path.join("/tmp", base_name + "_chunks")
    os.makedirs(out_dir, exist_ok=True)

    chunk_paths = []
    
    total_chunks = math.ceil(duration / chunk_duration)
    
    for i in tqdm(range(total_chunks), desc="Dividiendo audio en chunks", leave=False):
        start = i * chunk_duration
        offset = start + sift
        output_chunk = os.path.join(out_dir, f"{base_name}_chunk_{i}.mp3")

        if os.path.exists(output_chunk): 
            chunk_paths.append((output_chunk, offset))  # Se salta el chunk y se guarda el offset
            continue
        try:
            ffmpeg.input(audio_path, ss=start, t=chunk_duration) \
                .output(output_chunk, acodec="copy") \
                .run(overwrite_output=True, quiet=True,capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print(f"Error al dividir el audio: {e}")
            print(f'stdout: {e.stdout.decode("utf-8")}')
            print(f'stderr: {e.stderr.decode("utf-8")}')
            raise e
        chunk_paths.append((output_chunk, offset))  # Guardar también el offset
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


def write_files(formats, audio_stem, audio_output_host_dir, all_segments):
    combined_text = "".join(s["text"] for s in all_segments)
    for fmt in formats:
        output_file = os.path.join(audio_output_host_dir, f"{audio_stem}.{fmt}")
        writer = get_writer(fmt, audio_output_host_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            writer.write_result({"segments": all_segments, "text": combined_text}, f)

def transcribe_chunks(model, adjust_segments, audio_stem, audio_output_host_dir, all_segments, chunks):


    for chunk_path, offset in (chunk_pbar := tqdm(chunks, desc="Procesando fragmento", leave=False)):
        chunk_pbar.set_description(f"Procesando fragmento desde {offset}s: {chunk_path}")
        
        json_path = os.path.join(audio_output_host_dir, f"{audio_stem}_chunk_{offset}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                segments = json.load(jf)
            all_segments.extend(segments)
            continue
        try:
            result = model.transcribe(chunk_path, language="Spanish", verbose=None)
        except Exception as e:
            print(f"Error al transcribir el audio: {e}")
            continue
        # if result['temperature'] < 0.4:
        #     continue
        segments = result["segments"]
        segments = adjust_segments(segments, offset)
        all_segments.extend(segments)

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(segments, jf, ensure_ascii=False, indent=2)



# def test_methods():

#     window = tk.Tk()
#     window.title("Test methods")
#     window.geometry("300x200")

#     label = tk.Label(window, text="Test methods")
#     label.pack(pady=5)

#     available_functions = {}
#     for name in dir('.'):
#         elt = getattr('.', name)
#         if type(elt) == types.FunctionType:
#             print(elt.__name__)
#             available_functions[elt.__name__] = elt

#     functions_list = tk.Listbox(window, selectmode="multiple", height=len(available_functions))
    

# if __name__ == "__main__":
#     test_methods()