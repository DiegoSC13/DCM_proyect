import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def extract_frames(video_file, frame_nums, output_folder):
    '''
    Función para extraer frames de archivo .mp4 y guardarlos como .png
    INPUT: Video file, list with frames indexes, output folder
    OUTPUT: None
    '''
    # Abre el archivo de video
    cap = cv2.VideoCapture(video_file)

    # Verifica si el video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Crea la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Itera sobre los números de frames especificados
    for frame_num in frame_nums:
        # Establece el frame en el número especificado
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Lee el frame actual
        ret, frame = cap.read()
        
        # Verifica si se leyó correctamente
        if ret:
            # Convierte el frame a escala de grises
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Guarda el frame como imagen PNG en la carpeta de salida
            output_path = os.path.join(output_folder, f"frame_{frame_num}.png")
            cv2.imwrite(output_path, gray_frame)
            print(f"Frame {frame_num} guardado como {output_path}")
        else:
            print(f"No se pudo leer el frame {frame_num}")

    # Cierra el archivo de video
    cap.release()

    return

def subtract_frames(current_frame_path, reference_frame_path, output_path, clip=True):
    '''
    Función que lee dos imágenes, hace la diferencia, clippea el resultado para guardar en 8 bits y lo guarda en un path especificado
    INPUT: Video file, list with frames indexes, output folder
    OUTPUT: None
    '''
    # Lee las imágenes
    img1 = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    img2 = cv2.imread(reference_frame_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    # Verifica si las imágenes se leyeron correctamente
    if img1 is None or img2 is None:
        print("Error al leer las imágenes.")
        return

    # Realiza la resta de las imágenes
    diff_img = img1 - img2

    #Verifico el rango dinámico de la diferencia
    print('Valor mínimo de diferencia: ', np.min(diff_img))
    print('Valor máximo de diferencia: ', np.max(diff_img))

    if clip == True:
        # Clippeo entre -128 y 127, pierdo información de valores de diferencia altos
        diff_img_adjusted = np.clip(diff_img, -128, 127).astype(np.uint8)

    else:
        # Me quedo con el valor absoluto de cada valor, pierdo información de signo pero no de diferencia (misma energía en imagen residual)
        diff_img_adjusted = np.abs(diff_img)

    # Guarda la imagen resultante como PNG
    cv2.imwrite(output_path, diff_img_adjusted)
    print(f"Resultado de la resta ajustado y guardado como {output_path}")

    return

def optical_flow(current_frame_path, reference_frame_path, output_path):
    # Cargar las dos imágenes de entrada
    curr_frame = cv2.imread(current_frame_path)
    prev_frame = cv2.imread(reference_frame_path)

    # Convertir las imágenes a escala de grises
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Configurar parámetros para el cálculo del flujo óptico
    flow_parameters = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Calcular el flujo óptico usando el método de Lucas-Kanade
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_parameters)

    # Visualizar el flujo óptico sobre la imagen actual
    flow_visualization = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
    step = 10  # Espaciado para mostrar los vectores de flujo
    for y in range(0, flow_visualization.shape[0], step):
        for x in range(0, flow_visualization.shape[1], step):
            dx, dy = flow[y, x]
            cv2.arrowedLine(flow_visualization, (x, y), (int(x + dx), int(y + dy)), (255, 0, 0), 1)

    # Guardar el flujo óptico como una imagen
    print(flow.shape)
    #flow_image = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
    np.save(os.path.join(output_path,'flow_x.npy'), flow[..., 0])
    np.save(os.path.join(output_path,'flow_y.npy'), flow[..., 1])
    cv2.imwrite(os.path.join(output_path,'optical_flow_visualization.png'), flow_visualization)
    # np.save('/Users/diegosilveracoeff/Desktop/Fing/DCM/flow_x.npy', flow[..., 0])
    # np.save('/Users/diegosilveracoeff/Desktop/Fing/DCM/flow_y.npy', flow[..., 1])
    # cv2.imwrite('/Users/diegosilveracoeff/Desktop/Fing/DCM/optical_flow_visualization.png', flow_visualization)

    #TODO: Estudiar cuál es el problema de esta visualización
    # Mostrar la imagen con el flujo óptico
    plt.imshow(cv2.cvtColor(flow_visualization, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Ocultar ejes
    plt.show()

    return