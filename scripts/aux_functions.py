import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dctn, idctn
import huffman

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
        diff_img_adjusted = ((diff_img - (-128)) * (255 / (127 - (-128)))).clip(0, 255).astype(np.uint8)

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
            cv2.arrowedLine(flow_visualization, (x, y), (int(np.trunc(x + dx)), int(np.trunc(y + dy + 1))), (255, 0, 0), 1)

    # Guardar el flujo óptico como una imagen
    print(flow.shape)
    #flow_image = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
    np.save(os.path.join(output_path,'flow_x.npy'), flow[..., 0])
    print(f"flow_x guardado como {os.path.join(output_path,'flow_x.npy')}")
    np.save(os.path.join(output_path,'flow_y.npy'), flow[..., 1])
    print(f"flow_y guardado como {os.path.join(output_path,'flow_y.npy')}")
    cv2.imwrite(os.path.join(output_path,'optical_flow_visualization.png'), flow_visualization)
    print(f"Visualización del flujo óptico guardado como {os.path.join(output_path,'optical_flow_visualization.png')}")
    # np.save('/Users/diegosilveracoeff/Desktop/Fing/DCM/flow_x.npy', flow[..., 0])
    # np.save('/Users/diegosilveracoeff/Desktop/Fing/DCM/flow_y.npy', flow[..., 1])
    # cv2.imwrite('/Users/diegosilveracoeff/Desktop/Fing/DCM/optical_flow_visualization.png', flow_visualization)

    #TODO: Estudiar cuál es el problema de esta visualización
    # Mostrar la imagen con el flujo óptico
    plt.imshow(cv2.cvtColor(flow_visualization, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Ocultar ejes
    plt.show()

    return

def encoder_motion_correction(current_frame_path, reference_frame_path, flow_x_path, flow_y_path, output_path):
    
    curr_frame = cv2.imread(current_frame_path)
    prev_frame = cv2.imread(reference_frame_path)
    

    # Convertir las imágenes a escala de grises
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Cargar los archivos flow_x.npy y flow_y.npy
    flow_x = np.load(flow_x_path)
    flow_y = np.load(flow_y_path)

    # Combinar los componentes x e y para obtener la variable flow
    flow = np.stack((flow_x, flow_y), axis=-1)


    # Aplicar el flujo óptico al segundo frame
    corrected_frame = np.zeros_like(curr_gray)
    # corrected_frame = curr_gray
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            dx, dy = flow[y, x]
            x2 = min(max(x - dx + 1, 0), flow.shape[1] - 1)
            y2 = min(max(y - dy + 1, 0), flow.shape[0] - 1)
            corrected_frame[int(np.trunc(y2)), int(np.trunc(x2))] = curr_gray[y, x]

    #Alternativa para manejar sectores sin cubrir
    # corrected_frame_npwhere_reference_based = corrected_frame.copy()

    # for i in range(corrected_frame.shape[0]):
    #     for j in range(corrected_frame.shape[1]):
    #         corrected_frame_npwhere_reference_based[i][j] = np.where(corrected_frame_npwhere_reference_based[i][j] == 0, prev_gray[i][j], corrected_frame_npwhere_reference_based[i][j])

    cv2.imwrite(output_path, corrected_frame)


    print(f"Current frame con motion correction guardado como {output_path}")

    return

def decoder_motion_correction(current_frame_path, reference_frame_path, flow_x_path, flow_y_path, output_path):
    
    curr_frame = cv2.imread(current_frame_path)
    prev_frame = cv2.imread(reference_frame_path)
    

    # Convertir las imágenes a escala de grises
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Cargar los archivos flow_x.npy y flow_y.npy
    flow_x = np.load(flow_x_path)
    flow_y = np.load(flow_y_path)

    # Combinar los componentes x e y para obtener la variable flow
    flow = np.stack((flow_x, flow_y), axis=-1)


    # Aplicar el flujo óptico al segundo frame
    corrected_frame = np.zeros_like(curr_gray)
    # corrected_frame = curr_gray
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            dx, dy = flow[y, x]
            x2 = min(max(x + dx, 0), flow.shape[1] - 1)
            y2 = min(max(y + dy, 0), flow.shape[0] - 1)
            corrected_frame[int(np.trunc(y2)), int(np.trunc(x2))] = curr_gray[y, x]

    #Alternativa para manejar sectores sin cubrir
    # corrected_frame_npwhere_reference_based = corrected_frame.copy()

    # for i in range(corrected_frame.shape[0]):
    #     for j in range(corrected_frame.shape[1]):
    #         corrected_frame_npwhere_reference_based[i][j] = np.where(corrected_frame_npwhere_reference_based[i][j] == 0, prev_gray[i][j], corrected_frame_npwhere_reference_based[i][j])

    cv2.imwrite(output_path, corrected_frame)


    print(f"Current frame con motion correction guardado como {output_path}")

    return

def dct(img_path, output_path):
    #Leo imagen 
    img_to_transform = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    # Ejemplo de cálculo de la DCT
    dct_img = dctn(img_to_transform, norm='ortho')  # DCT tipo 2
    cv2.imwrite(output_path, dct_img)
    return

def idct(img_path, output_path):
    #Leo imagen 
    img_to_antitransform = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    # Ejemplo de cálculo de la DCT
    dct_img = idctn(img_to_antitransform, norm='ortho')  # DCT tipo 2
    cv2.imwrite(output_path, dct_img)
    return

def plot_one_img(img_path):
    
    #Leo imagen
    img = cv2.imread(img_path)

    #Ploteo
    plt.imshow(img, cmap='gray', vmin=0, vmax=np.max(img))
    plt.colorbar()
    plt.title('Coeficientes DCT (log)')
    plt.axis('off')  # Ocultar ejes
    plt.show()
    return

def plot_two_images(img1_path, img2_path):

    #Leo imágenes
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Crear figura y ejes para los subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Subplot 1
    axs[0].imshow(img1, cmap='gray', vmin=0, vmax=np.max(img1))
    axs[0].set_title('Coeficientes IDCT en ref_img')
    axs[0].axis('off')
    #plt.colorbar(ax=axs[0])

    # Subplot 2
    axs[1].imshow(img2, cmap='gray', vmin=0, vmax=np.max(img2))
    axs[1].set_title('Coeficientes IDCT en cur_img')
    axs[1].axis('off')
    #plt.colorbar(ax=axs[1])

    plt.tight_layout()  # Ajustar espaciado entre subplots
    plt.show()
    return

def count_pixel_values(image_path):
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(image_array.shape)
    rows, columns = image_array.shape
    # Redimensionar el arreglo a una sola dimensión
    array_1d = image_array.reshape(rows * columns)
    values, counts = np.unique(image_array, return_counts=True)
    result = [(str(value), count) for value, count in zip(values, counts) if count > 0]
    return result

def huffman_codebook(counted_pixels):
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    #print(img.shape)
    # Crea un árbol de Huffman a partir de las probabilidades
    arbol_huffman = huffman.codebook(counted_pixels) #Normaliza y devuelve lista

    # Imprime la codificación de cada variable
    simbolos = []
    codigos = []
    for simbolo, codigo in arbol_huffman.items():
        print(f'Símbolo: {simbolo}, Código Huffman: {codigo}')
        simbolos.append(simbolo)
        codigos.append(codigo)

    return np.array([int(simbolo) for simbolo in simbolos]), np.array(codigos)

def write_encoded_file(image_path, symbols, codes, output_path):
    encoded_file = ''
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(image_array.shape)
    rows, columns = image_array.shape
    # Redimensionar el arreglo a una sola dimensión
    array_1d = image_array.reshape(rows * columns)
    for i in range(len(array_1d)):
        j = np.where(symbols == array_1d[i])
        print('j: ', j[0][0])
        encoded_file = encoded_file + codes[j[0][0]]
    print(encoded_file)
    with open(output_path, 'wb') as f:
        # Convertir la cadena de bits en una secuencia de bytes
        byte_data = int(encoded_file, 2).to_bytes((len(encoded_file) + 7) // 8, byteorder='big')
        f.write(byte_data)
    return

def bin_to_string(bytes):
    '''
    Función utilizada en read_and_decode_file
    Recibe el contenido de un archivo binario y devuelve un string con el mismo
    '''

    # Utilizamos la función format() con '08b' para obtener una cadena de bits de 8 dígitos por byte
    string = ''.join(format(byte, '08b') for byte in bytes)
    return string

def read_fillout_number(string):
    '''
    Función utilizada en read_and_decode_file
    Recibe un string y devuelve el número entero correspondiente a ese string en binario.
    '''
    fillout_number = int(string, 2)
    return fillout_number

def read_and_decode_file(bin_path):
    '''
    Lee el archivo binario, halla el fillout_number y lo usa para obtener el mensaje original
    '''
    with open(bin_path, 'rb') as file:
        file_content = file.read()
        decoded_string = bin_to_string(file_content)
        fillout_number = read_fillout_number(decoded_string[:8])
        message = decoded_string[8+fillout_number:]
        print(f'Cadena binaria: {decoded_string}')
        print(f'Primeros 8 bits: {decoded_string[:8]}')
        print(f'Número entero: {fillout_number}')
        print(f'Imagen codificado: {message}')