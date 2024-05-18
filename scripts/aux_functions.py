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
            output_path = os.path.join(output_folder, f"frame_{frame_num}.tif")
            cv2.imwrite(output_path, gray_frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
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
    cv2.imwrite(output_path, diff_img_adjusted, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    print(f"Resultado de la resta ajustado y guardado como {output_path}")

    return

def energy(image_path):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    energy = np.sum(np.abs(image-128))
    print(f'La energía de {image_name} es {energy}')

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
    step = 8  # Espaciado para mostrar los vectores de flujo
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
    cv2.imwrite(os.path.join(output_path,'optical_flow_visualization.png'), flow_visualization, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    print(f"Visualización del flujo óptico guardado como {os.path.join(output_path,'optical_flow_visualization.png')}")

    #TODO: Estudiar cuál es el problema de esta visualización
    # Mostrar la imagen con el flujo óptico
    plt.imshow(cv2.cvtColor(flow_visualization, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Ocultar ejes
    plt.show()

    return

def motion_compensation(reference_frame_path, current_frame_path, flow_x_path, flow_y_path, output_path):

    curr_frame = cv2.imread(current_frame_path)
    ref_frame = cv2.imread(reference_frame_path)


    # Convertir las imágenes a escala de grises
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # Cargar los archivos flow_x.npy y flow_y.npy
    flow_x = np.load(flow_x_path)
    flow_y = np.load(flow_y_path)

    corrected_reference = np.zeros_like(curr_gray)
    for i in range(corrected_reference.shape[0]):
        for j in range(corrected_reference.shape[1]):
            corrected_reference[i][j] = ref_gray[i - round(flow_y[i][j])][j - round(flow_x[i][j])]

    #corrected_reference_path = os.path.join(folder_path, 'corrected_reference.tif')
    cv2.imwrite(output_path, corrected_reference, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    plt.imshow(corrected_reference, cmap='gray', vmin=0, vmax=np.max(corrected_reference))
    plt.colorbar()
    plt.title('Corrected reference image')
    plt.axis('off')  # Ocultar ejes
    plt.show()
    return

def dct(image_path, output_path):
    #Leo imagen 
    img_to_transform = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    # Ejemplo de cálculo de la DCT
    dct_img = dctn(img_to_transform, norm='ortho')  # DCT tipo 2
    cv2.imwrite(output_path, dct_img)
    return dct_img

def quantization(image, q_step, output_path):
    '''
    '''
    flat_image = image.reshape((image.shape[0] * image.shape[1]))
    quantized_list = [round(value / q_step) * q_step for value in flat_image]
    quantized_array =  np.array(quantized_list)
    quantized_image = quantized_array.reshape((image.shape[0], image.shape[1]))
    cv2.imwrite(output_path, quantized_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    return quantized_image, quantized_array

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

def plot_three_images(img1_path, img2_path, img3_path, tittles):

    #Leo imágenes
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)

    # Crear figura y ejes para los subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Subplot 1
    axs[0].imshow(img1, cmap='gray', vmin=0, vmax=np.max(img1))
    axs[0].set_title(tittles[0])
    axs[0].axis('off')
    #plt.colorbar(ax=axs[0])

    # Subplot 2
    axs[1].imshow(img2, cmap='gray', vmin=0, vmax=np.max(img2))
    axs[1].set_title(tittles[1])
    axs[1].axis('off')
    #plt.colorbar(ax=axs[1])

    # Subplot 3
    axs[2].imshow(img3, cmap='gray', vmin=0, vmax=np.max(img3))
    axs[2].set_title(tittles[2])
    axs[2].axis('off')

    plt.tight_layout()  # Ajustar espaciado entre subplots
    plt.show()
    return

def count_pixel_values(image_array):
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

def add_fillout_number(message):
    '''
    Función utilizada en write_encoded_file
    Recibe el mensaje codificado, completa con 0s para tener largo en bytes
    y agrega número de 0s agregados para correcta decodificación
    '''
    len_message = len(message)
    fillout_number = 8 - len_message % 8
    print('Fillout_number: ', fillout_number)
    for i in range(fillout_number):
        message = '0' + message
    message = format(fillout_number, '08b') + message
    print(message)
    return message

def write_encoded_file(q_array, symbols, codes, output_path):
    '''
    '''
    encoded_file = ''

    for i in range(len(q_array)):
        j = np.where(symbols == q_array[i])
        encoded_file = encoded_file + codes[j[0][0]]
    print('Largo de la imagen codificada (mensaje)', len(encoded_file))
    bytes_to_write = add_fillout_number(encoded_file)
    with open(output_path, 'wb') as f:
        # Convertir la cadena de bits en una secuencia de bytes
        final_bytes = int(bytes_to_write, 2).to_bytes((len(bytes_to_write) + 7) // 8, byteorder='big')
        f.write(final_bytes)
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

def read_bin_file(bin_path):
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
    return message

def decode_symbols(message, symbols, codes, dimension):
    '''
    Recibe como entrada el mensaje, y el huffman_codebook.
    Devuelve los valores originales de la imagen antes de codificarla.
    TODO: Cuando haga un encabezado con metada tengo que pasarle las dimensiones
          originales de la imagen para hacer reshape, ahora esta flatten.
    '''
    decoded_symbols = []
    coded_symbol = ''
    for c in message:
        coded_symbol += c
        if coded_symbol in codes:
            #print(coded_symbol)
            i = np.where(codes == coded_symbol)
            decoded_symbols.append(symbols[i][0]) #No sé por qué lleva ese 0 ahí..
            coded_symbol = ''

    return np.array(decoded_symbols).reshape(dimension)