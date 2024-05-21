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
    energy = np.sum(np.abs(image))
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

def motion_compensation_2(reference_frame_path, current_frame_path, flow_x_path, flow_y_path, output_path):

    curr_frame = cv2.imread(current_frame_path)
    ref_frame = cv2.imread(reference_frame_path)

    # Convertir las imágenes a escala de grises
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # Cargar los archivos flow_x.npy y flow_y.npy
    flow_x = np.load(flow_x_path)
    flow_y = np.load(flow_y_path)

    corrected_reference = np.zeros_like(curr_gray)
    height, width = corrected_reference.shape

    for i in range(height):
        for j in range(width):
            # Calcular los nuevos índices con el flujo
            new_i = i - round(flow_y[i][j])
            new_j = j - round(flow_x[i][j])

            # Comprobar si los nuevos índices están dentro de los límites
            if 0 <= new_i < height and 0 <= new_j < width:
                corrected_reference[i][j] = ref_gray[new_i][new_j]
            else:
                corrected_reference[i][j] = 0  # O algún valor predeterminado

    cv2.imwrite(output_path, corrected_reference, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    plt.imshow(corrected_reference, cmap='gray', vmin=0, vmax=np.max(corrected_reference))
    plt.colorbar()
    plt.title('Corrected reference image')
    plt.axis('off')  # Ocultar ejes
    plt.show()
    return

def dct_2(image_path, output_path):
    #Leo imagen 
    img_to_transform = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    width, height = img_to_transform.shape
    dct_img = np.empty((width, height))


    for x in range(0, width, 8):
        for y in range(0, height, 8):
            block = np.array(img_to_transform[x:x+8,y:y+8])
            #print(block.shape)
            #print('x,y: ',x,y)
            dct_block = dctn(block, norm='ortho')  # DCT tipo 2
            dct_img[x:x+8,y:y+8] = dct_block
    # Ejemplo de cálculo de la DCT
    #dct_img = dctn(img_to_transform, norm='ortho')  # DCT tipo 2
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

def plot_two_images(img1_path, img2_path, tittles):

    #Leo imágenes
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Crear figura y ejes para los subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

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

def reorder_array(arr):
    result = []
    n = len(arr)
    zero_count = 0
    for i in range(n):
        if arr[i] != 0:
            # if i < n - 1 and np.any(arr[i+1:] != 0):
            #     flag = 0
            # else:
            #     flag = 1
            # result.append((zero_count, arr[i], flag))
            result.append((zero_count, arr[i]))
            zero_count = 0
        else:
            zero_count += 1
    # return np.array(result, dtype=[('zero_count', 'i4'), ('value', 'i4'), ('flag', 'i4')])
    return np.array(result, dtype=[('zero_count', 'i4'), ('value', 'i4')])

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

    #return np.array([int(simbolo) for simbolo in simbolos]), np.array(codigos)
    return np.array(simbolos), np.array(codigos)

def write_encoded_file_2(reordered_array, symbols, codes, output_path):
    '''
    '''
    encoded_file = ''

    for i in range(len(reordered_array)):
        # print(reordered_array[i])
        # print(symbols)
        j = np.where(symbols == str(reordered_array[i]))
        # print(j)
        encoded_file = encoded_file + codes[j[0][0]]
    print('Largo de la imagen codificada (mensaje)', len(encoded_file))
    bytes_to_write = add_fillout_number(encoded_file)
    with open(output_path, 'wb') as f:
        # Convertir la cadena de bits en una secuencia de bytes
        final_bytes = int(bytes_to_write, 2).to_bytes((len(bytes_to_write) + 7) // 8, byteorder='big')
        f.write(final_bytes)
    return

def add_fillout_number(message):
    '''
    Función utilizada en write_encoded_file
    Recibe el mensaje codificado, completa con 0s para tener largo en bytes
    y agrega número de 0s agregados para correcta decodificación
    '''
    len_message = len(message)
    if len_message % 8 != 0:
        fillout_number = 8 - len_message % 8
    else:
        fillout_number = 0
    print('Fillout_number: ', fillout_number)
    for i in range(fillout_number):
        message = '0' + message
    message = format(fillout_number, '08b') + message
    print(message)
    return message

def read_bin_file(bin_path):
    '''
    Lee el archivo binario, halla el fillout_number y lo usa para obtener el mensaje original
    '''
    with open(bin_path, 'rb') as file:
        file_content = file.read()
        decoded_string = bin_to_string(file_content)
        fillout_number = read_fillout_number(decoded_string[:8])
        message = decoded_string[8+fillout_number:]
        #print(f'Cadena binaria: {decoded_string}')
        print(f'Primeros 8 bits: {decoded_string[:8]}')
        print(f'Número entero: {fillout_number}')
        #print(f'Imagen codificado: {message}')
    return message

def bin_to_string(bytes):
    '''
    Función utilizada en read_bin_file
    Recibe el contenido de un archivo binario y devuelve un string con el mismo
    '''

    # Utilizamos la función format() con '08b' para obtener una cadena de bits de 8 dígitos por byte
    string = ''.join(format(byte, '08b') for byte in bytes)
    return string

def read_fillout_number(string):
    '''
    Función utilizada en read_bin_file
    Recibe un string y devuelve el número entero correspondiente a ese string en binario.
    '''
    fillout_number = int(string, 2)
    return fillout_number

def decode_symbols_2(message, symbols, codes):
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
    print(len(decoded_symbols))
    print(decoded_symbols[2])
    return np.array(decoded_symbols)

def reconstruct_array(array_salida, original_length):
    reconstructed_array = np.zeros(original_length, dtype=int)
    index = 0
    for zero_count, value in array_salida:
        index += zero_count
        reconstructed_array[index] = value
        index += 1
    return reconstructed_array

def complete_with_zeros(list, length):
    """
    Completa una lista con ceros hasta alcanzar el largo especificado.

    :param lista: Lista original.
    :param largo: Largo deseado para la lista.
    :return: Lista completada con ceros hasta alcanzar el largo especificado.
    """
    # Calculamos cuántos ceros necesitamos agregar
    num_zeros = max(0, length - len(list))
    if num_zeros != 0:
        # Creamos una nueva lista agregando los ceros necesarios
        full_list = list + [0] * num_zeros
    else:
        full_list = list
    
    return full_list

def idct_2(img_to_transform, output_path):
    #Leo imagen 
    #img_to_transform = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    width, height = img_to_transform.shape
    idct_img = np.empty((width, height))
    #img_to_antitransform = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)

    for x in range(0, width, 8):
        for y in range(0, height, 8):
            block = np.array(img_to_transform[x:x+8,y:y+8])
            #print(block.shape)
            #print('x,y: ',x,y)
            idct_block = idctn(block, norm='ortho')  # DCT tipo 2
            idct_img[x:x+8,y:y+8] = idct_block

    # Ejemplo de cálculo de la DCT
    #decoded_residual = idctn(img, norm='ortho')  # DCT tipo 2
    cv2.imwrite(output_path, idct_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    return idct_img