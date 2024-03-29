import numpy as np

from matplotlib import pyplot as plt
from PIL import Image


def lectura(name_file):
    replica = Image.open("Numerical_propagators/imagenes/" +
                         str(name_file)).convert('L')
    replica.save("Numerical_propagators/imagenes/copia.png")
    return (replica)

# Función para el guardado de la imagen


def guardado(name_out, matriz):
    resultado = Image.fromarray(matriz)
    resultado.save("Numerical_propagators/magenes/"+str(name_out))


def mostrar(matriz, titulo, ejex, ejey):
    plt.imshow(matriz, cmap='gray')
    plt.title(str(titulo))
    plt.xlabel(str(ejex))
    plt.ylabel(str(ejey))
    plt.show()


def amplitud(matriz):
    amplitud = np.abs(matriz)
    return (amplitud)


def intensidad(matriz):
    intensidad = np.abs(matriz)
    intensidad = np.power(intensidad, 2)
    return (intensidad)


def fase(matriz):
    fase = np.angle(matriz, deg=False)
    return (fase)


def dual_img(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.show()


def dual_save(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.savefig('Numerical_propagators/imagenes/guardado.png', dpi=1000)
