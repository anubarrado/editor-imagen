# -*- coding: utf-8 -*-
from __future__ import division
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageFilter
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2
import math
import scipy as scp
import pylab as pyl
from nt_toolbox.general import *
from nt_toolbox.signal import *
from nt_toolbox.perform_wavelet_transf import *
import warnings
 

def ecualiza():    
    img = cv2.imread(fbi)
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
     
    cv2.imwrite('result.png',hist_equalization_result)    
    img2=cv2.imread('result.png')
    return cv2.imshow('image',img2) 
    
def grises():
    foto = Image.open(fbi)
    datos = foto.getdata()
    promedio =  [(datos[x][0] + datos[x][1] + datos[x][2]) // 3 for x in range(len(datos))]
    imagen_gris = Image.new('L', foto.size)
    imagen_gris.putdata(promedio)
    gris=imagen_gris    
    return foto.show(),gris.show()  

def brillar():    
    im=fbi
    alpha=int(br.get())
    original=Image.open(im)
    im = Image.open(im)
    im12 = im
    i = 0
    while i < im12.size[0]:
        j = 0
        while j < im12.size[1]:
            valor = im12.getpixel((i, j))           
            valor = valor + alpha
            if valor>= 255:
                valor = 255
            else:
                valor = valor
            im12.putpixel((i, j),(valor))
            j+=1
        i+=1        
    return original.show('Original'),im12.show('Brillo')
 
    
def contraste():    
    im=fbi
    alpha=int(ct.get())
    original=Image.open(im)
    im = Image.open(im)
    im12 = im
    i = 0
    while i < im12.size[0]:
        j = 0
        while j < im12.size[1]:
            valor = im12.getpixel((i, j))            
            valor = valor * alpha
            if valor >= 255:
                valor = 255
            if valor <= 0:
                valor = valor
            im12.putpixel((i, j),(valor))
            j+=1
        i+=1        
    return original.show(),im12.show()
    
def umbralizado():
    
    foto=Image.open(fbi)
    if foto.mode != 'L':
        foto=foto.convert('L')      

    umbral=int(br.get())
    
    datos=foto.getdata()
    datos_binarios=[]
    
    for x in datos:
        if x<umbral:
            datos_binarios.append(0)
            continue
        datos_binarios.append(1) 
    nueva_imagen=Image.new('1',foto.size)
    nueva_imagen.putdata(datos_binarios)
    nueva_imagen.save('umbral.jpg')
    
    foto_umbral=Image.open('umbral.jpg')
    return foto.show(), foto_umbral.show()
    
def suavizado():
    
    imagen_original = Image.open(fbi)
    tamaño = (3,3)
    a=1
    coeficientes= [a, a, a, a, a, a, a, a, a]
    imagen_filtrada = imagen_original.filter(ImageFilter.Kernel(tamaño, coeficientes))    
    
    return imagen_original.show(),imagen_filtrada.show()

def filtro_promedio():    
    img = cv2.imread(fbi)
    a=int(br.get())
    kernel = np.ones((a,a),np.float32)/(a*a)
    dst = cv2.filter2D(img,-1,kernel)
    return cv2.imshow('Original',img),cv2.imshow('Filtro_promedio',dst)

def filtro_gaussiano():    
    img=cv2.imread(fbi)
    a=int(br.get())
    gris_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussiana = cv2.GaussianBlur(gris_color, (a,a), 0)

    return cv2.imshow('Original',img),cv2.imshow('Filtro_Gaussiano',gaussiana)

def difuminar():    
    a=int(br.get())    
    img = cv2.imread('color2.jpg')
    blur = cv2.blur(img,(a,a))
    return cv2.imshow('Original',img),cv2.imshow('Difuminada',blur)

def bordes():    
    img=cv2.imread(fbi)
    canny = cv2.Canny(img, 600, 415)
    return cv2.imshow('Original',img), cv2.imshow('Bordes',canny)

def ej_bordes():    
    original= cv2.imread("monedas.jpg")
    cv2.imshow("original", original)
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gris, (5,5), 0)
    cv2.imshow("suavizado", gauss)
    canny = cv2.Canny(gauss, 50, 150)
    cv2.imshow("canny", canny)
    (_, contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("He encontrado {} objetos".format(len(contornos)))
    cv2.drawContours(original,contornos,-1,(0,0,255), 2)
    cv2.imshow("contornos", original)
    (_, contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("He encontrado {} objetos".format(len(contornos)))
    cv2.drawContours(original,contornos,-1,(0,0,255), 2)

def sobel():    
    a=int(br.get())
    img = cv2.imread(fbi,0)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=a)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=a)
    return cv2.imshow('Original',img), cv2.imshow('Sobel X',sobelx),cv2.imshow('Sobel Y',sobelx)

def laplace():    
    img = cv2.imread(fbi,0)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)    
    return cv2.imshow('Original',img), cv2.imshow('Laplaca',laplacian)

def mediana():    
    a=int(br.get())
    img = cv2.imread(fbi)
    median = cv2.medianBlur(img,a)
    return cv2.imshow('Original',img), cv2.imshow('Filtro_Mediana',median)

def dilatacion():    
    a=int(br.get())
    b=int(ct.get())
    img = cv2.imread(fbi,0)
    kernel = np.ones((a,a),np.uint8)
    dilatacion = cv2.dilate(img,kernel,iterations = b)
    return cv2.imshow('Original',img), cv2.imshow('Dilatada',dilatacion)

def erosion():    
    a=int(br.get())
    b=int(ct.get())
    img = cv2.imread(fbi,0)
    kernel = np.ones((a,a),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = b)
    return cv2.imshow('Original',img), cv2.imshow('Erosionada',erosion)

def cierre():    
    a=int(br.get())
    img = cv2.imread(fbi,0)
    kernel = np.ones((a,a),np.uint8)
    cierre = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return cv2.imshow('Original',img), cv2.imshow('Cierre',cierre)
    
def apertura():    
    a=int(br.get())
    img = cv2.imread(fbi,0)
    kernel = np.ones((a,a),np.uint8)
    apertura = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cv2.imshow('Original',img), cv2.imshow('Apertura',apertura)

def esqueletizacion():    
    img = cv2.imread(fbi,0)
    img1=img
    size = np.size(img1)
    skel = np.zeros(img1.shape,np.uint8)    
    ret,img1 = cv2.threshold(img1,172,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img1,element)
        temp1 = cv2.dilate(eroded,element)
        temp = cv2.subtract(img1,temp1)
        skel = cv2.bitwise_or(skel,temp)
        img1 = eroded.copy()
        
        zeros = size - cv2.countNonZero(img1)
        if zeros==size:
            done = True
    
    return cv2.imshow('Original',img), cv2.imshow('Esqueletizada',skel)

def rotacion():
    a=int(br.get())
    img = cv2.imread(fbi,0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),a,1)
    rot = cv2.warpAffine(img,M,(cols,rows))        
    return cv2.imshow('Original',img), cv2.imshow('Rotacion',rot)

def traslacion():    
    img = cv2.imread(fbi,0)
    rows,cols = img.shape
    M = np.float32([[1,0,50],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return cv2.imshow('Original',img), cv2.imshow('Traslacion',dst)

def trans_afin():    
    img = cv2.imread(fbi)
    rows,cols,ch = img.shape
    pts1 = np.float32([[100,400],[400,100],[100,100]])
    pts2 = np.float32([[50,300],[400,200],[80,150]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return cv2.imshow('Original',img), cv2.imshow('Transformación Afin',dst)
        
def abrir():
    global fbi
    filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    fbi=filename
    print (filename)
    
def trans_wavelet():
    global fbi
    n = 1024
    f0 = rescale(load_image(fbi, n))
    sigma = 0.15
    f = f0 + sigma*random.standard_normal((n,n))
    Jmin = 8
    wav  = lambda f  : perform_wavelet_transf(f, Jmin, +1)
    iwav = lambda fw : perform_wavelet_transf(fw, Jmin, -1)

    plt.figure(figsize=(10,10))
    plot_wavelet(wav(f), Jmin)
    plt.show()

def decodificar_bmp():    
    imagen=input('ingresar imagen en bmp: ')
    file = open(imagen,'rb')
    datos = file.read()
    print ("Tipo de Fichero -------------(0-1   0x00-0x01 bytes):" + str(datos[0]) + ' - ' + str(datos[1]))
    Tamano_Fichero = datos[2] + (datos[3]*256) + (datos[4]*65536) + (datos[5]*167772160)
    print ("Tamano de Fichero -----------(2-5   0x02-0x05 bytes):" + str(Tamano_Fichero)+ ' - ' + hex(Tamano_Fichero))
    print ("2 bytes reversados ----------(6-7   0x06-0x07 bytes)")
    print ("2 bytes reversados ----------(8-9   0x08-0x09 bytes)")
    Inicio_Imagen = datos[10] + (datos[11]*256) + (datos[12]*65536) + (datos[13]*167772160)
    print ("Inicio de Datos de Imagen ---(10-13 0x0A-0x0D bytes):" + str(Inicio_Imagen) + ' - ' + hex(Inicio_Imagen))
    Tamano_Cabecera = datos[14] + (datos[15]*256) + (datos[16]*65536) + (datos[17]*167772160)
    print ("Tamano de Cabecera de Bitmap (14-17 0x0E-0x11 bytes):" + str(Tamano_Cabecera) + ' - ' + hex(Tamano_Cabecera))
    Ancho_Pixeles = datos[18] + (datos[19]*256) + (datos[20]*65536) + (datos[21]*167772160)
    print ("Anchura (píxeles) -----------(18-21 0x12-0x15 bytes):" + str(Ancho_Pixeles) + ' - ' + hex(Ancho_Pixeles))
    Alto_Pixeles = datos[22] + (datos[23]*256) + (datos[24]*65536) + (datos[25]*167772160)
    print ("Altura (píxeles) ------------(22-25 0x16-0x19 bytes):" + str(Alto_Pixeles) + ' - ' + hex(Alto_Pixeles))
    Nro_Planos = datos[26] + (datos[27]*256)
    print ("Nro de Planos ---------------(26-27 0x1A-0x1B bytes):" + str(Nro_Planos) + ' - ' + hex(Nro_Planos))
    Tamano_punto = datos[28] + (datos[29]*256)
    print ("Tamaño de cada punto --------(28-29 0x1C-0x1D bytes):" + str(Tamano_punto) + ' - ' + hex(Tamano_punto))
    Compresion = datos[30] + (datos[31]*256) + (datos[32]*65536) + (datos[33]*167772160)
    print ("Compresion  -----------------(30-33 0x1E-0x21 bytes):" + str(Compresion) + ' - ' + hex(Compresion))
    Tamano_Imagen = datos[34] + (datos[35]*256) + (datos[36]*65536) + (datos[37]*167772160)
    print ("Tamano de Imagen ------------(34-37 0x22-0x25 bytes):" + str(Tamano_Imagen) + ' - ' + hex(Tamano_Imagen))
    Resolucion_Horizontal = datos[38] + (datos[39]*256) + (datos[40]*65536) + (datos[41]*167772160)
    print ("Resolucion Horizontal -------(38-41 0x26-0x29 bytes):" + str(Resolucion_Horizontal) + ' - ' + hex(Resolucion_Horizontal))
    Resolucion_Vertical = datos[42] + (datos[43]*256) + (datos[44]*65536) + (datos[45]*167772160)
    print ("Resolucion Vertical ---------(42-45 0x2A-0x2D bytes):" + str(Resolucion_Vertical) + ' - ' + hex(Resolucion_Vertical))
    Tamano_Tabla_Color = datos[46] + (datos[47]*256) + (datos[48]*65536) + (datos[49]*167772160)
    print ("Tamano de Tabla de Color ----(46-49 0x2E-0x31 bytes):" + str(Tamano_Tabla_Color) + ' - ' + hex(Tamano_Tabla_Color))
    Contador_Color_Importante = datos[50] + (datos[51]*256) + (datos[52]*65536) + (datos[53]*167772160)
    print ("Contador de Color Importante (50-53 0x32-0x35 bytes):" + str(Contador_Color_Importante) + ' - ' + hex(Contador_Color_Importante))

    R=np.zeros((Alto_Pixeles,Ancho_Pixeles))
    G=np.zeros((Alto_Pixeles,Ancho_Pixeles))
    B=np.zeros((Alto_Pixeles,Ancho_Pixeles))
    
    if Ancho_Pixeles*3*Alto_Pixeles == Tamano_Imagen:
        print ("Sin Ceros")
        separador = 0
    elif ((Ancho_Pixeles*3)+1)*Alto_Pixeles == Tamano_Imagen:
        print ("Con Ceros")
        separador = 1
    else:
        print ("Ninguno")
        print ("Upps ... Falta entender la decodificacion")

    fila=Alto_Pixeles-1
    columna=0
    puntero=0
    for posicion in range (Inicio_Imagen,Tamano_Fichero):
        if puntero == 0:
            B[fila,columna]=datos[posicion]
            puntero=puntero+1
        elif puntero == 1:
            G[fila,columna]=datos[posicion]
            puntero=puntero+1
        elif puntero == 2:
            R[fila,columna]=datos[posicion]
            if columna == (Ancho_Pixeles-1):
                columna = 0
                fila = fila - 1
                if separador == 1:
                    puntero=puntero+1
                else:
                    puntero=0
            else :
                columna=columna+1
                puntero = 0
        else :
            puntero = 0

    return print ("Tipo de Fichero -------------(0-1   0x00-0x01 bytes):" + str(datos[0]) + ' - ' + str(datos[1]))
    Tamano_Fichero = datos[2] + (datos[3]*256) + (datos[4]*65536) + (datos[5]*167772160)
    print ("Tamano de Fichero -----------(2-5   0x02-0x05 bytes):" + str(Tamano_Fichero)+ ' - ' + hex(Tamano_Fichero))
    print ("2 bytes reversados ----------(6-7   0x06-0x07 bytes)")
    print ("2 bytes reversados ----------(8-9   0x08-0x09 bytes)")
    Inicio_Imagen = datos[10] + (datos[11]*256) + (datos[12]*65536) + (datos[13]*167772160)
    print ("Inicio de Datos de Imagen ---(10-13 0x0A-0x0D bytes):" + str(Inicio_Imagen) + ' - ' + hex(Inicio_Imagen))
    Tamano_Cabecera = datos[14] + (datos[15]*256) + (datos[16]*65536) + (datos[17]*167772160)
    print ("Tamano de Cabecera de Bitmap (14-17 0x0E-0x11 bytes):" + str(Tamano_Cabecera) + ' - ' + hex(Tamano_Cabecera))
    Ancho_Pixeles = datos[18] + (datos[19]*256) + (datos[20]*65536) + (datos[21]*167772160)
    print ("Anchura (píxeles) -----------(18-21 0x12-0x15 bytes):" + str(Ancho_Pixeles) + ' - ' + hex(Ancho_Pixeles))
    Alto_Pixeles = datos[22] + (datos[23]*256) + (datos[24]*65536) + (datos[25]*167772160)
    print ("Altura (píxeles) ------------(22-25 0x16-0x19 bytes):" + str(Alto_Pixeles) + ' - ' + hex(Alto_Pixeles))
    Nro_Planos = datos[26] + (datos[27]*256)
    print ("Nro de Planos ---------------(26-27 0x1A-0x1B bytes):" + str(Nro_Planos) + ' - ' + hex(Nro_Planos))
    Tamano_punto = datos[28] + (datos[29]*256)
    print ("Tamaño de cada punto --------(28-29 0x1C-0x1D bytes):" + str(Tamano_punto) + ' - ' + hex(Tamano_punto))
    Compresion = datos[30] + (datos[31]*256) + (datos[32]*65536) + (datos[33]*167772160)
    print ("Compresion  -----------------(30-33 0x1E-0x21 bytes):" + str(Compresion) + ' - ' + hex(Compresion))
    Tamano_Imagen = datos[34] + (datos[35]*256) + (datos[36]*65536) + (datos[37]*167772160)
    print ("Tamano de Imagen ------------(34-37 0x22-0x25 bytes):" + str(Tamano_Imagen) + ' - ' + hex(Tamano_Imagen))
    Resolucion_Horizontal = datos[38] + (datos[39]*256) + (datos[40]*65536) + (datos[41]*167772160)
    print ("Resolucion Horizontal -------(38-41 0x26-0x29 bytes):" + str(Resolucion_Horizontal) + ' - ' + hex(Resolucion_Horizontal))
    Resolucion_Vertical = datos[42] + (datos[43]*256) + (datos[44]*65536) + (datos[45]*167772160)
    print ("Resolucion Vertical ---------(42-45 0x2A-0x2D bytes):" + str(Resolucion_Vertical) + ' - ' + hex(Resolucion_Vertical))
    Tamano_Tabla_Color = datos[46] + (datos[47]*256) + (datos[48]*65536) + (datos[49]*167772160)
    print ("Tamano de Tabla de Color ----(46-49 0x2E-0x31 bytes):" + str(Tamano_Tabla_Color) + ' - ' + hex(Tamano_Tabla_Color))
    Contador_Color_Importante = datos[50] + (datos[51]*256) + (datos[52]*65536) + (datos[53]*167772160)
    print ("Contador de Color Importante (50-53 0x32-0x35 bytes):" + str(Contador_Color_Importante) + ' - ' + hex(Contador_Color_Importante))
    
#interfaz
ventana=Tk()
ventana.geometry("650x300")
ventana.title("Editor de Imágenes")

imagen=PhotoImage(file="imagen_fondo.png")
fondo = Label(ventana,image=imagen).place(x=0,y=0)

barraMenu=Menu(ventana)
menuArchivo=Menu(barraMenu)

menuArchivo.add_command(label="Abrir",command=abrir)
menuArchivo.add_command(label="Salir",command=ventana.destroy)
menuGlobal=Menu(barraMenu)
menuGlobal.add_command(label="Estructura .bmp",command=decodificar_bmp)
menuGlobal.add_command(label="rgb a grises",command=grises)
menuGlobal.add_command(label="Contraste",command=contraste)
menuGlobal.add_command(label="Ecualizar",command=ecualiza)
menuGlobal.add_command(label="Brillo",command=brillar)
menuGlobal.add_command(label="Umbralizar",command=umbralizado)

menuTFLocal=Menu(barraMenu)
menuTFLocal.add_command(label="Suavizar",command=suavizado)
menuTFLocal.add_command(label="Filtro_Promedio",command=filtro_promedio)
menuTFLocal.add_command(label="Difuminada",command=difuminar)
menuTFLocal.add_command(label="Bordes",command=bordes)
menuTFLocal.add_command(label="Ejemplo_Bordes",command=ej_bordes)
menuTFLocal.add_command(label="Filtro_Sobel",command=sobel)
menuTFLocal.add_command(label="Filtro_Laplace",command=laplace)
menuTFLocal.add_command(label="Filtro_Gaussiano",command=filtro_gaussiano)
menuTFLocal.add_command(label="Filtro_Mediana",command=mediana)
menuTFLocal.add_command(label="Dilatada",command=dilatacion)
menuTFLocal.add_command(label="Erosionada",command=erosion)
menuTFLocal.add_command(label="Cierre",command=cierre)
menuTFLocal.add_command(label="Apertura",command=apertura)
menuTFLocal.add_command(label="Esqueletizada",command=esqueletizacion)

menuTFGeometrica=Menu(barraMenu)
menuTFGeometrica.add_command(label="Rotacion",command=rotacion)
menuTFGeometrica.add_command(label="Traslación",command=traslacion)
menuTFGeometrica.add_command(label="Transformación Afin",command=trans_afin)
menuDomFrecuencial=Menu(barraMenu)
menuDomFrecuencial.add_command(label="TransformadaWavelet",command=trans_wavelet)

barraMenu.add_cascade(label="Archivo",menu=menuArchivo)
barraMenu.add_cascade(label="Procesamineto_GlobalImagenes",menu=menuGlobal)
barraMenu.add_cascade(label="Transformaciones_Locales",menu=menuTFLocal)
barraMenu.add_cascade(label="Transformaciones_Geometricas",menu=menuTFGeometrica)
barraMenu.add_cascade(label="Dominio_Frecuencial",menu=menuDomFrecuencial)

ventana.config(menu=barraMenu)
br=IntVar()
sclBrillo=Scale(ventana,label="Barra de Control para imagenes", orient=HORIZONTAL, width=25,from_=-50,to=180,tickinterval=10,length=600,variable=br,fg='blue4').place(x=20,y=20)
ct=IntVar()
sclContraste=Scale(ventana,label="Contraste y N° de iteracciones", orient=HORIZONTAL, width=25,from_=0,to=7,tickinterval=0.5,length=600,variable=ct,fg='blue4').place(x=20,y=100)
ventana.mainloop()