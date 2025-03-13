import fun_gps # activar GPS --> sudo gpsd /dev/serial0 -F /var/run/gpsd.sock
# Verificar GPS --> cgps -s

def main():

    # Funcion utilizada cuando una persona se suba o baje del bus
    #datosGPS = fun_gps.obtenerDatosGPS()
    #print(datosGPS)

    # Funcion para validar que el bus salio del parqueadero
    validarPosicion = fun_gps.validarPosicionGPS()
    #print("distancia: "+ str(validarPosicion))

    #Hora programada obtenida de la API
    horaProgramada = 2
    if horaProgramada >= 1:
        
        #Validar suscripcion --> Obtenida de la API
        suscripcion = "activo"
        if suscripcion == "activo":
            
            #Validar salida del bus
            if validarPosicion > 0.2:
                #activar contador
                print("Activar contador")


main()