import modules.fun_gps as fun_gps # activar GPS --> sudo gpsd /dev/serial0 -F /var/run/gpsd.sock
# Verificar GPS --> cgps -s
import modules.counter as counter
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--enable_edgetpu', action='store_true')
    args = parser.parse_args()
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
                counter.run(args.model, args.width, args.height, args.num_threads, args.enable_edgetpu)


if __name__ == '__main__':
    main()