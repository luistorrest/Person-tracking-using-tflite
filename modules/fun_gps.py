import gps
from math import radians, sin, cos, sqrt, atan2

def obtenerDatosGPS():

    try:
        # Iniciar sesion con el GPS
        session = gps.gps(mode=gps.WATCH_ENABLE)
        
        vGPS = False

        while vGPS == False:
            report = session.next()

            if report['class'] == 'TPV':  # TPV contiene la informacion de posicion y tiempo
                latitud = getattr(report, 'lat', 'N/A')
                longitud = getattr(report, 'lon', 'N/A')
                hora_gps = getattr(report, 'time', 'N/A')  # Hora UTC del GPS

                #print(f"Latitud: {latitud}, Longitud: {longitud}, Hora GPS: {hora_gps}")
                datos_gps = [latitud, longitud, hora_gps]

                del session # Eliminar sesion GPS
                vGPS = True
                return datos_gps

        return "Datos no obtenidos"
    except KeyError:
        return "Error: GPS no encontrado, activar."
    
def validarPosicionGPS():

    # coordenada API (latitud, longitud)
    latEstacion = 2.451526667
    lonEstacion = -76.589311667

    # Radio de la Tierra en kilometros
    R = 6371.0

    # obtener datos GPS (latitud, longitud)
    obtGPS = obtenerDatosGPS()  

    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(radians, [latEstacion, lonEstacion, obtGPS[0], obtGPS[1]])

    # Diferencias de coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formula de Haversine
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distancia en kilometros
    distancia = R * c
    return distancia



