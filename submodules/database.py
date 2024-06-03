import pandas as pd
import mysql.connector
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import threading
import sys
import PIL
import utm
from math import radians, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt
# import seaborn as sns
import torch
from .GPModels import GaussianProcessGPyTorch
import colorcet
sys.path.append('../')

class Database():
    def __init__(self,selected_map = 'alamillo'):
        self.sensors=['Conductivity', 'PH', 'Sonar', 'Temperature', 'Turbidity']
        self.password_path=self.resource_path("pass.json")
        with open(self.password_path,encoding="utf-8") as f:
            passw=json.load(f)
        self.user=passw["user"]
        self.password=passw["password"]
        self.host=passw["host"]
        self.selected_map = selected_map
        self.busy=False

        if self.selected_map == 'alamillo':
            map_folder=self.resource_path(f'assets/Maps')
            self.map_coords = { 'lat_min': 37.417823087, 'lat_max': 37.421340387, 'lon_min': -6.001553482, 'lon_max': -5.997342323 }
            self.pos_ini = utm.from_latlon(self.map_coords['lat_min'], self.map_coords['lon_min'])
            
            #generate scenario map
            image = cv2.imread(f"{map_folder}/{self.selected_map.capitalize()}Image2.png", 0)
            # Binarize the image
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            # Convert the binary image to 0s and 1s
            self.scenario_map = np.where(binary_image > 0, 1, 0).astype(float)
            # cargar imagen satelite
            self.satelite_img = plt.imread(f'{map_folder}/{self.selected_map.capitalize()}Sat.png')
            #get map shape
            rows, cols = self.scenario_map.shape
            self.res_lat, self.res_lon = (self.map_coords['lat_max'] - self.map_coords['lat_min']) / rows, (self.map_coords['lon_max'] - self.map_coords['lon_min']) / cols
            


    def query(self, date='2024-02-27'):
        #execute in a thread
        start=datetime.now()
        querythread=threading.Thread(target=self.__query,args=(date,))
        self.busy=True
        querythread.start()
        querythread.join()
        afterquery=datetime.now()
        difference = afterquery - start
        seconds_in_day = 24 * 60 * 60
        aux=divmod(difference.days * seconds_in_day + difference.seconds, 60)
        print(f"query delayed {aux[0]} minutes {aux[1]} seconds")
        gp_thread=threading.Thread(target=self.obtain_prediction_maps)
        gp_thread.start()
        gp_thread.join()
        aftergp=datetime.now()
        difference = aftergp - start
        aux=divmod(difference.days * seconds_in_day + difference.seconds, 60)
        print(f"gp delayed {aux[0]} minutes {aux[1]} seconds")


    def __query(self, date='2024-02-27'):
        self.date=date
        cnx = mysql.connector.connect(user=self.user, password=self.password, port="6006",
                                    host=self.host, database='wqp')
        # Crear un cursor
        cursor = cnx.cursor()
        # Definir la consulta SQL con el filtro de fecha
        # query = "SELECT * FROM ASV_US.ASV WHERE date(Date) >= '2024-02-16' AND date(Date) <= '2024-02-15'"
        query = f"SELECT * FROM wqp.WQP where date(Date) = '{date}';"

        # Ejecutar la consulta
        cursor.execute(query)

        # Obtener nombres de columnas
        column_names = cursor.column_names

        # Obtener los resultados
        results = cursor.fetchall()

        # Cerrar el cursor y la conexión a la base de datos
        cursor.close()
        cnx.close()

        # Crear un DataFrame de pandas con los resultados
        self._df = pd.DataFrame(results, columns=column_names)

        if len(self._df)==0:
            print("empty dataframe")
            return
        
        # Eliminar las muestras con coordenadas GPS no válidas
        self.df = self._df[self._df['Latitude'] != 0].reset_index(drop=True)
        self.df = self._df[self._df['Longitude'] != 0].reset_index(drop=True)
        # Encontrar indice del primer dato dentro del agua basándonos en conductividad distinta de cero + margen de estabilización
        df_cond = self.df[self.df['Sensor'] == 'Conductivity'].reset_index(drop=True)
        ini_index = df_cond[df_cond['Data'] != 0].index[0] + 10

        # Convertir las coordenadas GPS a metros respecto al inicio de la imagen con la función haversine
        self.df['Y'] = self.df.apply(lambda row: self.haversine(self.map_coords['lat_min'], self.map_coords['lon_min'], row['Latitude'], self.map_coords['lon_min']), axis=1)
        self.df['X'] = self.df.apply(lambda row: self.haversine(self.map_coords['lat_min'], self.map_coords['lon_min'], self.map_coords['lat_min'], row['Longitude']), axis=1)

        # Guardar el DataFrame en un archivo CSV con la fecha
        if(not os.path.exists("sql_data")):
            os.makedirs("sql_data")
        self.df.to_csv(f'sql_data/wqp_{self.date}.csv', index=False)

        print(self.df)
        self.busy=False


        # # WQPs de mediciones
        # sensores = [*self._df['Sensor'].unique()]

        # if 'Sensor_battery' in sensores:
        #     sensores.remove('Sensor_battery')
        # if 'Battery' in sensores:
        #     sensores.remove('Battery')

    def print_measures_on_map(self,sensor="Conductivity"):

        if sensor not in self.sensors:
            print("sensor not in database")
            return


        sensor_df = self.df[self.df['Sensor'] == sensor].reset_index(drop=True)

        # Seleccionar rango de la misión
        condition = (sensor_df['Date'] > '2024-02-27 11:46:16') & (sensor_df['Date'] < '2024-02-27 13:18:16')
        sensor_df = sensor_df[condition].reset_index(drop=True)
        # Eliminar muestras en un rango de tiempo
        rango_perdida_gps = (sensor_df['Date'] > f'2024-02-27 12:42:16') & (sensor_df['Date'] < f'2024-02-27 12:43:16')
        rango_paso_puente= (sensor_df['Date'] > f'2024-02-27 12:37:16') & (sensor_df['Date'] < f'2024-02-27 12:39:16')
        rango_paso_puente_vuelta= (sensor_df['Date'] > f'2024-02-27 12:51:26') & (sensor_df['Date'] < f'2024-02-27 12:51:38')
        rango_ramas = (sensor_df['Date'] > f'2024-02-27 13:00:00') & (sensor_df['Date'] < f'2024-02-27 13:10:00')
        suma_rangos = rango_perdida_gps | rango_paso_puente | rango_paso_puente_vuelta | rango_ramas
        sensor_df = sensor_df[~suma_rangos]
        # sensor_df = sensor_df[rango_tiempo].reset_index(drop=True)

        sensor_df[condition]

        # Convertir las coordenadas GPS a metros respecto al inicio de la imagen
        # img = plt.imread('AlamilloImage.png')
        p0 = utm.from_latlon(self.map_coords['lat_min'], self.map_coords['lon_min'])
        # p1 = utm.from_latlon(map_coords['lat_max'], map_coords['lon_max'], force_zone_number=p0[2])
        p1_x = self.haversine(self.map_coords['lat_min'], self.map_coords['lon_min'], self.map_coords['lat_min'], self.map_coords['lon_max'])
        p1_y = self.haversine(self.map_coords['lat_min'], self.map_coords['lon_min'], self.map_coords['lat_max'], self.map_coords['lon_min'])
        p1 = (p0[0] + p1_x, p0[1] + p1_y)

        x0_extend = p0[0] - self.pos_ini[0]
        y0_extend = p0[1] - self.pos_ini[1]
        x1_extend = p1[0] - self.pos_ini[0]
        y1_extend = p1[1] - self.pos_ini[1]


        # Pintar el mapa de latitud y longitud
        plt.imshow(self.satelite_img, extent=[self.map_coords['lon_min'], self.map_coords['lon_max'], self.map_coords['lat_min'], self.map_coords['lat_max']]) # extend=[xmin,xmax,ymin,ymax]
        plt.scatter(sensor_df['Longitude'].to_numpy(), sensor_df['Latitude'].to_numpy(), c='r', s=0.3, marker='.')
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # plt.savefig('ruta.svg')
        plt.show()

        # Pintra el mapa de UTM
        plt.imshow(self.scenario_map, extent=[x0_extend, x1_extend, y0_extend, y1_extend])
        plt.scatter(sensor_df['X'].to_numpy(), sensor_df['Y'].to_numpy(), c='r', s=7, marker='.')
        plt.show()


        # Dimensiones mapa
        rows, cols = self.scenario_map.shape

        # Resolución de la matriz
        res_lat, res_lon = (self.map_coords['lat_max'] - self.map_coords['lat_min']) / rows, (self.map_coords['lon_max'] - self.map_coords['lon_min']) / cols


        # Ejemplo de uso para una muestra con coordenadas GPS
        lat_sample, lon_sample = self.df['Latitude'].astype(float).iloc[100], self.df['Longitude'].astype(float).iloc[100]
        row_idx, col_idx = self.gps_to_matrix_idx(lat_sample, lon_sample, self.map_coords['lat_max'], self.map_coords['lon_min'], res_lat, res_lon)

        print(f"Image shape: ({rows}, {cols})")
        print(f"Resolution: ({res_lat}, {res_lon})")
        print(f"Coordinates in the map: ({row_idx}, {col_idx})")


    def get_gp(self, sensor="Conductivity"):

        if sensor not in self.sensors:
            print("sensor not in database")
            return
        
        sensor_df = self.df[self.df['Sensor'] == sensor].reset_index(drop=True)

        # Seleccionar rango de la misión
        sensor_df = sensor_df[(sensor_df['Date'] > '2024-02-27 11:46:16') & (sensor_df['Date'] < '2024-02-27 13:18:16')].reset_index(drop=True)

        # Eliminar muestras en un rango de tiempo
        rango_perdida_gps = (sensor_df['Date'] > f'2024-02-27 12:42:16') & (sensor_df['Date'] < f'2024-02-27 12:43:16')
        rango_paso_puente= (sensor_df['Date'] > f'2024-02-27 12:37:16') & (sensor_df['Date'] < f'2024-02-27 12:39:16')
        rango_paso_puente_vuelta= (sensor_df['Date'] > f'2024-02-27 12:51:26') & (sensor_df['Date'] < f'2024-02-27 12:51:38')
        rango_ramas = (sensor_df['Date'] > f'2024-02-27 13:00:00') & (sensor_df['Date'] < f'2024-02-27 13:10:00')
        suma_rangos = rango_perdida_gps | rango_paso_puente | rango_paso_puente_vuelta | rango_ramas
        sensor_df = sensor_df[~suma_rangos]
        # sensor_df = sensor_df[rango_tiempo].reset_index(drop=True)

        # Quitar las medidas con valor cero
        sensor_df = sensor_df[sensor_df['Data'] != 0].reset_index(drop=True)

        # Limpiar outliers
        # Q1 = sensor_df['Data'].quantile(0.25)
        # Q3 = sensor_df['Data'].quantile(0.75)
        # IQR = Q3 - Q1
        # sensor_df = sensor_df[~((sensor_df['Data'] < (Q1 - 3 * IQR)) | (sensor_df['Data'] > (Q3 + 3 * IQR)))]


        
        # Recortar a una de cada n muestras
        n = 8
        latitudes = np.array(sensor_df['Latitude'].astype(float))[::n]
        longitudes = np.array(sensor_df['Longitude'].astype(float))[::n]
        y, x = zip(*[self.gps_to_matrix_idx(lat, lon, self.map_coords['lat_max'], self.map_coords['lon_min'], self.res_lat, self.res_lon) for lat, lon in zip(latitudes, longitudes)])
        # y, x = np.array(sensor_df['Y'].astype(float))[::n], np.array(sensor_df['X'].astype(float))[::n]
        xy = list(zip(y, x)) # Pares de coordenadas GPS

        if sensor == 'Sonar':
            # Pasar a metros
            data = np.array(sensor_df['Data'].astype(float))[::n] / 1000
        else:
            data = np.array(sensor_df['Data'].astype(float))[::n]
            
        gaussian_process = GaussianProcessGPyTorch(scenario_map = self.scenario_map, initial_lengthscale = 300, kernel_bounds = (200, 400), training_iterations = 50, scale_kernel=True, device = 'cuda' if torch.cuda.is_available() else 'cpu')
        gaussian_process.fit_gp(X_new=xy, y_new=data, variances_new=[0.005]*len(data))
        mean_map, uncertainty_map = gaussian_process.predict_gt()
        return mean_map, uncertainty_map, x, y

    def plot_mean(self, mean_map, sensor, x, y):
        fig, axis = plt.subplots()
        # Punto de despliegue
        # plt.text(350, 1100, 'Punto de despliegue', fontsize=9, rotation=0, ha='center', va='center', color='w')
        # plt.scatter(175, 1050, c='r', s=50, marker='X', zorder=2)
        plt.xticks([])
        plt.yticks([])

        # Contorno
        cs_internos = plt.contour(mean_map, colors='black', alpha=0.7, linewidths=0.7, zorder=1)
        cs_externo = plt.contour(mean_map, colors='black', alpha=1, linewidths=1.7, zorder=1)

        cs_internos.collections[0].remove()
        for i in range(1, len(cs_externo.collections)):
            cs_externo.collections[i].remove()
        plt.clabel(cs_internos, inline=1, fontsize=3.5)

        # Mapa y puntos de muestreo
        plt.scatter(x, y, c='black', s=1, marker='.', alpha=0.5)
        # vmin_dict = {'Sonar': 2, 'Conductivity': 2.29, 'PH': 7.48, 'Temperature': 17.1, 'Turbidity': 30}
        # vmax_dict = {'Sonar': 0.5, 'Conductivity': 2.14, 'PH': 7.16, 'Temperature': 14.50, 'Turbidity': 15}
        # plt.imshow(mean_map, cmap='viridis', alpha=1, origin='upper', vmin=vmin_dict[sensor], vmax=vmax_dict[sensor])
        vmin = np.min(mean_map[mean_map > 0])
        vmax = np.max(mean_map[mean_map > 0])
        plt.imshow(mean_map, cmap='viridis', alpha=1, origin='upper', vmin=vmin, vmax=vmax)

        # Recortar el mapa
        if self.selected_map == 'alamillo':
            plt.ylim(1150, 200)

        # Leyendas
        unidades_dict = {'Sonar': 'Profundidad (m)', 'Conductivity': 'Conductividad (mS/cm)', 'PH': 'pH', 'Temperature': 'Temperatura (ºC)', 'Turbidity': 'Turbidez (NTU)'}
        nombre_dict = {'Sonar': 'Batimetría', 'Conductivity': 'Conductividad', 'PH': 'pH', 'Temperature': 'Temperatura', 'Turbidity': 'Turbidez'}
        plt.colorbar(shrink=0.65).set_label(label=unidades_dict[sensor],size=12)#,weight='bold')
        if self.selected_map == 'alamillo':
            # plt.text(1950, 650, unidades_dict[sensor], fontsize=12, rotation=90, ha='center', va='center', color='k')
            plt.title(f'{nombre_dict[sensor]} del Lago Mayor (Parque del Alamillo)')
        if(not os.path.exists("outs")):
            os.makedirs("outs")
        savepath=self.resource_path(f"outs/{nombre_dict[sensor]}_{self.selected_map}.pdf")
        plt.savefig(savepath, format='pdf')
        # plt.show()



    def create_map(self, mean_map, sensor, x, y):
        fig, axis = plt.subplots()
        # Punto de despliegue
        # plt.text(350, 1100, 'Punto de despliegue', fontsize=9, rotation=0, ha='center', va='center', color='w')
        # plt.scatter(175, 1050, c='r', s=50, marker='X', zorder=2)
        plt.xticks([])
        plt.yticks([])

        # Contorno
        cs_internos = plt.contour(mean_map, colors='black', alpha=0.7, linewidths=0.7, zorder=1)
        cs_externo = plt.contour(mean_map, colors='black', alpha=1, linewidths=1.7, zorder=1)

        cs_internos.collections[0].remove()
        for i in range(1, len(cs_externo.collections)):
            cs_externo.collections[i].remove()
        plt.clabel(cs_internos, inline=1, fontsize=3.5)

        # Mapa y puntos de muestreo
        plt.scatter(x, y, c='black', s=1, marker='.', alpha=0.5)
        # vmin_dict = {'Sonar': 2, 'Conductivity': 2.29, 'PH': 7.48, 'Temperature': 17.1, 'Turbidity': 30}
        # vmax_dict = {'Sonar': 0.5, 'Conductivity': 2.14, 'PH': 7.16, 'Temperature': 14.50, 'Turbidity': 15}
        # plt.imshow(mean_map, cmap='viridis', alpha=1, origin='upper', vmin=vmin_dict[sensor], vmax=vmax_dict[sensor])
        vmin = np.min(mean_map[mean_map > 0])
        vmax = np.max(mean_map[mean_map > 0])
        plt.imshow(mean_map, cmap='viridis', alpha=1, origin='upper', vmin=vmin, vmax=vmax)

        # Recortar el mapa
        if self.selected_map == 'alamillo':
            plt.ylim(1150, 200)

        # Leyendas
        unidades_dict = {'Sonar': 'Profundidad (m)', 'Conductivity': 'Conductividad (mS/cm)', 'PH': 'pH', 'Temperature': 'Temperatura (ºC)', 'Turbidity': 'Turbidez (NTU)'}
        nombre_dict = {'Sonar': 'Batimetría', 'Conductivity': 'Conductividad', 'PH': 'pH', 'Temperature': 'Temperatura', 'Turbidity': 'Turbidez'}
        plt.colorbar(shrink=0.65).set_label(label=unidades_dict[sensor],size=12)#,weight='bold')
        if self.selected_map == 'alamillo':
            # plt.text(1950, 650, unidades_dict[sensor], fontsize=12, rotation=90, ha='center', va='center', color='k')
            plt.title(f'{nombre_dict[sensor]} del Lago Mayor (Parque del Alamillo)')
        # if(not os.path.exists("outs")):
        #     os.makedirs("outs")
        # savepath=self.resource_path(f"outs/{nombre_dict[sensor]}_{self.selected_map}.pdf")
        # plt.savefig(savepath, format='pdf')
        # plt.show()

        self.image_to_tk=PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        

    def plot_uncertainty(self, uncertainty_map, sensor):
        non_water_mask = self.scenario_map == 0
        uncertainty_map[non_water_mask] = np.nan
        plt.imshow(uncertainty_map, cmap='gray', alpha=1, origin='upper')
        plt.xticks([])
        plt.yticks([])
        # Recortar el mapa
        if self.selected_map == 'alamillo':
            plt.ylim(1150, 200)
        plt.colorbar(shrink=0.65)
        nombre_dict = {'Sonar': 'Batimetría', 'Conductivity': 'Conductividad', 'PH': 'pH', 'Temperature': 'Temperatura', 'Turbidity': 'Turbidez'}
        if self.selected_map == 'alamillo':
            plt.title(f'Desviación típica de {nombre_dict[sensor]}')
        if(not os.path.exists("outs")):
            os.makedirs("outs")
        savepath=self.resource_path(f"outs/{nombre_dict[sensor]}_{self.selected_map}_std.pdf")
        plt.savefig(savepath, format='pdf')
        # plt.show()

    def haversine(self, lat1, lon1, lat2, lon2):
        # Radio de la Tierra en kilómetros
        R = 6371.0
        
        # Convertir coordenadas de grados a radianes
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        
        # Diferencias de coordenadas
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        # Fórmula del haversine
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        # Distancia total en kilómetros
        distance = R * c * 1000
        return distance
    
    def gps_to_matrix_idx(self, lat, lon, lat_max, lon_min, res_lat, res_lon):

        row_idx = int((lat_max - lat) / res_lat)
        col_idx = int((lon - lon_min) / res_lon)

        # Limitar los índices dentro de los rangos válidos de la matriz
        # row_idx = max(0, min(row_idx, rows - 1))
        # col_idx = max(0, min(col_idx, cols - 1))

        return row_idx, col_idx
    
    def obtain_prediction_maps(self):
        # Obtener todos los mapas de predicción
        for sensor in self.sensors: # ['Conductivity', 'PH', 'Sonar', 'Temperature', 'Turbidity']
            mean_map, uncertainty_map, x, y = self.get_gp(sensor)
            # self.plot_mean(mean_map, sensor, x, y)
            # self.plot_uncertainty(uncertainty_map, sensor)
            self.create_map(mean_map, sensor, x, y)
    
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)
    
