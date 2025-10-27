import pandas as pd
import numpy as np
import mat73
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def power_law(x, a, b):
    return a * np.power(x, b)

def michaelis_menten(x, a, b, c):
    return (a * x) / (b + x) + c

modelos = {
    'michaelis_menten': (michaelis_menten, 3),  # 3 parámetros
    'power_law': (power_law, 2)           # 2 parámetros
}

def vecinos_infinito(x0, y0):
    yield (x0, y0)  # incluir el punto central primero
    capa = 1
    while True:
        x, y = x0 - capa, y0 - capa
        # lado superior (de izquierda a derecha)
        for i in range(2 * capa):
            yield (x + i, y)
        # lado derecho (de arriba a abajo)
        for i in range(2 * capa):
            yield (x + 2 * capa, y + i)
        # lado inferior (de derecha a izquierda)
        for i in range(2 * capa):
            yield (x + 2 * capa - i, y + 2 * capa)
        # lado izquierdo (de abajo a arriba)
        for i in range(2 * capa):
            yield (x, y + 2 * capa - i)
        capa += 1

#We save the dataframe with the samples
resultado=pd.read_csv('resultado_final_muestras.csv')

n_areas=np.unique(resultado['OBJECTID'])

n_areas=sorted(n_areas)

print(n_areas)

#We load the mask of the regions, where in each position says which number corresponds to that area
data=np.load('mask.npz')
mask=data["mask"]

area_grid = np.load('area_grid.npy')

#We load the mask of the latitude, longitude and the mask of the land, shelf and ocean
data_Mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
#print(data_Mask.keys())

landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']

#We create the vectors of latitude and longitude
lon_grid, lat_grid = np.meshgrid(landShelfOcean_Lon, landShelfOcean_Lat)

#We create the coordinates matrix
coords = np.stack((lon_grid, lat_grid), axis=-1)


fig1, axs1 = plt.subplots(5, 7)
fig2, axs2 = plt.subplots(5, 7)

output=[]
for K in range(len(n_areas)):

    print("Area", n_areas[K])
        
    if n_areas[K]!=47 and n_areas[K]!=64 and K>35:

        A=np.where(mask[K,:,:]==1)

        combined_A = np.stack((A[0], A[1]), axis=-1)

        #select a random point from the combined_A
        
        print(len(combined_A))

        #Select the dataframe that correspond to the area
        df_area=resultado[resultado["OBJECTID"]==n_areas[K]]

        #Calculate the number of unique samples in that area
        muestras_area = df_area['muestra'].nunique() 

        area=np.zeros((3,len(combined_A)))
        num_generos=np.zeros((3,len(combined_A)))



        for m in range(3):
            print(m)
            index1= np.random.randint(0, len(combined_A)-1)

            #Select the lat and lon of the selected point
            lon_fix=coords[combined_A[index1][0], combined_A[index1][1], 0]
            lat_fix=coords[combined_A[index1][0], combined_A[index1][1], 1]

            #Initialize the number of remaining samples and the number of samples from previous iterations
            muestras_restantes = 0
            post_muestras=0

            #Create arrays to save the area and the number of genera


            muestras_unicas_total=set()
            muestras_select_total=set()

            i_select=0

            #It iterates over the neighbors of the selected point following an spiral pattern
            for i, (x, y) in enumerate(vecinos_infinito(0, 0)):
                
                if i_select%100==0 and len(combined_A)>400:
                    print(i_select)
                #It selects the actual lat and lon of the point, adding 0.5 degrees for each step in x and y
                lat_act=lat_fix + y*0.5

                lon_act=lon_fix + x*0.5
                if lon_act>180:
                    lon_act=lon_act-360
                elif lon_act<-180:
                    lon_act=lon_act+360

                #It selects all the index of the point in the combined array
                index_grid = np.where((coords[:,:,0]==lon_act) & (coords[:,:,1]==lat_act))#????

                #If there are any point in the combined array that corresponds to the lat and lon selected
                if len(index_grid[0])>0 and len(index_grid[1])>0:

                    #We check if the point is in the area we want to study
                    if mask[K,index_grid[0][0], index_grid[1][0]] == 1:

                        #We take the points in the area that are inside that grid cell
                        masque = (
                            (df_area['decimalLatitude'] >= lat_act-0.25) &
                            (df_area['decimalLatitude'] <= lat_act+0.25) &
                            (df_area['decimalLongitude'] >= lon_act-0.25) &
                            (df_area['decimalLongitude'] <= lon_act + 0.25)
                        )

                        #We create a new dataframe with the selected points
                        df_grid=df_area[masque]

                        #Inside that dataframe, we select the unique samples
                        muestras_unicas = set(df_grid['muestra'].unique())
                        muestras_unicas=list(muestras_unicas-muestras_unicas_total)
                        muestras_unicas_total.update(muestras_unicas)

                        #We calculate the number of samples to get in that area following the equation explained above
                        n_muestras_total = int(50*(area_grid[index_grid[0][0], index_grid[1][0]]/10**6)/3083)

                        #If the number of samples to get is greater than the number of unique samples in that area
                        #Select all the samples and add the remaining samples to the next iteration
                        if n_muestras_total>len(muestras_unicas):
                            n_muestras = len(muestras_unicas)
                            muestras_restantes += (n_muestras_total - len(muestras_unicas))
                        #If the number of samples to get plus the remaining samples from the previous iteration is greater than the number of unique samples in that area
                        #Select all the samples and subtract the number of samples selected from the remaining samples
                        elif n_muestras_total+muestras_restantes>len(muestras_unicas):
                            n_muestras=len(muestras_unicas)
                            muestras_restantes -= (len(muestras_unicas)-n_muestras_total)
                        #Else, if the number of samples to get plus the remaining samples from the previous iteration is less than the number of unique samples in that area
                        #Select the number of samples to get plus the remaining samples and set the remaining samples to 0
                        else:
                            n_muestras=n_muestras_total+muestras_restantes
                            muestras_restantes=0

                        post_muestras+=n_muestras
                        #print("muestras_area",len(muestras_unicas))
                        #print("n_muestras_total",n_muestras_total)
                        #print("nmuestras",n_muestras)
                        #print("n muestras + pasado", post_muestras)
                        #print("muestras_restantes",muestras_restantes)

                        #we select the specific number of samples inside the selected samples
                        muestras_seleccionadas = np.random.choice(muestras_unicas, n_muestras, replace=False)
                        muestras_select_total.update(muestras_seleccionadas)
                        #print("muestras seleccionadas totales", len(muestras_select_total))
                        #print("muestras_seleccionadas", len(muestras_seleccionadas))

                        #In the dataframe, we select the samples that are in the selected samples
                        df_muestra = df_area[df_area['muestra'].isin(muestras_select_total)]
                        #print(df_muestra.shape)
                        #input("check")

                        #Then, we calculate the number of unique genera in the selected samples
                        n_generos = df_muestra['genusid'].nunique() 
                        #print(n_generos)
                        num_generos[m,i_select]=n_generos
                        #input("check")

                        if i_select == 0:
                            area[m,0]=area_grid[index_grid[0][0], index_grid[1][0]]/10**6
                            #muestras_unicas_total=len(muestras_unicas)

                        else:
                            area[m,i_select]=area[m,i_select-1]+area_grid[index_grid[0][0], index_grid[1][0]]/10**6
                            #muestras_unicas_total+=len(muestras_unicas)

                        #print(f"Iteration { i_select} de {len(combined_A)}")
                        #print("Muestras procesadas", len(muestras_unicas_total))
                        #print(muestras_unicas_total)
                        #print(muestras_unicas)
                        #print(lat_act, lon_act)
                        #print("Muestras totales", muestras_area)
                        #print("--------------------------------")
                        #input("check")
                        i_select+=1

                        if i_select == len(combined_A):
                            break

        print(f"{len(muestras_unicas_total)} de {muestras_area}")

        # Ajustar todos los modelos y calcular su RMSE
        best_rmse = np.inf
        best_model = None
        best_params = None


        try:
            params, _ = curve_fit(power_law, area.ravel(), num_generos.ravel())
            y_estim = power_law(area.ravel(), *params)
            y_pred = power_law(np.arange(0,np.max(area),50), *params)
            rmse = np.sqrt(mean_squared_error(num_generos.ravel(), y_estim))
            print("Interpolada correctamente")
        except RuntimeError:
            continue  # Si no converge el ajuste, pasa al siguiente modelo

        #for nombre, (modelo, n_param) in modelos.items():
            #bounds = (0, [np.inf] * n_param)
            #params, _ = curve_fit(modelo, area.ravel(), num_generos.ravel(), bounds=bounds)
            #y_pred = modelo(area.ravel(), *params)
            #rmse = np.sqrt(mean_squared_error(num_generos.ravel(), y_pred))

            #if rmse < best_rmse:
            #    best_rmse = rmse
            #    best_model = modelo
            #    best_params = params
            #    best_nombre = nombre

        #print(best_params)
        #print(best_nombre)
        #
        #y_pred = best_model(np.arange(0,np.max(area),50), *best_params)



        #output.append({'region': K, 
        #    'area': area,
        #    'n_generos': num_generos
        #})

            #print(output)


        if K<35:
            n=K
        
            axs1[K%5, int(K/5)].scatter(area.ravel(), num_generos.ravel(), color='blue', s=0.5, alpha=0.6)
            try:
                axs1[K%5, int(K/5)].plot(np.arange(0,np.max(area),50), y_pred, color='red')
            except:
                pass
            if K%5==0:
                axs1[K%5, int(K/5)].set_xlabel('Area (km²)')
            if int(K/5)==0:
                axs1[K%5, int(K/5)].set_ylabel('Number of genera')
            axs1[K%5, int(K/5)].set_title(f'{n_areas[K]}')

        else: 
            n=K-35
            axs2[n%5, int(n/5)].scatter(area.ravel(), num_generos.ravel(), color='blue', s=0.5, alpha=0.6)
            try:
                axs2[K%35, int(K/35)].plot(np.arange(0,np.max(area),50), y_pred, color='red')
            except:
                pass
            if n%5==0:
                axs2[n%5, int(n/5)].set_xlabel('Area (km²)')
            if int(n/5)==0:
                axs2[n%5, int(n/5)].set_ylabel('Number of genera')
            axs2[n%5, int(n/5)].set_title(f'{n_areas[K]}')

#fig1.savefig('Genera_vs_Area_1-35.png')
fig2.savefig('Genera_vs_Area_36-70.png')




