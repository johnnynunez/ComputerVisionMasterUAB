import os
import csv
import glob

def main():
    # Cambiar la ruta al directorio que contiene los archivos CSV
    csv_folder = "./Results/Task3/S04/"

    # Encuentra todos los archivos CSV en el directorio especificado
    csv_files = glob.glob(os.path.join(csv_folder, "tracking_maskflownet_*.pkl.csv"))

    for csv_file in csv_files:
        # Extrae el nombre del archivo CSV (sin la extensión)
        filename = os.path.splitext(os.path.basename(csv_file))[0]

        # Encuentra el identificador (por ejemplo, 'c016') en el nombre del archivo
        identifier = filename.split("_")[-1]
        identifier = identifier.split(".")[0]

        # Define el nombre y la ruta del archivo de salida TXT
        txt_file = os.path.join(csv_folder, f"{identifier}.txt")


        # Lee el archivo CSV y escribe el contenido (sin el encabezado) en el archivo TXT
        with open(csv_file, "r") as input_file, open(txt_file, "w") as output_file:
            csv_reader = csv.reader(input_file)
            next(csv_reader)  # Salta el encabezado del archivo CSV

            for row in csv_reader:
                output_file.write(",".join(row) + "\n")

if __name__ == "__main__":
    main()
