# TFGTiempoReal
TFG sobre detección a tiempo real de peatones utilizando modelo panóptico y clasificando la spersonas por vulnerabilidades.

Los pesos y configuraciones utilizadas no han sido añadidas en el repositorio, en caso de querer obtenerlas se debe solicitar permiso al autor.

Para poder implementar DeepLabv3plus se ha reutilizado parte de la estructura proporcionada por VainF https://github.com/VainF/DeepLabV3Plus-Pytorch.

Se ha modificado la estructura para adaptarla a la necesidad de detección de escenarios viales, adaptando su implementación para ser utilizado junto a Mapillary, tan solo hay que añadir sus mapeados y propiedades necesarias en la sección de entrenamiento y de visualización en caso de implementar algún otro conjunto.

El Visualizador_Complete.py contiene la ejecución completa del programa esta se sitúa en la carpeta DeepLab. Se sitúa en esa sección debido a que se reutilizan librerías locales de DeepLabv3plus, realmente se podría implemnetar en una carpeta ajena.
