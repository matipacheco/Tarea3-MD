# Tarea3-MD
Tarea 3 Minería de Datos (Sistemas Recomendadores)

Estando en el directorio YOUR_SPARK_HOME, compilar de la siguiente manera:

bin/spark-submit YOUR_SPARK_HOME/recomendations.py rank lambda_

, donde rank y lamnda_ son los distintos valores con los que se experimentará

Luego, con el script anterior ejecutado, se debe ejecutar el siguiente script para obtener las top10 recomendaciones:

bin/spark-submit YOUR_SPARK_HOME/top10recomendations.py
