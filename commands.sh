# cambiar permisos de directorios
sudo chown -R abelharo:abelharo .

# ver info de cpu y mem jetson
jtop

# abrir VSCode 
sudo code --no-sandbox --user-data-dir /home/abelharo/code/

# ssh a cmts
ssh -X -p 3322 ahararm@cmts1.gap.upv.es

# scp a cmts
scp -P 3322 -r ahararm@cmts1.gap.upv.es:/mnt/beegfs/colab/ahararm/ <dir>

# copiar carpetas de un docker al host
docker cp use-gpu:/ultralytics/runs/detect/train2 ./models/canicas/2024_10_24/train_2024_10_24_yolo11n

sudo jetson_clocks
