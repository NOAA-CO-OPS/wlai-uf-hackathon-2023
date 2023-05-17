SITES_FILE="runs/stations.csv"
OUTDIR="out/"
DATADIR="data/"
EPOCHS=4

header=1
while read row; do

  if [ $header -eq 1 ]; then
    header=0
    continue
  fi

  # Extract station ID
  station=$(echo -n $row | awk -F',' '{printf "%d", $5}')
  
  # Create python run
  echo "python train.py -s ${station} -m $OUTDIR/model_${site}.hdf5 -l $OUTDIR/trainlog_${site}.csv -e $EPOCHS"
done < $SITES_FILE
