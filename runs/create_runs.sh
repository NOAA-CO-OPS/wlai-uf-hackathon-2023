SITES_FILE="runs/sites.txt"
OUTDIR="out/"
DATADIR="data/"

EPOCHS=4

while read site; do
  echo python train.py \
       -s $site \
       -m $OUTDIR/model_${site}.hdf5 \
       -l $OUTDIR/test-log.csv \
       -e $EPOCHS
done < $SITES_FILE

