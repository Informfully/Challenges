SUFFIX=""
# SUFFIX="_large"

source venv/bin/activate

cd yfcc100m
pip install -e .
cd ..

mkdir data
mkdir data/yfcc100m
mkdir data/yfcc100m/meta
python -m yfcc100m.yfcc100m.convert_metadata data/yfcc100m -o data/yfcc100m/meta

mkdir data/yfcc100m/images$SUFFIX
python -m yfcc100m.yfcc100m.download data/yfcc100m/meta/ -o data/yfcc100m/images$SUFFIX/ --num_shards 250
# python -m yfcc100m.yfcc100m.download data/yfcc100m/meta/ -o data/yfcc100m/images$SUFFIX/ --num_shards 500

shard=0
MAX_SHARDS=250
for file in data/yfcc100m/images$SUFFIX/*.zip; do
    if [ $shard -ge $MAX_SHARDS ]; then
        break
    fi
    unzip $file -d data/yfcc100m/images$SUFFIX/
    shard=$((shard + 1))
done