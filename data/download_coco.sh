#!/usr/bin/env bash
# Crea cartella dataset
mkdir -p data/coco
cd data/coco

# Scarica immagini Val2017 (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Scarica annotazioni (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip

# Torna alla root
cd ../..