#!/bin/bash

start=`date +%s`
echo "Generatin fake dataset ...."
python generate_fake_dataset.py
echo "training autoencoder ..."
python train_autoencoder.py
echo "extracting autoencoder features ..."
python feature_extraction.py
echo "clusting over observatons ..."
python clustering.py
echo "train aspect transition model"
#python train_aspect_transition.py
end=`date +%s`
runtime=$((end-start))

echo "Runtime was $runtime secs"
