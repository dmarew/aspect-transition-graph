#!/bin/bash
echo "Generatin fake dataset ...."
python generate_fake_dataset.py
echo "training autoencoder ..."
python train_autoencoder.py
echo "extracting autoencoder features ..."
python feature_extraction.py
echo "clusting over observatons ..."
python hierarchical_clustering.py
echo "train aspect transition model"
python train_aspect_transition.py
