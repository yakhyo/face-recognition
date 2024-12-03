#!/bin/bash

# Define the networks
networks=("sphere20" "sphere36" "sphere64")

# Loop through each network and execute the training
for network in "${networks[@]}"; do
    echo "Starting training for network: $network"
    python train.py --network "$network"
    if [ $? -eq 0 ]; then
        echo "Training completed for network: $network"
    else
        echo "Error occurred during training for network: $network"
        exit 1
    fi
done

echo "All trainings completed successfully."
