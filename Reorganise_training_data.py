cd ~/workspace_bhavana/jersey_number_recognition_20260626/datasets/training_dataset_Ying

mkdir -p images labels

# Copy images
find . -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec cp {} images/ \;

# Copy labels
find . -maxdepth 1 -type f -iname "*.txt" -exec cp {} labels/ \;

echo "Done!"
echo "Images: $(find images -type f | wc -l)"
echo "Labels: $(find labels -type f | wc -l)"
