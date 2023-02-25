import os
import csv
import cv2
import pytesseract
from Levenshtein import distance

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set the path to the dataset
dataset_path = r"C:\Users\Vitoria\Desktop\mltu-main (copy)\Word Extraction\word extraction\Datasets\Testing"

# Create lists to store the CER values and the image filenames
cer_values = []
accuracy_values = []
image_filenames = []

# Loop over all the images in the dataset directory
for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        # Load the image and convert it to grayscale
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use pytesseract to extract the text from the image
        text = pytesseract.image_to_string(gray)

        # Get the label from the filename
        label = os.path.splitext(filename)[0]

        # Calculate the CER and append it to the list of CER values
        cer_value = distance(text, label) / max(len(text), len(label))
        cer_values.append(cer_value)

        # Calculate the accuracy and append it to the list of accuracy values
        accuracy = 1 - cer_value
        accuracy_values.append(accuracy)

        # Append the image filename to the list of image filenames
        image_filenames.append(filename)

# Calculate the average CER and accuracy
avg_cer = sum(cer_values) / len(cer_values)
avg_accuracy = sum(accuracy_values) / len(accuracy_values)

# Print the average CER and accuracy
print("\nAverage CER:", avg_cer)
print("\nAverage Accuracy:", avg_accuracy)

# Write the CER and accuracy values and image filenames to a CSV file
with open("cer_accuracy_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "CER", "Accuracy"])
    for i in range(len(cer_values)):
        writer.writerow([image_filenames[i], cer_values[i], accuracy_values[i]])
