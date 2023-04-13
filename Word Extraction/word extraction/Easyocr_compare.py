import os
import csv
import cv2
import easyocr
from Levenshtein import distance
import matplotlib.pyplot as plt

def wer(predicted_text, ground_truth):
    predicted_words = predicted_text.split()
    ground_truth_words = ground_truth.split()
    wer_value = distance(predicted_words, ground_truth_words) / max(len(predicted_words), len(ground_truth_words))
    return wer_value

# Set the path to the dataset
dataset_path = r"C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Datasets\Testing"

# Create lists to store the CER values, WER values, and image filenames
cer_values = []
wer_values = []
acc_values = []
image_filenames = []

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Loop over all the images in the dataset directory
for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        # Load the image and convert it to grayscale
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use EasyOCR to extract the text from the image
        result = reader.readtext(gray)

        # Get the label from the filename
        label = os.path.splitext(filename)[0]

        # Extract the predicted text from the EasyOCR result
        text = "".join([x[1] for x in result])

        # Calculate the CER and append it to the list of CER values
        cer_value = distance(text, label) / max(len(text), len(label))
        cer_values.append(cer_value)

        # Calculate the WER and append it to the list of WER values
        wer_value = wer(text, label)
        wer_values.append(wer_value)

        # Calculate the accuracy and append it to the list of accuracy values
        accuracy = (1 - cer_value) * 100
        acc_values.append(accuracy)

        # Append the image filename to the list of image filenames
        image_filenames.append(filename)

# Calculate the average CER, WER, and accuracy
avg_cer = sum(cer_values) / len(cer_values)
avg_wer = sum(wer_values) / len(wer_values)
avg_acc = sum(acc_values) / len(acc_values)

# Print the average CER, WER, and accuracy
print("\nAverage CER:", avg_cer)
print("\nAverage WER:", avg_wer)
print("\nAverage accuracy:", avg_acc)

# Write the CER values and image filenames to a CSV file
with open("cer_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "CER"])
    for i in range(len(cer_values)):
        writer.writerow([image_filenames[i], cer_values[i]])

# Write the WER values and image filenames to a CSV file
with open("wer_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "WER"])
    for i in range(len(wer_values)):
        writer.writerow([image_filenames[i], wer_values[i]])

# Write the accuracy values and image filenames to a CSV file
with open("acc_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "Accuracy"])
    for i in range(len(acc_values)):
        writer.writerow([image_filenames[i], acc_values[i]])

plt.plot(image_filenames, cer_values)
plt.title("Character Error Rate (CER) for EasyOCR")
plt.xlabel("Image filename")
plt.xticks(rotation=90)
plt.ylabel("CER")
plt.tight_layout()
plt.savefig("cer_graph_easyocr.png")
plt.show()


plt.plot(image_filenames, wer_values)
plt.title("Word Error Rate (WER) for EasyOCR")
plt.xlabel("Image filename")
plt.xticks(rotation=90)
plt.ylabel("WER")
plt.tight_layout()
plt.savefig("wer_graph_easyocr.png")
plt.show()


plt.plot(image_filenames, acc_values)
plt.title("Accuracy for EasyOCR")
plt.xlabel("Image filename")
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("accuracy_graph_easyocr.png")
plt.show()
