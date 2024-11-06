from tflite_support import metadata

# Load the TFLite model
model_path = "lstm_2hand.tflite"
displayer = metadata.MetadataDisplayer.with_model_file(model_path)

# Get the metadata as a JSON string
model_metadata = displayer.get_metadata_json()
print("Model Metadata:")
print(model_metadata)

# Get the associated files (e.g., label files)
associated_files = displayer.get_packed_associated_file_list()
print("Associated Files:")
for file_name in associated_files:
    print(file_name)

# Optionally, extract specific associated files
for file_name in associated_files:
    file_content = displayer.get_associated_file_buffer(file_name)
    with open(file_name, "wb") as f:
        f.write(file_content)
        print(f"Extracted {file_name}")
