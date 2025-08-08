from gradio_client import Client
import shutil
import os

client = Client("http://localhost:7860/")
result = client.predict(
		text="Welcome to the future of AI-powered speech synthesis!",
		voice="expr-voice-5-f",
		speed=1,
		api_name="/generate_speech"
)

# The result is typically a tuple containing the path to the generated audio file
if result:
    print(f"Raw result: {result}")

    # Handle tuple result - usually the first element is the file path
    if isinstance(result, tuple):
        audio_file_path = result[0]
    else:
        audio_file_path = result

    # Extract filename from the result path
    original_filename = os.path.basename(audio_file_path)

    # Create a new filename in current directory
    current_dir_filename = f"generated_speech_{original_filename}"

    # Copy the file to current directory
    shutil.copy2(audio_file_path, current_dir_filename)
    print(f"Audio saved to: {current_dir_filename}")
    print(f"Original file path: {audio_file_path}")
else:
    print("No audio file generated")