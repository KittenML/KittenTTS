import os
import sys
import threading
import warnings
import winsound
from tkinter import filedialog
import customtkinter as ctk

from kittentts import get_model, KittenTTS

# Ignore warnings from some dependencies
warnings.filterwarnings("ignore")

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class KittenTTSApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("KittenTTS - Ultra-Lightweight Voice Generator")
        self.geometry("700x550")
        self.minsize(600, 500)
        
        self.current_model_id = "KittenML/kitten-tts-nano-0.1"
        self.tts_model = None
        self.output_file = "output.wav"

        # --- Grid Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        # --- Header ---
        self.header_label = ctk.CTkLabel(
            self.main_frame, 
            text="🐈 KittenTTS Generator", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nw")

        # --- Controls Frame ---
        self.controls_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Model Selection
        self.model_label = ctk.CTkLabel(self.controls_frame, text="Model:")
        self.model_label.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="w")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.controls_frame, 
            values=["KittenML/kitten-tts-nano-0.1", "KittenML/kitten-tts-mini-0.8"],
            command=self.change_model
        )
        self.model_dropdown.grid(row=1, column=0, padx=(0, 10), pady=5, sticky="ew")

        # Voice Selection
        self.voice_label = ctk.CTkLabel(self.controls_frame, text="Voice:")
        self.voice_label.grid(row=0, column=1, padx=10, pady=0, sticky="w")
        self.voice_dropdown = ctk.CTkOptionMenu(
            self.controls_frame, 
            values=["Loading..."]
        )
        self.voice_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Speed Selection
        self.speed_label = ctk.CTkLabel(self.controls_frame, text="Speed: 1.0x")
        self.speed_label.grid(row=0, column=2, padx=(10, 0), pady=0, sticky="w")
        self.speed_slider = ctk.CTkSlider(
            self.controls_frame, 
            from_=0.5, to=2.0, 
            number_of_steps=15,
            command=self.update_speed_label
        )
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=1, column=2, padx=(10, 0), pady=5, sticky="ew")

        # --- Text Input ---
        self.text_label = ctk.CTkLabel(self.main_frame, text="Text to Synthesize:")
        self.text_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.text_input = ctk.CTkTextbox(self.main_frame, height=150, font=ctk.CTkFont(size=14))
        self.text_input.grid(row=3, column=0, padx=20, pady=(5, 20), sticky="nsew")
        self.text_input.insert("0.0", "Welcome to KittenTTS! Type something here and click Generate to hear it spoken out loud.")

        # --- Action Buttons Frame ---
        self.action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.action_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.action_frame.grid_columnconfigure(0, weight=1)

        # Status Label
        self.status_label = ctk.CTkLabel(self.action_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Generate Button
        self.generate_btn = ctk.CTkButton(
            self.action_frame, 
            text="Generate & Play ▶", 
            command=self.generate_audio,
            height=40,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.generate_btn.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="e")

        # Export Button
        self.export_btn = ctk.CTkButton(
            self.action_frame, 
            text="Export Audio 💾", 
            command=self.export_audio,
            height=40,
            fg_color="transparent",
            border_width=2,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.export_btn.grid(row=0, column=2, padx=(10, 0), pady=10, sticky="e")

        # Load initial model asynchronously
        threading.Thread(target=self.load_model_bg, args=(self.current_model_id,), daemon=True).start()

    def update_speed_label(self, value):
        self.speed_label.configure(text=f"Speed: {value:.1f}x")

    def change_model(self, selected_model):
        if selected_model == self.current_model_id:
            return
        
        self.current_model_id = selected_model
        # Disable generation while loading
        self.generate_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.status_label.configure(text=f"Status: Downloading/Loading {selected_model}...")
        self.voice_dropdown.configure(values=["Loading..."])
        self.voice_dropdown.set("Loading...")
        
        threading.Thread(target=self.load_model_bg, args=(selected_model,), daemon=True).start()

    def load_model_bg(self, model_id):
        self.update_status(f"Loading '{model_id}'...")
        self.generate_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        
        try:
            model = KittenTTS(model_name=model_id)
            if model.model is None:
                raise Exception("Failed to load model architecture.")
            
            self.tts_model = model
            
            # Extract aliases for dropdown
            if hasattr(self.tts_model.model, 'voice_aliases'):
                aliases = list(self.tts_model.model.voice_aliases.keys())
            else:
                aliases = ["Default"]

            # Update UI from main thread
            self.after(0, self.update_voices_ui, aliases)
            
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", err=True)
            self.after(0, lambda: self.generate_btn.configure(state="normal"))
            self.after(0, lambda: self.export_btn.configure(state="normal"))

    def update_voices_ui(self, aliases):
        self.voice_dropdown.configure(values=aliases)
        if aliases:
            self.voice_dropdown.set(aliases[1] if len(aliases) > 1 else aliases[0]) # Default to Jasper if possible
        self.update_status("Model loaded. Ready to generate.")
        self.generate_btn.configure(state="normal")
        self.export_btn.configure(state="normal")

    def update_status(self, text, err=False):
        color = "red" if err else "gray"
        self.after(0, lambda: self.status_label.configure(text=f"Status: {text}", text_color=color))

    def generate_audio(self):
        text = self.text_input.get("0.0", "end").strip()
        if not text:
            self.update_status("Error: Input text is empty.", err=True)
            return

        voice = self.voice_dropdown.get()
        speed = self.speed_slider.get()

        if not self.tts_model:
            self.update_status("Error: Model not loaded yet.", err=True)
            return

        self.generate_btn.configure(state="disabled", text="Generating...")
        self.export_btn.configure(state="disabled")
        self.update_status("Generating speech audio...")

        # Run in background to prevent UI freeze
        threading.Thread(target=self.generate_audio_bg, args=(text, voice, speed), daemon=True).start()

    def generate_audio_bg(self, text, voice, speed):
        try:
            self.tts_model.generate_to_file(text, self.output_file, voice=voice, speed=speed)
            self.update_status("Audio generated successfully. Playing...")
            
            # Play sound using built-in Windows sound engine
            # SND_FILENAME: The sound parameter is the name of a WAV file
            # SND_ASYNC: The sound is played asynchronously, and the function returns immediately after beginning the sound
            winsound.PlaySound(self.output_file, winsound.SND_FILENAME | winsound.SND_ASYNC)

            # Re-enable button
            self.after(0, lambda: self.generate_btn.configure(state="normal", text="Generate & Play ▶"))
            self.after(0, lambda: self.export_btn.configure(state="normal"))
            self.update_status("Ready")

        except Exception as e:
            self.update_status(f"Generation error: {str(e)}", err=True)
            self.after(0, lambda: self.generate_btn.configure(state="normal", text="Generate & Play ▶"))
            self.after(0, lambda: self.export_btn.configure(state="normal"))

    def export_audio(self):
        text = self.text_input.get("0.0", "end").strip()
        if not text:
            self.update_status("Error: Input text is empty.", err=True)
            return

        voice = self.voice_dropdown.get()
        speed = self.speed_slider.get()

        if not self.tts_model:
            self.update_status("Error: Model not loaded yet.", err=True)
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav"), ("All Files", "*.*")],
            title="Export Audio As"
        )
        
        if not save_path:
            return

        self.generate_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled", text="Exporting...")
        self.update_status(f"Exporting speech audio to {os.path.basename(save_path)}...")

        threading.Thread(target=self.export_audio_bg, args=(text, voice, speed, save_path), daemon=True).start()

    def export_audio_bg(self, text, voice, speed, save_path):
        try:
            self.tts_model.generate_to_file(text, save_path, voice=voice, speed=speed)
            self.update_status(f"Successfully exported to {os.path.basename(save_path)}")
            
            # Re-enable buttons
            self.after(0, lambda: self.generate_btn.configure(state="normal"))
            self.after(0, lambda: self.export_btn.configure(state="normal", text="Export Audio 💾"))

        except Exception as e:
            self.update_status(f"Export error: {str(e)}", err=True)
            self.after(0, lambda: self.generate_btn.configure(state="normal"))
            self.after(0, lambda: self.export_btn.configure(state="normal", text="Export Audio 💾"))

def main():
    app = KittenTTSApp()
    app.mainloop()

if __name__ == "__main__":
    main()
