from kittentts import KittenTTS

# it will run blazing fast on any GPU. But this example will run on CPU.

# Step 1: Load the model
m = KittenTTS("KittenML/kitten-tts-mini-0.8") # 80M version (highest quality)
# m = KittenTTS("KittenML/kitten-tts-micro-0.8") # 40M version (balances speed and quality )
# m = KittenTTS("KittenML/kitten-tts-nano-0.8") # 15M version (tiny and faster )


# Step 2: Generate the audio 

# this is a sample from the TinyStories dataset. 
text ="""One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. """


# available_voices : ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
voice = 'Bruno'



audio = m.generate(text=text, voice=voice )

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
print(f"Audio saved to output.wav")