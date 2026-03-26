from kittentts import KittenTTS
import soundfile as sf

m = KittenTTS('KittenML/kitten-tts-mini-0.8', language='ko')
audio = m.generate('안녕하세요, 임성구입니다.', voice='Bruno')
sf.write('test_ko.wav', audio, 24000)
print('done')
