from pathlib import Path
from manim import logger
from manim_voiceover.helper import prompt_ask_missing_package, remove_bookmarks, wav2mp3
from manim_voiceover.services.base import SpeechService
from kokoro import KPipeline
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import torch

def text_to_speech(text: str=None, output_file: str=None, voice: str=None, lang_code: str="a", volume: float=2):
    pipeline = KPipeline(lang_code=lang_code)

    generator = pipeline(
        text, voice,
        speed=1, split_pattern=r'\n+'
    )
    
    for i, (gs, ps, audio) in enumerate(generator):
        # Convert audio to NumPy if it's a Torch tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        # Apply volume adjustment correctly
        audio = np.clip(audio * volume, -1.0, 1.0)  # Ensure it stays within valid range

        # Convert to 16-bit PCM format (standard for WAV)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save audio
        sf.write(output_file, audio_int16, 24000)



class KokoroService(SpeechService):
    """Speech service class for kokoro_self (using text_to_speech)."""
    
    def __init__(self, engine=None, volume: float=1.5, **kwargs):

        """"""
        self.volume = volume
        
        if engine is None:
            engine = text_to_speech

        self.engine = engine
        SpeechService.__init__(self,transcription_model='base', **kwargs)


    def generate_from_text(self, text: str, voice: str, cache_dir: str = None, path: str = None) -> dict:
        """"""
        inner = remove_bookmarks(text)
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_data = {"input_text": text, "service": "kokoro_self"}

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_data_hash(input_data) + ".mp3"
        else:
            audio_path = path

        # Call text_to_speech from kokoro_self and get the output as .wav
        audio_path_str = str(Path(cache_dir) / audio_path.replace(".mp3", ".wav"))
        text_to_speech(inner, output_file=audio_path_str, voice=voice)  # You can change the voice_name as needed.
        # Convert the .wav file to .mp3
        mp3_audio_path = str(Path(cache_dir) / audio_path)
        wav2mp3(audio_path_str, mp3_audio_path)

        # Optionally, remove the original .wav file
        remove_bookmarks(audio_path_str)

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }

        return json_dict
