import librosa
from modelscope import AutoProcessor, Qwen2AudioForConditionalGeneration
import os

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen2-Audio-7B" ,trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen2-Audio-7B" ,trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>请尽可能详细地描述你听到的语音，包括但不限于情感、语气、语言、音色等。"

for audio_name in os.listdir("data/meld"):
    audio_path = os.path.join("data/meld", audio_name)
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=prompt, audios=audio, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(audio_name, ":", response)