from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-speech", "suno/bark-small")

speech = synthesiser("Hello, my dog is cooler than you!, how are youd oing today, i hope you are fine", forward_params={"do_sample": True})

scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
