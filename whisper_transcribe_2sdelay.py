#!/usr/bin/env python3

import argparse
import tempfile
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)



q = queue.Queue()

Threshold = 0.1     # Minimum volume threshold to activate listening
Vocals = [50, 1000] # Frequency range to detect sounds that could be speech
SampleRate = 44100  # Stream device recording frequency
BlockSize = 30      # Block size in milliseconds
def callback(indata, frames, time, status):
    # print(frames)
    # freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames

    """This is called (from a separate thread) for each audio block."""
    # if  Vocals[0] <= freq <= Vocals[1]:
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
    # q.put((indata.copy(),indata[:,0]))
import time
import sys
import os
import numpy as np
def transcriber():
    import whisper
    model = whisper.load_model("small")
    model.transcribe(audio=np.zeros((int(SampleRate * BlockSize / 1000)), dtype=np.float32))
    device=9
    samplerate=44100
    channels=2
    try:
        if samplerate is None:
            device_info = sd.query_devices(device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            samplerate = int(device_info['default_samplerate'])
            print(int(device_info['default_samplerate']))

        # Make sure the file is opened before recording anything:
    # with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
    #                 channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=samplerate, device=device,
                            channels=channels, callback=callback,blocksize=int(SampleRate * BlockSize / 1000),dtype=np.float32):

            print('#' * 80)
            print('#' * 25+'press Ctrl+C to stop live transcribe'+'#' * 25)
            print('#' * 80)
            last_save_time = time.time()
            audio_buffer = np.zeros((int(SampleRate * BlockSize / 1000), 2))
            # audio_buffer2 = np.zeros((int(SampleRate * BlockSize / 1000), 1))
            # i = 0
            while True:
                current_time = time.time()
                if current_time - last_save_time >= 2:
                    # result = model.transcribe(audio=audio_buffer)
                    # print(result["text"])

                    filename = f"tempaudio.wav"
                    # input_transformed=audio_buffer.flatten().astype(numpy.float32)
                    # print(filename)
                    # print(sys.getsizeof(audio_buffer))
                    sfile=sf.SoundFile(filename,mode='x', samplerate=samplerate,channels=channels)
                    # audio=np.frombuffer(audio_buffer, np.int16).flatten().astype(np.float32) / 32768.0
                    # import torch
                    # abuf=torch.from_numpy(audio)

                    sfile.write(audio_buffer)
                    sfile.close()
                    result = model.transcribe(audio=filename)
                    answer=result["text"]
                    if answer != "":
                        print(answer)
                    # print(result["text"])
                    sfile.close()
                    if os.path.exists(filename):
                        os.remove(filename)
                    audio_buffer = np.zeros((int(SampleRate * BlockSize / 1000), 2)) # Reset the buffer
                    # audio_buffer2 = np.zeros((int(SampleRate * BlockSize / 1000), 1))
                    last_save_time = current_time
                    # i+=1
                ab = q.get()
                # ab,ab2 = q.get()
                audio_buffer = numpy.concatenate((audio_buffer, ab))
                # audio_buffer2 = numpy.concatenate((audio_buffer2, ab2))
                # file.write(whatstheaudio)
    except KeyboardInterrupt:
        print('\nTranscribing has been stopped ')

transcriber()