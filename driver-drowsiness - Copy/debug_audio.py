import pyttsx3
import pyaudio
import wave
import sys

def test_tts():
    print("Testing Text-to-Speech...")
    try:
        engine = pyttsx3.init()
        print("TTS Engine initialized.")
        engine.say("Testing voice output system.")
        engine.runAndWait()
        print("TTS test complete. Did you hear the voice?")
    except Exception as e:
        print(f"TTS Error: {e}")

def get_pyaudio_input_device_index():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    default_device_index = -1
    
    print("\n----------------------")
    print("Available Audio Input Devices:")
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            print(f"Input Device id {i} - {name}")
            if default_device_index == -1:
                default_device_index = i
                
    try:
        default_info = p.get_default_input_device_info()
        print(f"\nSystem Default Input Device: ID {default_info['index']} - {default_info['name']}")
        default_device_index = default_info['index']
    except Exception as e:
        print(f"\nCould not determine system default input device: {e}")
        
    p.terminate()
    return default_device_index

def test_mic(device_index):
    if device_index is None or device_index < 0:
        print("No valid input device found.")
        return

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "debug_recording.wav"

    p = pyaudio.PyAudio()

    print(f"\nRecording {RECORD_SECONDS} seconds of audio from device ID {device_index}...")
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Failed to open stream on device {device_index}: {e}")
        p.terminate()
        return

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Saved recording to {WAVE_OUTPUT_FILENAME}")
    print("Please play this file manually to verify microphone clarity.")

if __name__ == "__main__":
    print("=== Audio Debug Tool ===")
    test_tts()
    dev_index = get_pyaudio_input_device_index()
    test_mic(dev_index)
