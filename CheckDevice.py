import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INPUT_DEVICE_INDEX = 4
OUTPUT_DEVICE_INDEX = 10

def VoiceChanger():
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        input_device_index=INPUT_DEVICE_INDEX,
                        output_device_index=OUTPUT_DEVICE_INDEX,
                        frames_per_buffer=CHUNK)
        
        print(f"=== [실행 중] ID 4 -> ID 10 ===")

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            audio_np = np.frombuffer(data, dtype=np.int16)
            processed_audio = audio_np 

            stream.write(processed_audio.tobytes())

    except KeyboardInterrupt:
        print("=== 종료 ===")
    except Exception as e:
        print(f"=== 에러: {e} ===")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    VoiceChanger()