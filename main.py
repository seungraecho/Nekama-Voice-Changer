import pyaudio
import numpy as np
import soundfile as sf
import os
import time
from rvc_infer import VoiceConverter 

# ==========================================
# [설정 영역]
# ==========================================
MODEL_PATH = "C:/Users/tmdfo/hogarakav2.pth" 

INPUT_DEVICE_INDEX = 4   # CABLE Output
OUTPUT_DEVICE_INDEX = 6  # 헤드폰 (Arctis Nova)

# ★ 끊김 해결의 핵심: CHUNK 크기 증가
# 6144 -> 10240 (약 0.23초 지연되지만 끊김은 사라짐)
CHUNK = 10240            
PITCH_CHANGE = 0         
RATE = 44100             
FORMAT = pyaudio.paInt16
CHANNELS = 2             

def run_voice_changer():
    try:
        converter = VoiceConverter(MODEL_PATH)
    except Exception as e:
        print(f"\n[치명적 에러] 모델 로딩 실패: {e}")
        return

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
        
        print(f"\n=== [RVC 끊김 방지 모드] ===")
        print(f"CHUNK 크기: {CHUNK} (안정성 우선)")
        print(f"설정: Pitch={PITCH_CHANGE}, Rate={RATE}Hz")
        print("이제 소리가 훨씬 부드럽게 들릴 것입니다.\n")

        temp_input = "temp_in.wav"
        temp_output = "temp_out.wav"

        while True:
            # A. 소리 읽기
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError:
                # 버퍼 오버플로우 발생 시 무시하고 진행
                continue
                
            audio_np = np.frombuffer(data, dtype=np.int16)
            
            # 스테레오 변환 (안전 장치)
            try:
                audio_stereo = audio_np.reshape(-1, 2)
            except Exception:
                stream.write(data)
                continue

            # B. 무음 감지 (노이즈 게이트)
            if np.max(np.abs(audio_stereo)) < 300:
                stream.write(data) 
                continue

            # C. AI 변환
            try:
                # 1. 파일 쓰기
                audio_mono = audio_stereo[:, 0] 
                sf.write(temp_input, audio_mono, RATE, subtype='PCM_16')

                # 2. AI 변환 (시간 소요)
                converter.convert_audio(temp_input, temp_output, PITCH_CHANGE, target_sr=RATE)

                # 3. 파일 읽기
                processed_data, sr = sf.read(temp_output, dtype='int16')
                
                # 4. 스테레오 복제
                if len(processed_data.shape) == 1:
                    processed_stereo = np.column_stack((processed_data, processed_data))
                else:
                    processed_stereo = processed_data

                # 5. 길이 맞추기 (중요: 입력과 출력 길이가 다르면 밀림 현상 발생)
                # AI가 처리하면서 길이가 미세하게 변할 수 있으므로, CHUNK 크기에 맞게 자르거나 채움
                current_len = len(processed_stereo)
                target_len = CHUNK // 2 # CHUNK는 바이트 단위가 아니라 프레임 단위면 CHUNK, 여기선 read가 프레임 단위라 CHUNK가 맞음
                
                # PyAudio read(CHUNK)는 CHUNK 개수의 '프레임'을 읽음.
                # 그러나 numpy shape는 (CHUNK, 2)가 됨.
                if current_len > CHUNK:
                    processed_stereo = processed_stereo[:CHUNK]
                elif current_len < CHUNK:
                    # 부족하면 0으로 채움 (Pad)
                    padding = np.zeros((CHUNK - current_len, 2), dtype='int16')
                    processed_stereo = np.vstack((processed_stereo, padding))

                stream.write(processed_stereo.tobytes())

            except Exception as e:
                # print(f"[스킵] {e}")
                stream.write(data)

    except KeyboardInterrupt:
        print("\n=== 종료 ===")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_output): os.remove(temp_output)

if __name__ == "__main__":
    run_voice_changer()