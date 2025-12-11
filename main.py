"""
실시간 RVC 오디오 변환기
"""

import pyaudio
import numpy as np
import time
from rvc_converter import VoiceConverterStream

# 설정
MODEL_PATH = "C:/Users/tmdfo/hogarakav2.pth"
INPUT_DEVICE_INDEX = None
OUTPUT_DEVICE_INDEX = None
INPUT_DEVICE_KEYWORD = "CABLE Output"
OUTPUT_DEVICE_KEYWORD = "Arctis Nova"

CHUNK = 24576
RATE = 40000
FORMAT = pyaudio.paInt16
PITCH_CHANGE = 14
NOISE_GATE_THRESHOLD = 50


def find_device(p, keyword, is_input=True):
    keyword_lower = keyword.lower()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()
        if keyword_lower in name:
            if is_input and info['maxInputChannels'] > 0:
                return i, info
            elif not is_input and info['maxOutputChannels'] > 0:
                return i, info
    return None, None


class StreamProcessor:
    # 스트림 오디오 처리
    
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.crossfade_len = 2048
        self.prev_tail = None
        self.prev_end_value = 0.0
    
    def apply_crossfade(self, audio):
        # 클릭 제거 및 크로스페이드 적용
        audio = audio.copy()
        
        # 클릭 제거
        smooth_len = 64
        if len(audio) > smooth_len and self.prev_tail is not None:
            start_diff = abs(audio[0] - self.prev_end_value)
            if start_diff > 0.08:
                t = np.linspace(0, 1, smooth_len)
                blend = (1 - np.cos(t * np.pi)) / 2
                audio[:smooth_len] = (
                    self.prev_end_value * (1 - blend) +
                    audio[:smooth_len] * blend
                ).astype(np.float32)
        
        # 크로스페이드
        if self.prev_tail is not None and len(audio) >= self.crossfade_len:
            t = np.linspace(0, np.pi, self.crossfade_len)
            fade_in = ((1 - np.cos(t)) / 2).astype(np.float32)
            fade_out = ((np.cos(t) + 1) / 2).astype(np.float32)
            
            audio[:self.crossfade_len] = (
                audio[:self.crossfade_len] * fade_in +
                self.prev_tail * fade_out
            )
        
        self.prev_end_value = audio[-1] if len(audio) > 0 else 0.0
        if len(audio) >= self.crossfade_len:
            self.prev_tail = audio[-self.crossfade_len:].copy()
        
        return audio
    
    def to_int16_stereo(self, audio_float):
        audio_int16 = (audio_float * 32768.0).clip(-32768, 32767).astype(np.int16)
        if self.out_channels == 2:
            return np.column_stack((audio_int16, audio_int16))
        return audio_int16


def main():
    print("=" * 50)
    print("[RVC 실시간 변환기]")
    print(f"RATE={RATE}, CHUNK={CHUNK}")
    print("=" * 50)
    
    p = pyaudio.PyAudio()
    
    in_idx, in_info = find_device(p, INPUT_DEVICE_KEYWORD, is_input=True)
    out_idx, out_info = find_device(p, OUTPUT_DEVICE_KEYWORD, is_input=False)
    
    if in_idx is None or out_idx is None:
        print("[에러] 장치를 찾을 수 없습니다.")
        p.terminate()
        return
    
    in_channels = min(int(in_info['maxInputChannels']), 2)
    out_channels = min(int(out_info['maxOutputChannels']), 2)
    
    print(f"\n입력: [{in_idx}] {in_info['name']} ({in_channels}ch)")
    print(f"출력: [{out_idx}] {out_info['name']} ({out_channels}ch)")
    
    input_stream = p.open(
        format=FORMAT, channels=in_channels, rate=RATE,
        input=True, input_device_index=in_idx, frames_per_buffer=CHUNK
    )
    
    output_stream = p.open(
        format=FORMAT, channels=out_channels, rate=RATE,
        output=True, output_device_index=out_idx, frames_per_buffer=CHUNK
    )
    
    print("\n[RVC 로딩 중...]")
    converter = VoiceConverterStream(MODEL_PATH)
    processor = StreamProcessor(out_channels)
    
    print("[GPU 웜업 중...]")
    warmup = np.zeros(CHUNK, dtype=np.float32)
    for _ in range(3):
        try:
            converter.convert(warmup, RATE, PITCH_CHANGE)
        except:
            pass
    print("[웜업 완료]")
    
    latency_ms = CHUNK / RATE * 1000
    print(f"\n[실행 중] 예상 지연: ~{latency_ms:.0f}ms")
    print("Ctrl+C로 종료\n")
    
    stats = {'processed': 0, 'passthrough': 0, 'errors': 0}
    last_print = time.time()
    
    try:
        while True:
            try:
                data = input_stream.read(CHUNK, exception_on_overflow=False)
            except:
                continue
            
            audio_np = np.frombuffer(data, dtype=np.int16)
            
            if in_channels == 2:
                audio_stereo = audio_np.reshape(-1, 2)
                audio_mono = audio_stereo[:, 0]
            else:
                audio_mono = audio_np
            
            if np.max(np.abs(audio_mono)) < NOISE_GATE_THRESHOLD:
                output_stream.write(data)
                stats['passthrough'] += 1
            else:
                try:
                    audio_float = audio_mono.astype(np.float32) / 32768.0
                    processed, output_sr = converter.convert(audio_float, RATE, PITCH_CHANGE)
                    
                    target_len = len(audio_mono)
                    if len(processed) > target_len:
                        processed = processed[:target_len]
                    elif len(processed) < target_len:
                        padding = np.zeros(target_len - len(processed), dtype=np.float32)
                        processed = np.concatenate([processed, padding])
                    
                    processed = processor.apply_crossfade(processed)
                    output_audio = processor.to_int16_stereo(processed)
                    output_stream.write(output_audio.tobytes())
                    stats['processed'] += 1
                    
                except Exception as e:
                    output_stream.write(data)
                    stats['errors'] += 1
            
            now = time.time()
            if now - last_print >= 5:
                total = stats['processed'] + stats['passthrough'] + stats['errors']
                if total > 0:
                    rvc_pct = stats['processed'] / total * 100
                    print(f"[통계] RVC: {stats['processed']} ({rvc_pct:.1f}%), "
                          f"통과: {stats['passthrough']}, 에러: {stats['errors']}")
                last_print = now
                
    except KeyboardInterrupt:
        print("\n[종료 중...]")
    finally:
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()
        print("[완료]")


if __name__ == "__main__":
    main()