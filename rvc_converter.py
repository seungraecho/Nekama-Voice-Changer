"""
RVC 스트리밍 변환기
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from rvc_python.infer import RVCInference


class VoiceConverterStream:
    def __init__(self, model_path, device="cuda:0"):
        print(f"[RVC] GPU({device})로 모델 로딩 중...")
        
        try:
            self.rvc = RVCInference(device=device)
        except Exception:
            self.rvc = RVCInference()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 없음: {model_path}")
            
        self.rvc.load_model(model_path)
        
        # RVC 파라미터
        self.rvc.f0method = "rmvpe"
        self.rvc.index_rate = 0.4
        self.rvc.filter_radius = 3
        self.rvc.resample_sr = 0
        self.rvc.rms_mix_rate = 0.3
        self.rvc.protect = 0.5
        
        print(f"[RVC] 모델 로드 완료")
        print(f"[RVC] 출력 샘플레이트: 40000Hz")
        
        self.temp_dir = tempfile.gettempdir()
        self.temp_input = os.path.join(self.temp_dir, "rvc_temp_in.wav")
        self.temp_output = os.path.join(self.temp_dir, "rvc_temp_out.wav")
    
    def convert(self, audio_data, sample_rate, pitch_change=0):
        # RVC 변환
        
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        self.rvc.f0up_key = pitch_change
        
        sf.write(self.temp_input, audio_data, sample_rate, subtype='FLOAT')
        self.rvc.infer_file(self.temp_input, self.temp_output)
        processed, output_sr = sf.read(self.temp_output, dtype='float32')
        
        return processed, output_sr
    
    def cleanup(self):
        for f in [self.temp_input, self.temp_output]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
                    
    def __del__(self):
        self.cleanup()