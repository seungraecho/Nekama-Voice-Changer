import os
import inspect
from rvc_python.infer import RVCInference

class VoiceConverter:
    def __init__(self, model_path, device="cuda:0"):
        print(f"[RVC] RTX 5060 GPU({device})로 모델 로딩을 시작합니다...")
        
        try:
            self.rvc = RVCInference(device=device)
        except Exception:
            self.rvc = RVCInference()
            print("[Info] Device 인자 없이 초기화했습니다.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
        self.rvc.load_model(model_path)
        print(f"[RVC] 모델 로드 완료! 준비 끝.")

    def convert_audio(self, input_path, output_path, pitch_change=0, target_sr=0):
        """
        오디오 변환 (타겟 샘플 레이트 지정 가능)
        """
        # 1. 라이브러리 함수가 어떤 재료를 원하는지 검사
        try:
            sig = inspect.signature(self.rvc.infer_file)
            params = sig.parameters
        except Exception:
            params = {} 

        call_args = {}
        
        # 2. 피치(Pitch) 설정
        if "f0_up_key" in params: call_args["f0_up_key"] = pitch_change
        elif "pitch" in params: call_args["pitch"] = pitch_change
        elif "f0_change" in params: call_args["f0_change"] = pitch_change
        elif "up_key" in params: call_args["up_key"] = pitch_change
        else:
            try:
                self.rvc.f0_up_key = pitch_change
                self.rvc.args.f0_up_key = pitch_change
            except:
                pass

        # 3. 알고리즘 설정
        if "f0_method" in params: call_args["f0_method"] = "rmvpe"
        elif "algorithm" in params: call_args["algorithm"] = "rmvpe"
        elif "method" in params: call_args["method"] = "rmvpe"

        # 4. 기타 옵션 (리샘플링 적용)
        if "index_rate" in params: call_args["index_rate"] = 0.75
        if "filter_radius" in params: call_args["filter_radius"] = 3
        
        # [핵심 수정] 타겟 샘플 레이트가 있으면 적용
        if "resample_sr" in params: call_args["resample_sr"] = target_sr 
        
        if "rms_mix_rate" in params: call_args["rms_mix_rate"] = 0.25
        if "protect" in params: call_args["protect"] = 0.33

        # 5. 변환 실행
        self.rvc.infer_file(input_path, output_path, **call_args)