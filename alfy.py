import os
# CPU에서 실행
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import httpx
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
import sys
import asyncio

# dncskin_classification 폴더에서 모듈을 가져오기 위해 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dncskin_classification")))

try:
    import print_log
    # 로그 출력을 위한 초기 설정
    print_log.set_log_path("workspace/logs", "fastapi_server")
    print_log.print_logi("FastAPI 서버 완료", "Server")
except ImportError:
    print_log = None
    print("로그를 확인할 수 없습니다. 'print_log' 모듈이 존재하지 않거나 로깅 설정에 문제가 있습니다.")

app = FastAPI(title="Alfy Pet Skin Diagnosis API", description="강아지/고양이 피부 질병 진단 API")

DOG_MODEL_DIR = "/workspace/dncskin_classification/workspace/data/dog/result_cla/multi_A1_A2_A3_A4_A5_A6/model"
CAT_MODEL_DIR = "/workspace/dncskin_classification/workspace/data/cat/result_cla/binary_A2/model"

models = {"dog": None, "cat": None}

def load_ai_model(model_dir: str, model_type: str = 'multi'):
    """
    체크포인트 디렉토리에서 모델을 로드합니다.
    가중치(weights)만 저장된 체크포인트이므로 구조를 먼저 생성합니다.
    """
    if not os.path.exists(model_dir):
        return None

    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

        # 모델 아키텍처 생성
        if model_type == 'multi':
            num_class = 7
            activation = 'softmax'
        else:
            num_class = 2
            activation = 'sigmoid'
            
        network = InceptionResNetV2(include_top=False, weights=None, input_shape=(224, 224, 3), pooling='avg')
        model = Sequential([
            network,
            Dense(2048, activation='relu'),
            Dense(num_class, activation=activation)
        ])

        if latest_ckpt:
            print(f"최신 체크포인트 발견: {latest_ckpt}")
            # 가중치 로드
            model.load_weights(latest_ckpt).expect_partial()
            return model
        else:
            # SavedModel 폴더 자체를 모델로 로드 시도
            return tf.keras.models.load_model(model_dir)
    except Exception as e:
        print(f"[{model_dir}] 모델 로드 중 오류 발생: {e}")
        return None

try:
    print("고양이 Model 로드를 시도합니다...")
    models["cat"] = load_ai_model(CAT_MODEL_DIR, model_type='binary')
    if models["cat"]:
        print("고양이 Model 로드 완료!")
    else:
        print(f"exception: '{CAT_MODEL_DIR}' 경로에서 고양이 모델을 찾을 수 없거나 로드에 실패했습니다.")

    print("강아지 Model 로드를 시도합니다...")
    models["dog"] = load_ai_model(DOG_MODEL_DIR, model_type='multi')
    if models["dog"]:
        print("강아지 Model 로드 완료!")
    else:
        print(f"exception: '{DOG_MODEL_DIR}' 경로에서 강아지 모델을 찾을 수 없거나 로드에 실패했습니다.")

except Exception as e:
    print(f"전체 모델 초기화 중 오류 발생: {e}")

# 강아지 질병 클래스
DOG_DISEASE_CLASSES = {
    0: "A1_구진/플라크",
    1: "A2_비듬/각질/상피성잔고리",
    2: "A3_태선화/과다색소침착",
    3: "A4_농포/여드름",
    4: "A5_미란/궤양",
    5: "A6_결절/종괴",
    6: "정상 (무증상)"
}

# 고양이 질병 클래스
CAT_DISEASE_CLASSES = {
    0: "정상 (무증상)",
    1: "A2_비듬/각질/상피성잔고리"
}

# 피부 진단 api
@app.post("/diagnose")
async def diagnose_skin(animal: str, file: UploadFile = File(...)):
    """
    - animal: 'dog' 또는 'cat' 입력
    - file: 진단할 이미지 파일
    """
    animal = animal.lower()
    if animal not in ["dog", "cat"]:
        raise HTTPException(status_code=400, detail="animal 파라미터는 'dog' 또는 'cat'이어야 합니다.")

    try:
        if print_log:
            print_log.print_logi(f"진단 요청 수신: {animal} - {file.filename}", "DiagnoseAPI", on_screen_display=True, force_flush=True)
            
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 이미지 전처리
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # 모델 추론
        selected_model = models.get(animal)
        if selected_model is None:
            if print_log:
                print_log.print_loge(f"{animal} 모델이 로드되어 있지 않습니다.", "DiagnoseAPI", on_screen_display=True, force_flush=True)
            raise HTTPException(status_code=500, detail=f"서버에 {animal}용 AI 모델이 로드되어 있지 않습니다.")
            
        predictions = selected_model.predict(image_array)[0]
        if print_log:
            print_log.print_logv(f"추론 완료: {predictions}", "DiagnoseAPI", force_flush=True)
        
        # 결과 해석
        max_index = int(np.argmax(predictions))
        confidence = float(predictions[max_index]) * 100
        
        if animal == "dog":
            classes_dict = DOG_DISEASE_CLASSES
        else:
            classes_dict = CAT_DISEASE_CLASSES
            
        disease_name = classes_dict.get(max_index, "알 수 없는 피부상태")
        details = {classes_dict.get(i, f"Class {i}"): f"{float(prob)*100:.2f}%" for i, prob in enumerate(predictions)}
        
        return {
            "status": "success",
            "animal": animal,
            "filename": file.filename,
            "prediction": {
                "disease": disease_name,
                "probability": f"{confidence:.2f}%"
            },
            "details": details
        }
        
    except Exception as e:
        if print_log:
            print_log.print_loge(f"AI 진단 실패: {str(e)}", "DiagnoseAPI", on_screen_display=True, force_flush=True)
        raise HTTPException(status_code=500, detail=f"AI Diagnosis failed: {str(e)}")

@app.get("/logs", response_class=PlainTextResponse)
async def get_server_logs(tail: int = 5):
    """
    서버 로그를 반환하는 API
    """
    log_file_path = "/workspace/afly/server.log"
    
    if not os.path.exists(log_file_path):
        return "기록된 서버 로그 파일이 없습니다.\n"
        
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        return "".join(lines[-tail:])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"로그 읽기 실패: {str(e)}")

@app.get("/log/stream")
async def stream_server_logs():
    """
    서버 로그를 실시간으로 스트리밍하는 API
    """
    log_file_path = "/workspace/afly/server.log"
    
    if not os.path.exists(log_file_path):
        # 스트리밍 시도 시점에는 파일이 아직 안 만들어졌을 수 있으므로 대기 후 스트리밍
        with open(log_file_path, 'w', encoding='utf-8') as f: pass

    async def log_generator():
        # 브라우저의 초기 렌더링 버퍼(약 1KB)를 채워서 즉시 화면에 출력되도록 강제하는 공백
        yield (" " * 1024) + "\n"
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-1:]:  # 처음 접속 시 최근 딱 1줄만 출력
                    yield line
                
                while True:
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.5)
                        continue
                    yield line
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        log_generator(), 
        media_type="text/plain", 
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )
