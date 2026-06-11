import os

# CPU에서 실행
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import io
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

# TensorFlow가 GPU를 찾더라도 CPU만 사용하도록 강제
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# dncskin_classification 폴더에서 모듈을 가져오기 위해 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dncskin_classification")))

try:
    import print_log
    print_log.set_log_path("workspace/logs", "fastapi_server")
    print_log.print_logi("FastAPI 서버 시작", "Server")
except ImportError:
    print_log = None
    print("로그 모듈을 사용할 수 없습니다. 'print_log' 모듈이 없거나 설정에 문제가 있습니다.")

app = FastAPI(
    title="Alfy Pet Skin Diagnosis API",
    description="강아지/고양이 피부 질병 진단 API"
)

DOG_MODEL_DIR = "../model/dog"
CAT_MODEL_DIR = "../model/cat"

models = {
    "dog": None,
    "cat": None
}

model_status = {
    "dog": {
        "loaded": False,
        "message": "강아지 모델이 아직 로드되지 않았습니다."
    },
    "cat": {
        "loaded": False,
        "message": "고양이 모델이 아직 로드되지 않았습니다."
    }
}


def load_ai_model(model_dir: str, animal: str, model_type: str = "multi"):
    """
    체크포인트 디렉토리에서 모델을 로드합니다.
    가중치(weights)만 저장된 체크포인트이므로 구조를 먼저 생성합니다.
    """
    if not os.path.exists(model_dir):
        msg = f"{animal} 모델 경로가 없습니다."
        print(msg)
        model_status[animal] = {
            "loaded": False,
            "message": msg
        }
        return None

    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

        if model_type == "multi":
            num_class = 7
            activation = "softmax"
        else:
            num_class = 2
            activation = "sigmoid"

        network = InceptionResNetV2(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            pooling="avg"
        )

        model = Sequential([
            network,
            Dense(2048, activation="relu"),
            Dense(num_class, activation=activation)
        ])

        if latest_ckpt:
            print(f"{animal} 체크포인트 모델 로드 중...")
            model.load_weights(latest_ckpt).expect_partial()

            model_status[animal] = {
                "loaded": True,
                "message": f"{animal} 모델 로드 완료"
            }
            return model

        model = tf.keras.models.load_model(model_dir)
        model_status[animal] = {
            "loaded": True,
            "message": f"{animal} SavedModel 로드 완료"
        }
        return model

    except Exception as e:
        msg = f"{animal} 모델 로드 중 오류 발생: {e}"
        print(msg)
        model_status[animal] = {
            "loaded": False,
            "message": msg
        }
        return None


try:
    print("고양이 모델 로드를 시도합니다...")
    models["cat"] = load_ai_model(CAT_MODEL_DIR, animal="cat", model_type="binary")

    if models["cat"]:
        print("고양이 모델 로드 완료!")
    else:
        print(model_status["cat"]["message"])

    print("강아지 모델 로드를 시도합니다...")
    models["dog"] = load_ai_model(DOG_MODEL_DIR, animal="dog", model_type="multi")

    if models["dog"]:
        print("강아지 모델 로드 완료!")
    else:
        print(model_status["dog"]["message"])

except Exception as e:
    print(f"전체 모델 초기화 중 오류 발생: {e}")


DOG_DISEASE_CLASSES = {
    0: "A1_구진/플라크",
    1: "A2_비듬/각질/상피성잔고리",
    2: "A3_태선화/과다색소침착",
    3: "A4_농포/여드름",
    4: "A5_미란/궤양",
    5: "A6_결절/종괴",
    6: "정상 (무증상)"
}

CAT_DISEASE_CLASSES = {
    0: "정상 (무증상)",
    1: "A2_비듬/각질/상피성잔고리"
}


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Alfy Pet Skin Diagnosis API is running.",
        "model_status": model_status
    }


@app.post("/diagnose")
async def diagnose_skin(animal: str, file: UploadFile = File(...)):
    """
    - animal: 'dog' 또는 'cat' 입력
    - file: 진단할 이미지 파일
    """
    animal = animal.lower()

    if animal not in ["dog", "cat"]:
        raise HTTPException(
            status_code=400,
            detail="animal 파라미터는 'dog' 또는 'cat'이어야 합니다."
        )

    try:
        if print_log:
            print_log.print_logi(
                f"진단 요청 수신: {animal} - {file.filename}",
                "DiagnoseAPI",
                on_screen_display=True,
                force_flush=True
            )

        selected_model = models.get(animal)

        if selected_model is None:
            if print_log:
                print_log.print_loge(
                    model_status[animal]["message"],
                    "DiagnoseAPI",
                    on_screen_display=True,
                    force_flush=True
                )

            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"서버에 {animal}용 AI 모델이 로드되어 있지 않습니다.",
                    "model_status": model_status[animal]
                }
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = selected_model.predict(image_array, verbose=0)[0]

        if print_log:
            print_log.print_logv(
                f"추론 완료: {predictions}",
                "DiagnoseAPI",
                force_flush=True
            )

        max_index = int(np.argmax(predictions))
        confidence = float(predictions[max_index]) * 100

        if animal == "dog":
            classes_dict = DOG_DISEASE_CLASSES
        else:
            classes_dict = CAT_DISEASE_CLASSES

        disease_name = classes_dict.get(max_index, "알 수 없는 피부상태")

        details = {
            classes_dict.get(i, f"Class {i}"): f"{float(prob) * 100:.2f}%"
            for i, prob in enumerate(predictions)
        }

        return {
            "status": "success",
            "animal": animal,
            "filename": file.filename,
            "model_loaded": True,
            "model_status": model_status[animal],
            "prediction": {
                "disease": disease_name,
                "probability": f"{confidence:.2f}%"
            },
            "details": details
        }

    except HTTPException:
        raise

    except Exception as e:
        if print_log:
            print_log.print_loge(
                f"AI 진단 실패: {str(e)}",
                "DiagnoseAPI",
                on_screen_display=True,
                force_flush=True
            )

        raise HTTPException(
            status_code=500,
            detail=f"AI Diagnosis failed: {str(e)}"
        )

