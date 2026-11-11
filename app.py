import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 한글 폰트 설정
# -----------------------------
def 한글폰트설정():
    후보 = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"]
    사용가능 = {f.name for f in fm.fontManager.ttflist}

    for 폰트 in 후보:
        if 폰트 in 사용가능:
            plt.rcParams["font.family"] = 폰트
            break

    plt.rcParams["axes.unicode_minus"] = False

한글폰트설정()

st.set_page_config(
    page_title="고등학생 AI 실습 플랫폼",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 고등학생 AI 학습 실험실")
st.write("7개 주제의 데이터로 손실함수(MSE)와 경사하강법을 활용해 딥러닝 모델을 학습시켜 보세요!")

# -----------------------------
# 데이터 생성 함수
# -----------------------------
rng = np.random.default_rng(42)

def 유튜브_데이터():
    n = 100
    구독자수 = rng.integers(1, 100, n)
    영상길이 = rng.integers(5, 30, n)
    썸네일_품질 = rng.integers(1, 10, n)
    조회수 = (
        구독자수 * 500
        + 영상길이 * 200
        + 썸네일_품질 * 300
        + rng.integers(-500, 500, n)
    )
    X = np.column_stack([구독자수, 영상길이, 썸네일_품질])
    y = 조회수
    return X, y, ["구독자수", "영상길이", "썸네일 품질"], "조회수"

def 게임_데이터():
    n = 100
    공격력 = rng.integers(30, 100, n)
    방어력 = rng.integers(30, 100, n)
    속도 = rng.integers(30, 100, n)
    체력 = rng.integers(30, 100, n)
    전투력 = (
        공격력 * 3
        + 방어력 * 2
        + 속도 * 2
        + 체력 * 2
        + rng.integers(-50, 50, n)
    )
    X = np.column_stack([공격력, 방어력, 속도, 체력])
    y = 전투력
    return X, y, ["공격력", "방어력", "속도", "체력"], "전투력"

def 스포츠_데이터():
    n = 100
    체력 = rng.integers(50, 100, n)
    스피드 = rng.integers(50, 100, n)
    기술 = rng.integers(50, 100, n)
    경험 = rng.integers(1, 10, n)
    득점 = (
        체력 * 0.3
        + 스피드 * 0.4
        + 기술 * 0.5
        + 경험 * 2
        + rng.integers(-5, 5, n)
    )
    X = np.column_stack([체력, 스피드, 기술, 경험])
    y = 득점
    return X, y, ["체력", "스피드", "기술", "경험"], "득점"

def 음악_데이터():
    n = 100
    템포 = rng.integers(60, 180, n)
    에너지 = rng.integers(1, 10, n)
    댄스 = rng.integers(1, 10, n)
    스트리밍 = (
        템포 * 100
        + 에너지 * 500
        + 댄스 * 400
        + rng.integers(-1000, 1000, n)
    )
    X = np.column_stack([템포, 에너지, 댄스])
    y = 스트리밍
    return X, y, ["템포", "에너지", "댄스 가능성"], "스트리밍 수"

def 영화_데이터():
    n = 100
    상영시간 = rng.integers(80, 180, n)
    액션장면 = rng.integers(0, 40, n)
    유명배우 = rng.integers(0, 5, n)
    평점 = (
        상영시간 * 0.02
        + 액션장면 * 0.05
        + 유명배우 * 0.5
        + rng.normal(0, 0.3, n)
    )
    X = np.column_stack([상영시간, 액션장면, 유명배우])
    y = 평점
    return X, y, ["상영시간", "액션장면 수", "유명배우 수"], "평점"

def 동물_데이터():
    n = 100
    다리수 = rng.integers(0, 8, n)
    몸길이 = rng.integers(10, 200, n)
    수명 = rng.integers(1, 30, n)
    몸무게 = (
        다리수 * 2
        + 몸길이 * 0.5
        + 수명 * 1.5
        + rng.integers(-10, 10, n)
    )
    X = np.column_stack([다리수, 몸길이, 수명])
    y = 몸무게
    return X, y, ["다리 수", "몸길이", "수명"], "몸무게"

def sns_데이터():
    n = 100
    게시글길이 = rng.integers(20, 300, n)
    해시태그 = rng.integers(0, 15, n)
    사진수 = rng.integers(0, 6, n)
    좋아요 = (
        게시글길이 * 2
        + 해시태그 * 30
        + 사진수 * 100
        + rng.integers(-100, 100, n)
    )
    X = np.column_stack([게시글길이, 해시태그, 사진수])
    y = 좋아요
    return X, y, ["게시글 길이", "해시태그 수", "사진 수"], "좋아요 수"

데이터_생성 = {
    "유튜브 영상 조회수 예측": 유튜브_데이터,
    "게임 캐릭터 능력치 분석": 게임_데이터,
    "스포츠 선수 기록 분석": 스포츠_데이터,
    "음악 취향 분류": 음악_데이터,
    "영화 평점 예측": 영화_데이터,
    "동물 분류": 동물_데이터,
    "SNS 게시글 반응 예측": sns_데이터
}

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "학습완료" not in st.session_state:
    st.session_state.학습완료 = False

if "예측결과" not in st.session_state:
    st.session_state.예측결과 = None

if "모델" not in st.session_state:
    st.session_state.모델 = None

if "스케일러" not in st.session_state:
    st.session_state.스케일러 = None

if "학습기록" not in st.session_state:
    st.session_state.학습기록 = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "현재주제" not in st.session_state:
    st.session_state.현재주제 = None

if "애니메이션_완료" not in st.session_state:
    st.session_state.애니메이션_완료 = False

if "애니메이션속도" not in st.session_state:
    st.session_state.애니메이션속도 = 0.2

# -----------------------------
# 사이드바 설정
# -----------------------------
st.sidebar.header("⚙️ 실험 설정")

주제 = st.sidebar.selectbox(
    "실습 주제 선택",
    list(데이터_생성.keys())
)

# 주제가 바뀌면 학습 및 애니메이션 상태 초기화
if 주제 != st.session_state.현재주제:
    st.session_state.학습완료 = False
    st.session_state.모델 = None
    st.session_state.스케일러 = None
    st.session_state.학습기록 = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.현재주제 = 주제
    st.session_state.예측결과 = None
    st.session_state.애니메이션_완료 = False

학습률 = st.sidebar.number_input(
    "학습률(learning rate)",
    min_value=0.0001,
    max_value=1.0,
    value=0.001,
    step=0.0001,
    format="%.4f"
)

에포크 = st.sidebar.number_input(
    "학습 반복 횟수(epoch)",
    min_value=10,
    max_value=1000,
    value=200,
    step=10
)

배치크기 = st.sidebar.number_input(
    "배치 크기(batch size)",
    min_value=1,
    max_value=128,
    value=16
)

학습시작 = st.sidebar.button("🚀 모델 학습 시작")

# -----------------------------
# 데이터 불러오기
# -----------------------------
X, y, 입력변수, 정답변수 = 데이터_생성[주제]()

df = pd.DataFrame(X, columns=입력변수)
df[정답변수] = y

st.header(f"1️⃣ {주제}")
st.write("총 **100개**의 데이터가 준비되어 있습니다.")

st.subheader("📊 데이터 미리보기 (5개)")
st.dataframe(df.head(5))

# -----------------------------
# 손실함수 설명
# -----------------------------
st.header("2️⃣ 손실함수: 평균제곱오차(MSE)")

st.info("""
**평균제곱오차(MSE)**

예측값과 실제값의 차이를 제곱한 뒤 평균 낸 값입니다.

MSE가 작을수록 모델이 데이터를 잘 예측한다는 뜻입니다.

**경사하강법**

손실을 줄이는 방향으로 모델의 가중치를 조금씩 바꾸는 방법입니다.
""")

# -----------------------------
# 학습 실행
# -----------------------------
if 학습시작:

    with st.spinner("⏳ 모델 학습 중입니다... 잠시만 기다려 주세요."):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=학습률),
            loss="mse",
            metrics=["mae"]
        )

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=int(에포크),
            batch_size=int(배치크기),
            verbose=0
        )

        # 세션에 저장
        st.session_state.모델 = model
        st.session_state.스케일러 = scaler
        st.session_state.학습기록 = history
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.학습완료 = True
        st.session_state.애니메이션_완료 = False  # 새로 학습하면 애니메이션도 초기화

    st.success("✅ 학습이 완료되었습니다!")

# -----------------------------
# 학습 결과 표시
# -----------------------------
if st.session_state.학습완료:

    model = st.session_state.모델
    scaler = st.session_state.스케일러
    history = st.session_state.학습기록
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    st.header("3️⃣ 모델 학습 결과")

    # -----------------------------
    # 손실 그래프
    # -----------------------------
    st.subheader("📉 손실 감소 그래프")

    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="학습 손실")
    ax.plot(history.history["val_loss"], label="검증 손실")
    ax.set_xlabel("학습 반복 횟수")
    ax.set_ylabel("손실(MSE)")
    ax.set_title("학습이 진행될수록 손실 감소")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # 경사하강법 애니메이션
    # -----------------------------
    st.subheader("🧭 경사하강법 애니메이션")

    st.info("""
    아래 그래프는 간단한 선형 모델 y = wx + b를 예시로 보여줍니다.

    빨간 점은 모델의 현재 위치이고,  
    빨간 선은 손실을 줄이기 위해 이동한 경로입니다.

    학습이 진행될수록 손실이 작은 중앙으로 이동합니다.
    """)

    # 애니메이션이 아직 실행되지 않았을 때만 재생 루프 돌기
    if not st.session_state.애니메이션_완료:
        x_simple = X[:, 0]
        y_simple = y

        x_mean = x_simple.mean()
        y_mean = y_simple.mean()

        x_norm = (x_simple - x_mean) / x_simple.std()
        y_norm = (y_simple - y_mean) / y_simple.std()

        w = 0.0
        b = 0.0
        학습률_시각화 = 0.1
        반복횟수_시각화 = 30

        w_values = np.linspace(-2, 2, 100)
        b_values = np.linspace(-2, 2, 100)
        W, B = np.meshgrid(w_values, b_values)

        Z = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                예측 = W[i, j] * x_norm + B[i, j]
                Z[i, j] = np.mean((예측 - y_norm) ** 2)

        그래프자리 = st.empty()
        상태메시지 = st.empty()

        경로_w = [w]
        경로_b = [b]

        for step in range(반복횟수_시각화):
            예측 = w * x_norm + b
            오차 = 예측 - y_norm

            w_기울기 = np.mean(오차 * x_norm)
            b_기울기 = np.mean(오차)

            w -= 학습률_시각화 * w_기울기
            b -= 학습률_시각화 * b_기울기

            경로_w.append(w)
            경로_b.append(b)

            fig, ax = plt.subplots()

            contour = ax.contourf(W, B, Z, levels=30, cmap="viridis")
            fig.colorbar(contour, ax=ax)

            ax.plot(경로_w, 경로_b, color="red", marker="o")
            ax.set_xlabel("가중치 w")
            ax.set_ylabel("편향 b")
            ax.set_title(f"경사하강법 이동 경로 (반복 {step + 1}회)")

            그래프자리.pyplot(fig)
            상태메시지.info(f"⏳ 경사하강법 진행 중... {step + 1} / {반복횟수_시각화}")

            plt.close(fig)

            # 세션 상태에 저장된 속도(대기 시간) 반영
            time.sleep(st.session_state.애니메이션속도)

        상태메시지.success("✅ 경사하강법이 완료되었습니다!")
        st.session_state.애니메이션_완료 = True  # 애니메이션 완료 상태 저장

    else:
        st.success("✅ 경사하강법이 완료되었습니다!")

    # 👇 [🔄 다시 보기] 버튼과 [속도 조절 슬라이더]를 애니메이션 아래에 항상 고정 배치!
    if st.button("🔄 애니메이션 다시 보기"):
        st.session_state.애니메이션_완료 = False
        st.rerun()

    st.session_state.애니메이션속도 = st.slider(
        "🧭 애니메이션 대기 시간 (초)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.애니메이션속도,
        step=0.01,
        help="값이 작을수록 애니메이션이 빠르게 재생됩니다."
    )

    # -----------------------------
    # 테스트 평가
    # -----------------------------
    st.subheader("🎯 학습이 잘 되었는지 확인")

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    col1, col2 = st.columns(2)
    col1.metric("테스트 손실(MSE)", round(loss, 2))
    col2.metric("평균 오차(MAE)", round(mae, 2))

    예측값 = model.predict(X_test, verbose=0).flatten()

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, 예측값, color="blue")
    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
    ax2.set_xlabel("실제값")
    ax2.set_ylabel("예측값")
    ax2.set_title("실제값 vs 예측값")
    st.pyplot(fig2)

# -----------------------------
# 새 데이터 예측
# -----------------------------
st.header("4️⃣ 새 데이터로 예측해 보기")

if st.session_state.학습완료:

    model = st.session_state.모델
    scaler = st.session_state.스케일러

    새_입력 = []

    for 변수 in 입력변수:
        값 = st.number_input(
            f"새 {변수}",
            0,
            10000,
            50,
            key=f"입력_{변수}"
        )
        새_입력.append(값)

    예측버튼 = st.button("🔮 예측하기")

    if 예측버튼:
        새_데이터 = np.array([새_입력])
        새_데이터 = scaler.transform(새_데이터)

        예측 = model.predict(새_데이터, verbose=0)[0][0]
        st.session_state.예측결과 = 예측

    if st.session_state.예측결과 is not None:
        예측 = st.session_state.예측결과
        st.success(f"예측된 {정답변수}: **{예측:.2f}**")

else:
    st.warning("먼저 왼쪽에서 **모델 학습 시작** 버튼을 눌러주세요.")