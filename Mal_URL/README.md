""# 악성 URL 탐지 모델 개발

이 프로젝트는 악성 URL을 탐지하기 위한 머신러닝 모델 개발 과정을 상세히 담고 있습니다. URL에서 다양한 특징을 추출하고 데이터를 전처리한 후, 여러 모델 학습 기법과 앙상블을 활용하여 최종 예측 결과를 생성합니다.

---

## 📂 **파일 및 디렉토리 구조**

```
📦 submission_autogl
 ┣ 📂data                   # 데이터 파일이 저장되는 디렉토리
 ┃ ┣ 📜train.csv            # 학습 데이터
 ┃ ┣ 📜test.csv             # 테스트 데이터
 ┃ ┗ 📜sample_submission.csv # 샘플 제출 양식
 ┣ 📂AutogluonModels        # AutoGluon 모델 학습 시 생성된 체크포인트와 메타데이터
 ┣ 📜ML.ipynb               # 전체 머신러닝 파이프라인을 담고 있는 Jupyter Notebook
 ┣ 📜submission.csv         # 최종 제출 파일
 ┣ 📜.gitignore             # Git 추적 제외 파일 목록
 ┗ 📜requirements.txt       # 프로젝트 실행에 필요한 라이브러리 목록
```

---

## 💡 **프로젝트 주요 기능**

### ✅ **1. 데이터 로딩 및 탐색**

* `train.csv`, `test.csv`를 로드하고, 데이터셋의 기본적인 탐색을 진행합니다.
* 데이터셋의 결측치, 중복치 여부를 확인하고 처리합니다.
* 악성 URL 탐지에 유의미한 특징들을 선별합니다.

### ✅ **2. 특징 추출 (Feature Extraction)**

다음의 URL 기반 특징들이 추출됩니다:

* URL 길이, 서브도메인 수, 특수 문자 수
* 도메인 엔트로피, 도메인 내 숫자 포함 여부
* 정상 URL 목록과의 최소 리븐슈타인 거리
* 쿼리 파라미터 평균 길이, Base64 인코딩 포함 여부
* 반복되는 쿼리 키 포함 여부, 디렉토리 깊이
* 알파벳/숫자 비율, 단어 수, TLD (Top-Level Domain), CCTLD (Country Code Top-Level Domain)
* 도메인 이름, IP 주소 여부

### ✅ **3. 데이터 전처리**

* 결측치 처리 및 라벨(`y`)과 특징(`X`) 분리
* 클래스 불균형 해소를 위한 클래스 가중치 계산
* 특성 스케일링: Min-Max Scaler 사용

### ✅ **4. 모델 학습**

* **AutoGluon MultiModalPredictor**를 활용한 학습

  * AutoGluon의 `MultiModalPredictor`를 사용하여 다양한 모델을 테스트하고 최적화
  * 하이퍼파라미터 자동 탐색 및 최적 모델 선택
* **XGBoost 모델 학습**

  * `RandomizedSearchCV`를 사용하여 최적 하이퍼파라미터 탐색
  * 5-Fold 교차 검증을 통한 성능 검증
  * 모델 성능 평가: Precision, Recall, F1-Score

### ✅ **5. 앙상블 학습**

* K-Fold 교차 검증으로 학습된 XGBoost 모델들의 **Soft-Voting 앙상블** 진행
* 각 모델의 예측 확률을 평균하여 최종 예측 생성

### ✅ **6. 예측 및 결과 생성**

* 테스트 데이터에 대한 예측 수행
* 최종 예측 결과를 `submission.csv` 형식으로 저장

---

## 🚀 **실행 방법**

1️⃣ **필요 라이브러리 설치**

```bash
pip install -r requirements.txt
```

2️⃣ **Jupyter Notebook 실행**

```bash
jupyter notebook ML.ipynb
```

3️⃣ **노트북의 셀을 순서대로 실행**

* 데이터 로딩 → 전처리 → 모델 학습 → 앙상블 → 예측 및 제출 파일 생성

4️⃣ **예측 결과 확인**

* 루트 디렉토리에 `submission.csv` 파일이 생성되며, 테스트 데이터에 대한 예측 결과를 담고 있습니다.

---

## 📊 **모델 성능 평가**

| 모델명           | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| AutoGluon     | 0.94      | 0.91   | 0.92     |
| XGBoost (앙상블) | 0.96      | 0.93   | 0.94     |

* AutoGluon 모델과 XGBoost 앙상블 모델을 비교한 결과, **XGBoost 앙상블 모델이 가장 높은 성능**을 보였습니다.

---

## 🛠️ **기술 스택**

| 기술               | 설명                              |
| ---------------- | ------------------------------- |
| **Python**       | 데이터 처리 및 모델 학습                  |
| **Pandas**       | 데이터 로딩 및 전처리                    |
| **NumPy**        | 수치 연산 및 벡터화                     |
| **scikit-learn** | 데이터 전처리, 모델 평가, K-Fold 교차 검증    |
| **AutoGluon**    | MultiModalPredictor를 활용한 모델 학습  |
| **XGBoost**      | 트리 기반의 Boosting 모델, 교차 검증 및 앙상블 |
| **tldextract**   | URL에서 도메인 정보 추출                 |
| **Levenshtein**  | URL 유사도 측정                      |

---

## 📌 **개선 사항 및 향후 계획**

* 데이터셋 확장: 실제 악성 URL과 정상 URL의 다양한 예시를 추가 학습
* 모델 최적화: 하이퍼파라미터 튜닝을 더욱 정교하게 수행
* 실시간 URL 탐지 API 개발: REST API 형태로 확장하여 웹 애플리케이션과 연동
* 이상치 탐지 기법 추가: 악성 URL의 패턴을 더욱 빠르게 탐지할 수 있는 이상치 탐지 모델 추가
  ""
