# 13조 잡음처리반 — 음성 잡음 제거 챌린지 (2차)

| 항목 | 내용 |
|---|---|
| 팀 / 회차 | 13조 잡음처리반 / 2차 챌린지 |
| 제출 버전 | 태그 `{{SUB_TAG}}` / 커밋 `{{SUB_COMMIT}}` |
| 모델 | `gru_v8p2zw80` (BiGRU 마스크) — 가중치 `weights/gru_v8p2zw80.pt`, 설정 `configs/submission.yaml` |
| Validation | PESQ-WB 3.6393 / SNR 1.06 dB (164개 파일, `pesq==0.0.4`) |

## 설치

Python 3.12

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

`input/`의 WAV 파일을 처리해 `output/`에 같은 파일명으로 저장하며 (16 kHz 모노, 입력과 같은 샘플 수, CPU만 사용, 절대 경로·다운로드 없음), 폴더는 `python main.py --input <입력 폴더> --output <출력 폴더>`로 바꿀 수 있습니다.

## 학습 요약

- 가중치: 미세조정한 두 가중치의 혼합 (WiSE-FT) = 0.2 × `gru_v8p2z3e` (챌린지 train split만 학습) + 0.8 × `gru_v8p2z6a` (`gru_v8p2z3e`를 외부 데이터로 이어 학습), 혼합 후 추가 학습 없음.
- 외부 데이터: `gru_v8p2z6a` 학습에 LibriSpeech train-clean-100 부분 집합 (clean 음성, CC BY 4.0)과 DEMAND / ESC-50 / MUSAN 잡음의 train 분할 (CC BY-SA 3.0 / CC BY-NC 3.0 / CC BY 4.0)을 사용했습니다 (출처·분할·용도: 학습 코드 저장소의 `docs/external_data.md`). 사전학습 가중치는 쓰지 않았고 validation은 가중치 선택에만 사용했습니다.
- 손실: 0.2 × cmag (압축 크기 손실) + 0.32 × torch-pesq (미분 가능한 PESQ), 학습에만 사용.
- 증강: 동적 혼합 / 외부 잡음 혼합 (SNR −10 ~ 20 dB), 속도 변환, 무작위 EQ.
- 설정 (`gru_v8p2z6a` 학습, `gru_v8p2z3e`에서 시작): Adam lr 5e-5, batch 16, seed 7, 300 epoch (best 298), Google Colab T4 GPU.
- 선택: 혼합 비율 α = 0.8은 validation과 외부 잡음 dev 세트로만 골랐습니다. 직전 후보 `gru_v8p2z3e` 대비 처음 보는 잡음 평가 세트 +0.233 PESQ, validation −0.014, val_remix −0.005.
- 학습 코드·실험 기록: https://github.com/pauljh4323/SGSYS_13thTeam_nrc
