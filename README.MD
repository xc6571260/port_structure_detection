# Port Structure Damage Detection - 港口設施破損自動化檢測平台

本專案針對港口設施（如消波塊、胸牆等）影像，進行自動化**混凝土剝落**與**鋼筋裸露**損傷檢測。採用 UAV 拍攝高解析度影像，並利用 YOLOv8 語意分割模型實現破損辨識。全流程支援 GPU（CUDA 12.4），並可一鍵部署。

---

## 🌊 平台功能特色

- 支援混凝土剝落（Surface spalling）、鋼筋裸露（Rebar exposure damage）兩類損傷自動檢測
- 支援 GPU 與 CPU 雙模式推論，兼容多種硬體環境
- 可批次處理多張港區設施影像
- 分割及標註結果自動儲存於 output 資料夾
- Docker 一鍵部署，復現環境零困難

---

## 🛠️ 工作流程

1. 將需檢測的原始影像（建議解析度 4000x3000 或以上）放入 `input/` 資料夾
2. 影像自動切割為 1024x1024 小塊
3. 使用 YOLOv8 分割模型預測損傷區域
4. 合併預測結果，將分割區疊加至原圖，並標示類別、檢測數量
5. 輸出帶有標註之結果圖至 `output/` 資料夾
6. 終端自動列印各影像損傷統計（如：  
   `[INFO] image_name: IMG_0001.jpg detection_number: 5 Surface spalling: 2 Rebar exposure damage: 3`）

---

## ⚙️ 部署說明（Docker + CUDA 12.4）

### 1. 下載專案檔案

```bash
git clone <your-github-repo-url>
cd port_structure
```