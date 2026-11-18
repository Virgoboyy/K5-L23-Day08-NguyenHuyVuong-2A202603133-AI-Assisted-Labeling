# Báo cáo Day 08 — 2A202603133 - Nguyễn Huy Vượng

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào).

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26-n (`yolo26n.pt`, `--imgsz 640`) | `0.3` (mặc định theo `qc/threshold_sweep.md`) | Trên máy CPU, `yolo26n` chạy nhanh (~0.15–4.46s/ảnh). Ở `r02` (`conf 0.25`) có 120 box (recall cao nhưng có box thừa), sang `r03` (`conf 0.5`) giảm còn 54 box (mất nhiều người/xe nhỏ ở `img_002`, `img_011`). Khi quét ngưỡng trên 4 ảnh gold không có ngưỡng nào ≤ 0.5 đạt recall 90% (tối đa person 50%, motorbike 44%, car 86%), còn `bicycle`, `bus`, `truck` có < 5 vật nên giữ ngưỡng mặc định `0.3`. |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`), prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` | `0.3` (mặc định theo `qc/threshold_sweep.md`) | Ở `r04`, YOLOE bắt được xe ba bánh đặc trưng Đông Nam Á mà YOLO26 COCO nhận nhầm sang `car`/`truck`. Khi gộp ưu tiên đặt YOLOE trước để ghi đè box trùng của YOLO26 với IoU ≥ 0.6. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn:** **SAM 2.1 tiny (`sam2.1_t.pt`)** (dẫn `r05` trong `model_log.csv`). Lý do: máy chỉ chạy CPU (không có GPU NVIDIA/MPS), bản tiny nhẹ (~78 MB, ~4.83s/ảnh trên CPU), không yêu cầu xin quyền gated trên HuggingFace như SAM 3 nhưng vẫn cho biên mask sắc nét quanh xe ba bánh và người khi có đủ click dương/âm.
- Lỗi nào **không sửa được bằng đổi ngưỡng**? Lỗi **thiếu lớp trong danh mục đóng (closed-set)** của YOLO26: gọi tuk-tuk / xe ba gác là `car` hoặc `truck` (ví dụ trên `img_013.jpg`, `img_017.jpg`). Tăng hay giảm ngưỡng chỉ làm mất box hoặc tăng box thừa chứ không biến thành `three_wheeler`. Cách sửa: dùng model open-vocabulary (`YOLOE`) với prompt chữ (`tuk-tuk`, `cyclo`, `cargo tricycle`) và chạy `merge_prelabels.py` ưu tiên YOLOE đè lên YOLO26.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**:
  - Điểm cao mà sai: trên `img_031.jpg`, box `#252` gán `person` (`conf 0.46`) vào sạp áo khoác treo tường; ở `r02` YOLO26 gọi tuk-tuk là `car`/`truck` với độ tin cậy cao.
  - Điểm thấp mà đúng: trên `img_006.jpg` và `img_035.jpg`, nhiều người ngồi ăn hoặc xe máy đỗ sát nhau ở xa có điểm tin cậy dưới `0.3` nên bị lọc bỏ khi merge (`y26_all:person(dưới ngưỡng)×325`, `motorbike×172`).
- Giới hạn của việc chọn ngưỡng: Quét ngưỡng được chạy trên **chính 4 ảnh gold** (61 đối tượng, `submission/qc/threshold_sweep.md`). Vì chọn ngưỡng và đánh giá trên cùng một tập dữ liệu nhỏ (in-sample), các con số Precision/Recall của máy trên tập gold là **lạc quan**; các lớp ít vật (`bicycle`: 4, `truck`: 1, `bus`: 0) không đủ mẫu thống kê để chọn ngưỡng riêng.

## 2. AI error profile (từ `qc/review_stats.md`)

*(Tính trên 24 ảnh batch; đã loại 17 đối tượng nằm quanh các bẫy nhân tạo)*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 212 | 212/235 = 90.2% | 235 box máy |
| edited (sửa vị trí/kích thước) | 8 | 8/235 = 3.4% | 235 box máy |
| deleted (xoá) | 12 | 12/235 = 5.1% | 235 box máy |
| added (vẽ thêm) | 53 | 53/273 = 19.4% | 273 đối tượng sau duyệt |

*(Trong số deleted + added có 3 ca là đổi lớp -> số đối tượng máy bỏ sót thật ≈ 50).*

- **Lớp nào máy tệ nhất? Vì sao?**
  - Về **bỏ sót (FN)**: `person` (thêm 22 box trên 106 accepted), `motorbike` (thêm 13 box trên 59 accepted) và `car` (thêm 11 box trên 28 accepted) — đặc biệt trong các cảnh phố đông hoặc người ngồi sâu trong quán/vỉa hè (`img_006.jpg`, `img_035.jpg`) và dùng `yolo26n` với `--imgsz 640` làm giảm khả năng bắt vật nhỏ.
  - Về **nhận nhầm / box thừa (FP)**: `three_wheeler` có tỉ lệ xóa cao nhất (`4/18` box máy bị xóa), do YOLOE nhận nhầm một số xe máy hoặc xe tải nhỏ ở góc khuất thành xe ba bánh, đồng thời hay tạo box hơi lỏng (như `img_003.jpg`).
- **So với gold (nhãn gán tay từ đầu, `submission/labels/gold_manual.json`):**
  - Precision của máy: `26/35 = 74.3%` (KTC 95%: 58–86%, mẫu số: 35 box máy dự đoán trên 4 ảnh gold).
  - Recall của máy: `26/61 = 42.6%` (KTC 95%: 31–55%, mẫu số: 61 đối tượng gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy (từ `qc/review_stats.md`):** Bắt được **4/11 = 36.4%** (KTC 95%: 15–65%, không đạt ngưỡng 80%).
  - `extra` (box ma): **3/3** (100%) — phát hiện tốt các box nằm vô lý giữa nền đường/khoảng trống.
  - `wrong_class` (sai lớp): **1/2** — lọt `img_008.jpg`.
  - `missing` (bỏ sót): **0/3** — lọt `img_005.jpg`, `img_010.jpg`, `img_021.jpg`.
  - `loose` (box lỏng/lệch): **0/3** — lọt `img_003.jpg`, `img_013.jpg`, `img_023.jpg`.
  - **Vì sao lọt?** Đã bỏ qua bước **ẩn toàn bộ nhãn máy để quét ảnh mắt thường trước** (dẫn tới bỏ sót 3/3 bẫy `missing` trong các ảnh đông xe máy như `img_005`, `img_021`) và không zoom kiểm tra lại 4 cạnh của những box có sẵn trông "có vẻ đúng" (dẫn tới lọt 3/3 bẫy `loose`).
- **Accepted cao (`212/235 = 90.2%`) nhưng bắt bẫy thấp (`4/11 = 36.4%`) nói lên điều gì?** Đây là biểu hiện rõ rệt của **thiên kiến tự động hóa (automation bias)**: vì nhìn thấy đa số box máy gợi ý khớp sẵn vị trí nên người duyệt tin tưởng, bấm chấp nhận nhanh mà không kiểm tra kỹ độ ôm sát cạnh box (`loose`) cũng như không chủ động quét tìm vật bị bỏ sót (`missing`).

## 4. Máy có đáng dùng không?

- **Thời gian (từ `submission/time_log.csv`):**
  - Gán tay (`gold_manual`): `28 phút / 4 ảnh = 7.0 phút/ảnh`.
  - Duyệt có pre-label (`batch_review`): `65 phút / 24 ảnh = 2.7 phút/ảnh` (nhanh hơn ~2.6 lần).
  - **Thiên lệch so sánh thời gian:** Do đã gán tay 4 ảnh gold trước nên khi sang lô batch tôi đã quen giao diện CVAT và luật guideline hơn; đồng thời tốc độ duyệt nhanh 2.7 phút/ảnh một phần phải trả giá bằng việc bỏ lọt bẫy (`4/11`) và để sót lỗi (tỉ lệ ảnh lỗi QC `4/5`). Nếu duyệt đủ kỹ theo checklist 4 bước thì thời gian duyệt thực tế sẽ vào khoảng ~3.5–4 phút/ảnh.
- **Kết luận:** **Có đáng dùng cho `person`, `motorbike`, `car` và hỗ trợ tìm `three_wheeler`** vì tiết kiệm đáng kể thao tác vẽ lại 212 box đã đúng (`212/235 = 90.2%` accepted), **nhưng tuyệt đối không thể bỏ qua bước kiểm duyệt độc lập** vì Recall của máy `yolo26n` trên gold chỉ đạt `26/61 = 42.6%` (còn sót nhiều vật nhỏ/bị che).
- **Ba quy tắc giảm thiên kiến của tôi:**
  1. **Ẩn nhãn máy 20 giây đầu cho mỗi ảnh:** Luôn nhấn phím tắt ẩn toàn bộ box pre-label, đếm nhanh số đối tượng chính (đặc biệt khu vực vỉa hè, hàng quán, xe 3 bánh) rồi mới bật nhãn máy lên đối chiếu để chặn lỗi `missing`.
  2. **Kiểm tra zoom 4 cạnh với mọi box lớn (> 150 px) và lớp `three_wheeler`:** Bắt buộc kiểm tra độ ôm sát ≤ 2 px cho các box xe lớn từ YOLOE/YOLO26 để loại triệt để lỗi `loose`.
  3. **Rà soát riêng nhóm `conf` trung bình/thấp (`0.30 – 0.55`) và nghỉ giải lao mỗi 8 ảnh:** Lọc box `conf < 0.55` để kiểm lỗi `thua` (như áo khoác treo thành `person` ở `img_031.jpg`) và dừng nghỉ 2 phút sau mỗi 8 ảnh để tránh mỏi mắt gây "bấm Accept theo quán tính".

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 tiny (`sam2.1_t.pt`) | 3 (2 dương + 1 âm) | Đã tách bỏ được người lái áo xanh ngồi ghế trước nhờ 1 click âm (`720,300`); giữ nguyên mui và thân xe (275,125 px, 39.4% ảnh). | Mask bị loang nhẹ sang khách mặc áo hồng ngồi ở quán phía sau khung hở vì cùng tông màu với mui xe tuk-tuk hồng. | `submission/masks/tuktuk_img013.jpg` |
| `img_004.jpg` | SAM 2.1 tiny (`sam2.1_t.pt`) | 5 (3 dương + 2 âm + box gợi ý) | Tách hoàn toàn khỏi người đàn ông đứng bên phải và xe tuk-tuk xanh bên cạnh; bao trọn mui, kính chắn gió và 3 bánh (95,703 px, 13.7% ảnh). | Còn một lỗ nhỏ màu tối dưới khung sắt bên hông chưa khép kín hoàn toàn, nhưng viền tổng thể bám rất sát. | `submission/masks/cyclo_img004.jpg` |
| `img_017.jpg` | SAM 2.1 tiny (`sam2.1_t.pt`) | 5 (3 dương + 2 âm + box gợi ý) | Tách sạch chiếc tuk-tuk "MR.K." khỏi xe kế bên và nền cửa hàng quần áo phía sau (88,580 px, 12.7% ảnh). | Bỏ sót phần biển hộp đèn `TAXI` nhỏ trên đỉnh mui và còn khoảng trống hở qua khung sắt phía trên ghế sau. | `submission/masks/tuktuk_img017.jpg` |

- **Đối chiếu guideline mục 4.2:**
  - *Quy tắc giúp:* Quy tắc không tính bóng đổ, gộp kính chắn gió vào thân xe (mục 4.2.3) và bỏ qua lỗ nhỏ < 5% diện tích vật (mục 4.2.2) giúp đánh giá nhanh độ đạt chuẩn của mask xe tuk-tuk.
  - *Quy tắc cần bổ sung:* Khi sử dụng SAM trên xe có khung hở (tuk-tuk, xích lô), cần làm rõ quy tắc xử lý người ngồi phía sau khung hở trùng màu thân xe và các chi tiết gắn liền trên nóc (biển hiệu `TAXI`, khung mái che).
- **Interactor SAM trong CVAT (tuỳ chọn):** Chưa triển khai serverless Nuclio cho CVAT Docker do giới hạn RAM/CPU trên máy Windows, nên chạy toàn bộ qua script offline `scripts/sam2_points.py` để mô phỏng tương tác click dương/âm. Điểm khác biệt chính là offline phải ước lượng tọa độ pixel `(x, y)` qua nhiều lần thử thay vì click trực tiếp xem phản hồi tức thì trên canvas như Interactor của CVAT.

## 6. Calibration (Part 6)

- **Số quyết định:** `4` quyết định (`CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`) ghi trong `submission/qc/calibration_decisions.csv`.
  - Quan trọng nhất: **`CAL-01`** (`nguon = loi_may`, `img_013|img_017`) chuẩn hóa dứt điểm việc sửa lỗi YOLO26 gán nhầm tuk-tuk/ba gác sang `car`/`truck` thành `three_wheeler`, và **`CAL-04`** (`nguon = bay_lot`, `img_003|img_013|img_023`) bổ sung quy trình ẩn nhãn máy và kiểm tra mép box sau khi phát hiện lọt bẫy `loose`/`missing`.
- **Quy tắc mới đã vào `submission/guidelines/guideline_v1.1_calibrated.md`:**
  1. Chốt câu hỏi mở #1 (`CAL-02`, `img_002.jpg`): xe điện có bàn đạp trợ lực → `bicycle`; xe máy điện không bàn đạp dáng scooter → `motorbike`.
  2. Chốt câu hỏi mở #3 (`CAL-03`, `img_004.jpg`): hàng chở trên xe chỉ tính vào box xe nếu nhô ra ≤ 20% chiều dài xe; nhô quá 20% không kéo box theo.
  3. Bổ sung ca khó xe ba bánh có mui (`CAL-01`, `img_013.jpg`) và quy định kiểm tra mép box ≤ 2 px + quét nền trước khi bật nhãn máy (`CAL-04`, `img_003.jpg`).
- **Nguyên nhân gốc:** Kết hợp cả 3 yếu tố — **(1) máy** thiếu lớp `three_wheeler` và dùng `yolo26n` nên sót nhiều vật nhỏ (`CAL-01`), **(2) guideline v1.0** còn câu hỏi mở về xe điện và hàng nhô ra (`CAL-02`, `CAL-03`), và **(3) quy trình người duyệt** bị thiên kiến tự động hóa khi tin vào box máy có sẵn (`CAL-04`).

## 7. QC và quyết định (Part 6)

*(Số liệu từ `submission/qc/qc_review.csv` và `submission/qc/qc_report_auto.md`)*

- **Mẫu QC:** `5/24` ảnh batch (`0` ảnh `can_review` + `5` ảnh ngẫu nhiên với `seed 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), tổng cộng **60 đối tượng**. Người kiểm: Nguyễn Huy Vượng (**hạn chế:** tự kiểm lại chính lô mình vừa duyệt nên vẫn tồn tại rủi ro điểm mù giống nhau).
- **Lỗi theo đối tượng:** `5/60 = 8.3%` (mẫu số: 60 đối tượng trong mẫu QC).
- **Lỗi theo ảnh:** `4/5 = 80.0%` (mẫu số: 5 ảnh trong mẫu QC; vượt xa ngưỡng Release ≤ 2%).
- **Top 3 loại lỗi phát hiện trong mẫu QC:**
  1. `sai_vi_tri_kich_thuoc` (2 lỗi): `img_003.jpg` (box `#1 three_wheeler` lỏng mép dưới/phải), `img_031.jpg` (box `#251 person` bị cắt ngang hông thiếu phần chân nhìn thấy qua khung xe).
  2. `bo_sot` (2 lỗi): `img_006.jpg` (bỏ sót người ngồi ăn quay lưng giữa `#49` và `#51`), `img_035.jpg` (bỏ sót nhóm người ngồi ghế nhựa và xe máy đỗ vỉa hè phía sau `#282`).
  3. `thua` (1 lỗi): `img_031.jpg` (box `#252 person` nhận nhầm áo khoác treo trên sạp hàng).
- **Giới hạn mẫu nhỏ:** Với `4/5` ảnh lỗi, **cận trên khoảng tin cậy 95% (Wilson)** của tỉ lệ ảnh lỗi thật lên tới **96%** (`submission/qc/qc_report_auto.md`). Ngay cả khi mẫu 5 ảnh không có lỗi nào (0/5), cỡ mẫu 5 ảnh vẫn quá nhỏ để chứng minh thống kê rằng tỉ lệ lỗi thật của toàn lô ≤ 2%.
- **Script kiểm được / không kiểm được:**
  - *Kiểm được:* cấu trúc JSON, tên lớp hợp lệ theo `taxonomy.json`, box nằm trong khung ảnh, cảnh báo box nhỏ < 20 px, và đối chiếu khớp tọa độ với file `traps_key.json`.
  - *Không kiểm được:* box có ôm sát biên thực tế hay không (ví dụ `#251` trên `img_031.jpg` bị cụt chân), người duyệt có cùng bỏ sót đối tượng với máy hay không (ví dụ khách ngồi quán trên `img_006.jpg`, `img_035.jpg`), và người duyệt có tuân thủ checklist quét ảnh trước hay không.

**Quyết định: REWORK**, ngày `2026-11-18`.
- **Lý do (đi theo cây quyết định Hold → Rework → Release):**
  - Không `Hold` vì dữ liệu ảnh đạt chuẩn và các điểm mơ hồ chính đã được chốt trong `guideline_v1.1_calibrated.md`.
  - Phải chọn **`REWORK`** vì lô dữ liệu vi phạm trực tiếp cả 2 ngưỡng chất lượng bắt buộc: tỉ lệ ảnh có lỗi QC là **`4/5 = 80.0% > 2%`** và tỉ lệ bắt bẫy chỉ đạt **`4/11 = 36.4% < 80%`** (lọt 3/3 `missing` và 3/3 `loose`). Nếu báo `Release` sẽ đưa nhãn kém chất lượng và nhiều sót/lỏng vào tập huấn luyện.
- **Việc cần làm tiếp cho vòng Rework:**
  1. Mở lại Task `batch-review` trong CVAT, áp dụng nghiêm ngặt 3 quy tắc chống thiên kiến (ẩn nhãn máy 20 giây đầu để quét bổ sung vật sót, zoom kiểm tra mép box lớn và lọc kiểm tra các box `conf < 0.55`).
  2. Sửa toàn bộ các lỗi đã phát hiện trên `img_003.jpg`, `img_006.jpg`, `img_031.jpg`, `img_035.jpg` cùng các ảnh có bẫy lọt (`img_005`, `img_008`, `img_010`, `img_013`, `img_021`, `img_023`), xuất file `submission/labels/qc_fixed.json`.
  3. Chạy lại `build_release.py`, lấy mẫu QC mới với `qc_sample.py --force` và kiểm tra lại để đảm bảo bắt bẫy ≥ 80% trước khi xem xét Release.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Có sử dụng trợ lý AI (Antigravity) hỗ trợ liên kết môi trường `.venv`, chạy các lệnh tự động hóa script Python (`prelabel.py`, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `sam2_points.py`, `review_stats.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`), kiểm tra trực quan các ảnh kết quả và tổng hợp số liệu từ file QC vào báo cáo. Toàn bộ việc gán nhãn tay 4 ảnh gold và duyệt 24 ảnh batch trên CVAT đều được tự thực hiện trực tiếp; mọi số liệu trong báo cáo đều được kiểm chứng lại từ file đầu ra thực tế (`review_stats.md`, `qc_report_auto.md`).
- **Gold có bị nhiễm không?** **Không.** 4 ảnh gold (`img_001`, `img_012`, `img_020`, `img_036`) được gán tay 100% trong Task `gold-manual` trên CVAT ở Part 1 trước khi chạy bất kỳ model pre-label nào.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:** Máy Windows chạy CPU (không có GPU CUDA), Python 3.10.11 (dùng chung `venv` qua Junction Link `.venv`), CVAT Docker `v2.81.0`. Vướng mắc gặp phải: lần đầu chạy `YOLOE` bị lỗi thiếu thư viện `clip` và chưa có `mobileclip2_b.ts`, đã khắc phục bằng cài `git+https://github.com/ultralytics/CLIP.git` và tải `mobileclip2_b.ts`; chọn bộ model nhẹ `--set small` (`yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`) kèm `--imgsz 640` để chạy ổn định trên CPU.
- **Dùng cache `data/cache/` thay vì tự chạy?** **Không.** Tất cả các lệnh suy luận YOLO26, YOLOE, SAM 2.1 đều được tự chạy trực tiếp trên máy local.
- **Lỗi / bug script đã gặp (nếu có):** `merge_prelabels.py` báo lỗi tham số nếu truyền chuỗi rỗng `--thr ""` (theo dòng gợi ý từ `threshold_sweep.py` khi không có lớp nào có ngưỡng riêng), khắc phục bằng cách bỏ cờ `--thr` và dùng `--conf 0.3`. Ngoài ra `inject_traps.py` chỉ cài được 11/12 bẫy (thiếu 1 bẫy `wrong_class`) do các box từ `yolo26n` ở ảnh được chọn không đủ ngưỡng `score >= 0.5`.
- **Một điều tôi sẽ làm khác lần sau:** Bắt buộc bật thói quen **ẩn hoàn toàn nhãn máy để quét ảnh bằng mắt thường trước** khi xem gợi ý AI, và sử dụng model cỡ lớn hơn (`yolo26m` trên Google Colab GPU) thay vì `yolo26n` `--imgsz 640` để tăng Recall từ 42.6% lên cao hơn trước khi đưa vào duyệt.
