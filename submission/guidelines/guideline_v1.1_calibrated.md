# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603133 - Nguyễn Huy Vượng

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: bicycle hay motorbike? | Xe có bàn đạp trợ lực tính là `bicycle`; xe không có bàn đạp dáng scooter tính là `motorbike` | img_002.jpg | CAL-02 |
| 3 | Hàng nhô ra: xe đạp / ba gác chở hàng nhô xa: box xe có gồm hàng không? | Chỉ tính phần hàng nhô ra ≤ 20% chiều dài xe vào box xe; phần nhô vượt quá 20% thì không kéo box theo | img_004.jpg | CAL-03 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_013.jpg | Xe tuk-tuk Thái Lan có mui che và khách ngồi trong | YOLO26 gọi là `car` hoặc `truck` (YOLO26-n) | Sửa lại thành `three_wheeler`, vẽ thêm box `person` cho người lái nếu thấy rõ | Bổ sung Mục 2: Xe ba bánh chở khách/hàng dù có mui gắn liền vẫn là `three_wheeler` (CAL-01) |
| img_003.jpg | Box xe ba bánh bị lỏng biên dưới/phải (bẫy `loose`) và sót người/xe ở xa (bẫy `missing`) | YOLOE/pre-label tạo box rộng hoặc bỏ sót đối tượng | Siết chặt lại mép box ôm sát bánh và khung xe; ẩn nhãn máy quét nền trước khi duyệt | Bổ sung Mục 4.1.1: Kiểm tra độ ôm sát cạnh box (≤ 2 px) trên vật lớn và bắt buộc quét ảnh trước khi bật pre-label (CAL-04) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Bổ sung quy định xe điện, hàng hóa nhô ra khỏi xe, chuẩn hóa nhận diện xe ba bánh địa phương và siết kiểm tra mép box/vật sót | CAL-01, CAL-02, CAL-03, CAL-04 |
