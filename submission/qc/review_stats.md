# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 235; đối tượng sau duyệt: 273; ngưỡng accepted: IoU >= 0.9
- Đã **loại 17 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 212 | 212/235 = 90.2% | 235 box máy |
| edited (sửa) | 8 | 8/235 = 3.4% | 235 box máy |
| deleted (xoá) | 12 | 12/235 = 5.1% | 235 box máy |
| added (vẽ thêm) | 53 | 53/273 = 19.4% | 273 đối tượng sau duyệt |

Trong số deleted + added, **3 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 50.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 106 | 3 | 3 | 22 |
| bicycle | 4 | 0 | 1 | 3 |
| motorbike | 59 | 4 | 2 | 13 |
| car | 28 | 0 | 0 | 11 |
| bus | 1 | 0 | 1 | 1 |
| truck | 1 | 0 | 1 | 3 |
| three_wheeler | 13 | 1 | 4 | 0 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 61 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 26 | 9 | 35 | 74.3% (KTC 95%: 58–86%) | 42.6% (KTC 95%: 31–55%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 4/11 = 36.4% (KTC 95%: 15–65%) (ngưỡng đạt: 80%). Chỉ 11 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 0/3
  - wrong_class: 1/2
  - extra: 3/3
  - loose: 0/3
  - lọt: img_010.jpg (missing), img_008.jpg (wrong_class), img_023.jpg (loose), img_005.jpg (missing), img_013.jpg (loose), img_003.jpg (loose), img_021.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
