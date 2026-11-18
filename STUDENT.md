# Thông tin học viên

- Họ và tên: Nguyễn Huy Vượng
- MSHV: 2A202603133
- Lớp (L2 / L3 / khác): L23
- Tài khoản GitHub: Virgoboyy
- Repo bài nộp (URL): https://github.com/Virgoboyy/K5-L23-Day08-NguyenHuyVuong-2A202603133-AI-Assisted-Labeling
- Commit chốt bài (hash): xem commit mới nhất trên nhánh main
- Máy: Windows, CPU (không có GPU CUDA)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 212/235 (90.2%) / 8/235 (3.4%) / 12/235 (5.1%) (mẫu số: 235 box máy); added: 53/273 (19.4%) (mẫu số: 273 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 26/35 (74.3%) / 26/61 (42.6%) (mẫu số: 35 box máy / 61 đối tượng gold)
- Bắt bẫy: 4/11 = 36.4% (KTC 95%: 15–65%, 11 bẫy được cài hợp lệ từ yolo26n)
- Thời gian: gán tay 7.0 phút/ảnh (28 phút / 4 ảnh); duyệt có pre-label 2.7 phút/ảnh (65 phút / 24 ảnh)
- Lỗi QC: 80.0% theo ảnh (4/5 ảnh), 8.3% theo đối tượng (5/60 đối tượng)
- **Quyết định:** Rework — lý do một dòng: Ảnh có lỗi QC 4/5 (80.0% > 2%) và tỷ lệ bắt bẫy 4/11 (36.4% < 80%) do thiên kiến tự động hóa (lọt 3/3 missing và 3/3 loose), cần duyệt lại lô theo checklist 4 bước.
