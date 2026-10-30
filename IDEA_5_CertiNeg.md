# Ý tưởng 5 — CertiNeg

**Working title (EN):** *Certified Absence: Risk-Controlled Semantic NOT EXISTS Queries over LLM-Powered Operators with Cost-Aware Two-Stage Audits*
**Tên tiếng Việt:** Chứng nhận "không tồn tại": kiểm soát rủi ro cho truy vấn phủ định ngữ nghĩa trên toán tử LLM

| Mục | Nội dung |
|---|---|
| Hạng (Claude) | **5/5**. Người dùng chưa chọn |
| Nguồn gốc | Tinh chỉnh từ **V2-ANTI** của Codex. Estimand giữ nguyên (phần tốt nhất của Codex); thay đổi chiến lược thành **tổng quát hóa của Ý tưởng 1** sang xử lý truy vấn ngữ nghĩa |
| Trạng thái | **EXPERT HYPOTHESIS**. Chưa có dữ liệu, simulator hay định lý |
| Khả năng kịp ACIIDS 31/10/2026 | **Không.** Cần 6–8 tuần, tốt nhất sau khi Ý tưởng 1 có code audit |
| Session phù hợp | **DADIT** (query processing, data reliability); **DMPI** |
| Đích chính | VLDB/SIGMOD/ICDE nếu có lý thuyết phân bổ và tích hợp plan; VLDB Journal/TKDE. Kiểm tra rank khi chọn |
| Quan hệ với Ý tưởng 1 | Dùng chung engine audit (two-stage, betting test, nested selection). **Rủi ro tương quan**: nếu C4 của Ý tưởng 1 thất bại thì phần phân bổ ở đây cũng yếu. Không được nộp hai bài có đóng góp chồng lấn |

> Không có bảo đảm được nhận bài.

---

## 1. Tóm tắt một đoạn

Các hệ truy vấn ngữ nghĩa (LOTUS, các engine trong SemBench) cho phép viết toán tử như `sem_filter` bằng LLM, rồi kết hợp với group-by và phủ định. Ví dụ: "các camera-tuần **không** ghi nhận loài X", hay "các bài báo **không** có bình luận độc hại". Một false negative duy nhất của toán tử trong một nhóm đủ khiến cả nhóm bị kết luận "sạch" sai. Vì lỗi tương quan trong nhóm, recall mức hàng không quyết định được độ tin cậy mức nhóm. CertiNeg chứng nhận rằng **tỷ lệ nhóm bị kết luận vắng mặt sai** trong output của một truy vấn NOT EXISTS ngữ nghĩa ≤ ε với xác suất ≥ 1−δ. Nó dùng audit hai tầng: lấy mẫu nhóm với xác suất đã biết, và trong nhóm kiểm theo thứ tự proxy rồi dừng ở bằng chứng dương đầu tiên. Hệ cũng chọn tập con lồng nhau lớn nhất có thể chứng nhận, và **tích hợp audit vào plan** (đặt audit ở đâu, tương tác với proxy cascade).

## 2. Estimand (giữ từ V2-ANTI, đã chuẩn hóa)

- Nhóm g = 1..G, bản ghi i trong nhóm, nhãn tham chiếu y_{gi} ∈ {0,1}. Đặt Z_g = 1 nếu tồn tại i có y_{gi} = 1.
- Hệ đề xuất tập A các nhóm "vắng mặt", **đóng băng trước audit**. Khi đó FDP(A) = Σ_{g∈A} Z_g / max(1, |A|).
- Chứng chỉ có dạng P_audit[FDP(A_t̂) > ε] ≤ δ. Đây là validity theo phép lặp thiết kế audit, không phải xác suất hậu nghiệm, và tương đối với nhãn tham chiếu trong corpus đã ghi (không nói về thế giới thật ngoài corpus).
- Giới hạn cơ bản (không quảng bá là mới): với ε = 0 và không có giả định thêm, phải quét toàn bộ.

## 3. Công trình gần nhất và ranh giới

| Công trình | Đã làm gì | Ranh giới | Mức xác minh |
|---|---|---|---|
| **Evergreen** (arXiv 2604.26180) | Kiểm chứng quantified claims (nhóm, lượng từ, so sánh), early stopping, relevance sorting, confidence sequences, provenance | **Đối thủ nguy hiểm nhất.** CertiNeg phải chứng minh hơn Evergreen + stratified sampling chuẩn ở cùng ngân sách, nhờ thiết kế hai tầng bất đối xứng, tập lồng nhau và phân bổ theo chi phí | Handoff đã đọc full HTML |
| **When Plans Change Answers** (arXiv 2610.08089) | Hình thức hóa chất lượng mức output, fan-out, calibration phụ thuộc plan, contribution-weighted PPI. Lõi là positive RA; tác giả nêu negation là hướng mở rộng | CertiNeg nhắm đúng phần phủ định với ngữ nghĩa nhóm | SOURCE-REPORTED (handoff) |
| **LOTUS** (PVLDB 18(11), 2025) | Toán tử ngữ nghĩa + proxy cascade với mục tiêu chất lượng mức toán tử | Chất lượng mức toán tử khác mức nhóm phủ định | AGENT-REPORTED |
| **SemBench** (VLDB 2026) | Harness so sánh engine truy vấn ngữ nghĩa | Nguồn workload; cần mở rộng ngữ nghĩa nhóm và ghi rõ là mở rộng của dự án | AGENT-REPORTED |
| RLA (SHANGRLA, ALPHA, SUITE); conformal selection; PPI | Công cụ thống kê nền | Mượn và trích dẫn | Kiến thức nền |
| **Ý tưởng 1 (CertiClean)** | Cùng estimand trong làm sạch dữ liệu (dòng là nhóm ô) | Bài này phải có đóng góp **mới so với Ý tưởng 1**: nhóm kích thước dị biệt lớn, tích hợp plan, toán tử ngữ nghĩa đa phương thức, lý thuyết phân bổ | — |

**Đóng góp cần có để đáng một bài riêng:**
1. Lý thuyết phân bổ cho nhóm có kích thước rất lệch (camera có hàng nghìn ảnh so với vài ảnh). Chi phí âm tính tỷ lệ với kích thước nhóm, nên tối ưu khác hẳn trường hợp dòng/ô.
2. Tích hợp vào plan: audit đặt sau proxy cascade; tương tác giữa ngưỡng cascade và chi phí chứng nhận; plan tối thiểu chi phí tổng ở mức chứng chỉ cho trước.
3. Workload thật có nhóm tự nhiên, nhãn tham chiếu, và báo cáo trung thực khi không chứng nhận được.

## 4. Dữ liệu (các cổng dữ liệu phải qua trước)

| Nguồn | Nhóm tự nhiên | Trạng thái |
|---|---|---|
| iWildCam (WILDS) | camera hoặc camera × khoảng thời gian; vị từ "có loài X" | ID camera và nhãn loài đã được xác nhận qua tài liệu WILDS (handoff). **Timestamp/granularity UNVERIFIED** |
| SemBench | Các task filter/join có nhãn; mở rộng với ngữ nghĩa nhóm | Chưa cài |
| Civil Comments | bài báo (article ID) | **Data gate:** bản HF của Google không có article ID trong các trường liệt kê (agent báo). Không bịa nhóm |
| Amazon reviews | sản phẩm | **Không dùng** star rating làm nhãn khiếu nại (không hợp lệ) |
| RAHA (từ Ý tưởng 1) | thực thể (ProviderNumber, mã chuyến bay) | Cầu nối: cùng engine, kiểm tra tổng quát hóa |

Proxy: image encoder đóng băng (zero-shot) cho iWildCam; LLM nhỏ đóng băng cho văn bản; dự đoán được cache; mô phỏng audit chạy CPU.

## 5. Thiết kế thực nghiệm

- **Baseline:** quét toàn bộ có nhãn (tham chiếu); anti-join thuần proxy; cascade kiểu LOTUS; kiểm chứng kiểu Evergreen; audit nhóm đều; stratified/Neyman theo chi phí; CertiNeg.
- **Chỉ số:** tần suất chứng chỉ sai qua R lần audit (≤ δ); tỷ lệ nhóm hữu ích được trả về; số nhãn tin cậy; chi phí proxy; latency; tỷ lệ unresolved.
- **Kịch bản:** độ lệch kích thước nhóm; một so với nhiều positive mỗi nhóm; nhãn hiếm; camera/domain chưa thấy; ε = 0 để thấy chi phí tăng trung thực.
- **Nguyên tắc:** không chỉ dùng tiêm một-positive synthetic làm kết quả chính; tách validity trên một corpus khỏi khái quát hóa sang corpus khác.

## 6. Điều kiện tiếp tục / dừng

| Cổng | Kiểm tra | Nếu thất bại |
|---|---|---|
| G1 | Có ≥ 2 workload nhóm tự nhiên với nhãn tham chiếu | Dừng; không bịa nhóm |
| G2 | Lệch giữa recall mức hàng và FDP mức nhóm tái hiện được trên dữ liệu thật | Động lực yếu |
| G3 | CertiNeg thắng Evergreen-style + stratified/Neyman ở cùng ngân sách và validity | Không có đóng góp phương pháp. Định vị lại thành bài phân tích/benchmark, hoặc gộp vào bản journal của Ý tưởng 1 |

## 7. Kế hoạch (6–8 tuần, sau Ý tưởng 1)

Tuần 1: data gates (iWildCam timestamp, SemBench). Tuần 2: proxy + cache, tái hiện lệch hàng/nhóm. Tuần 3–4: chuyển engine audit từ Ý tưởng 1, thêm phân bổ theo kích thước nhóm. Tuần 5: tích hợp plan/cascade. Tuần 6: lý thuyết (validity, chi phí, phân bổ). Tuần 7–8: thí nghiệm đầy đủ, viết.

## 8. Phản biện dự kiến

| Phản biện | Trả lời |
|---|---|
| "Evergreen đã làm" | So sánh trực tiếp ở cùng ngân sách. Đóng góp chỉ được tuyên bố trên phần thắng đo được (phân bổ, tập lồng nhau, plan) |
| "Chứng nhận trên tập cố định là thống kê cổ điển" | Đồng ý, và trích dẫn. Phần mới là phân bổ hai tầng bất đối xứng và tích hợp plan |
| "Nhãn tham chiếu không phải sự thật" | Chứng chỉ phát biểu tương đối nhãn tham chiếu và corpus đã ghi |
| "Trùng với bài data cleaning trước" | Nêu rõ quan hệ; đóng góp mới nằm ở lý thuyết nhóm dị biệt, plan và workload ngữ nghĩa |

## 9. Rủi ro

Data gates; cạnh tranh Evergreen; rủi ro tương quan với Ý tưởng 1; đòi hỏi năng lực thống kê. Xếp hạng 5 vì giá trị A* tiềm năng cao nhưng phụ thuộc vào Ý tưởng 1 và không phục vụ deadline ACIIDS.

## 10. Tài liệu tham khảo cốt lõi

- Evergreen: https://arxiv.org/abs/2604.26180 — handoff
- When Plans Change Answers: https://arxiv.org/abs/2610.08089 — handoff
- LOTUS: https://arxiv.org/abs/2407.11418 — AGENT-REPORTED venue
- SemBench: https://github.com/SemBench/SemBench/ — AGENT-REPORTED venue
- WILDS iWildCam: https://wilds.stanford.edu/datasets/ ; https://github.com/p-lambda/wilds/blob/main/wilds/datasets/iwildcam_dataset.py
- Civil Comments (HF): https://huggingface.co/datasets/google/civil_comments — data gate
