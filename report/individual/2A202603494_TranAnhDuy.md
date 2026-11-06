# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603494
- **Họ và tên:** Trần Anh Duy
- **SEED** (in ở ô đầu notebook): 475

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp | F1 test | F1 SUB 40 | p95 ms/tin | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.893 | 0.947 | 0.100 | 0 |
| BoW + MLP | 0.932 | 0.974 | 0.357 | 0 |
| MiniLM + kNN | 0.913 | 1.000 | 35.663 | 0 |
| MiniLM + LogReg | 0.946 | 0.974 | 29.409 | 0 |
| MiniLM + MLP head | 0.946 | 0.974 | 28.343 | 0 |
| LLM nhỏ zero-shot | — | 0.950 | 2429.331 | 0.004 |
| LLM lớn zero-shot | — | 0.927 | 2919.139 | 0.008 |
| LLM nhỏ few-shot | — | 0.950 | 3155.071 | 0.015 |

## 3. Dự đoán và thực tế

1. Dự đoán MLP trên BoW sẽ kém LogReg vì mạng có nhiều tham số hơn. Thực tế MLP đạt F1 test **0.932**, cao hơn LogReg **0.893**. Trong lần chạy này, hidden layer vẫn học được thêm những kết hợp giữa các từ.
2. Dự đoán LLM lớn sẽ tốt hơn LLM nhỏ. Trên SUB 40 tin, model nhỏ đạt F1 **0.950**, còn model lớn đạt **0.927**. Tập SUB quá nhỏ để kết luận model nhỏ tốt hơn nói chung; kết quả chỉ cho thấy model lớn không thắng trong lần thử này.
3. Dự đoán few-shot sẽ tăng F1. Thực tế F1 vẫn là **0.950**, còn chi phí tăng từ khoảng **0.004** lên **0.015 USD/1.000 tin**. Thêm ví dụ làm prompt dài hơn nhưng chưa giúp phân loại tốt hơn trên SUB này.

## 4. Phân tích 50 case khó

Sau khi thêm 5 case, lần chạy cuối đánh giá **55 tin**. BoW + LogReg tụt nhiều nhất, từ F1 **0.947** trên SUB xuống **0.308** trên bộ case khó. Model chỉ đúng **9%** nhóm spam giọng tự nhiên và **11%** nhóm thao túng: các tin này thiếu từ khóa spam quen thuộc nên BoW dễ bỏ sót.

Case **#100** là spam viết như lời nhắn thân quen về ảnh; LLM nhỏ với prompt phòng thủ đoán nhầm thành ham. Case **#101** giả danh người quản lý để đổi thông tin thanh toán; **tất cả model** đều đoán nhầm thành ham. Hai case cho thấy một tin nhắn có vẻ bình thường vẫn có thể chứa yêu cầu đáng ngờ khi xét ý định và ngữ cảnh.

## 5. Thử thách cá nhân

**Đề #2:** Giảm từ điển xuống 100 từ và tăng lên 5.000 từ, huấn luyện lại BoW + LogReg và so F1 test. Tôi tạo từ điển từ tập train, vector hóa train/test theo từ điển mới, rồi dùng lại `train_logreg` với `lr=0.5` và 300 epoch.

Vocab **100 từ** cho F1 **0.871**; vocab **5.000 từ** cho F1 **0.893**, bằng vocab **2.000 từ** ở Phần 1. Từ điển 100 từ bỏ qua nhiều từ hữu ích để nhận diện spam. Thêm từ từ 2.000 lên 5.000 không cải thiện F1 trong lần chạy này, nên từ điển lớn hơn không tự động cho kết quả tốt hơn.

## 6. Phương án cho sản phẩm thật

Cả 5 cách không dùng LLM đều đạt p95 dưới 50 ms/tin. Các cách dùng LLM mất khoảng 2.4–3.2 giây nên không đạt yêu cầu. Tôi chọn BoW + MLP: F1 test đạt 0.932, p95 0.357 ms và tốc độ xử lý cả lô khoảng 49.905 tin/giây, cao hơn mức trung bình 116 tin/giây. Encoder + LogReg và Encoder + MLP head có F1 cao hơn (0.946), nhưng tốc độ đo được chỉ khoảng 49 và 48 tin/giây; nếu dùng cho toàn bộ lưu lượng thì cần thêm năng lực xử lý. Với 300 triệu tin/tháng, chi phí API ước tính theo bảng là khoảng 1.200 USD cho LLM nhỏ zero-shot, 2.400 USD cho LLM lớn và 4.500 USD cho LLM nhỏ few-shot.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4: API và prompt | Hỏi cách cấu hình Groq, gọi API, tạo prompt và đọc nhãn. | AI gợi ý code cho `call_openai`, `build_classification_prompt`, `parse_label` và cách đo độ trễ, chi phí. | Tôi chạy test giả lập, gọi API trên Colab; các kiểm tra Phần 4 đạt và kết quả có `invalid=0`. |
| Phần 4: phân tích kết quả | Hỏi về zero-shot, few-shot, giới hạn token và lỗi của `parse_label_v2`. | AI giải thích cách đọc kết quả và các trường hợp dễ phân loại sai. | Tôi đối chiếu output: LLM nhỏ F1 0.950, LLM lớn 0.927; few-shot không tăng F1. |
| Phần 5: so sánh model | Hỏi cách đọc Pareto, stress test và chọn phương án theo SLA. | AI gợi ý cách diễn giải F1, p95, chi phí và nhóm case khó. | Tôi đối chiếu lại bảng và sửa câu trả lời theo lần chạy cuối, gồm 55 case. |

Trước bài lab, tôi nghĩ model phức tạp hơn và prompt dài hơn thường cho kết quả tốt hơn. Kết quả MLP, LLM và stress test cho thấy cần kiểm tra trên cùng dữ liệu, xem cả chất lượng lẫn độ trễ. Tôi cũng hiểu rằng điểm cao trên test thường chưa bảo đảm model xử lý tốt các tin nhắn khác kiểu dữ liệu đã học.
