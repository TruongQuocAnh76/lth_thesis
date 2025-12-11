# Đề tài

1. **Adaptive Hybrid Pruning with Dynamic Switching** (Cải Tiến Pruning Hybrid Thích Ứng Để Tìm Vé Số May Mắn Trong Mạng Nơ-ron Sâu): cải tiến phương pháp hybrid oneshot - iterative pruning của Janusz et al. (2025) bằng cách tìm phương pháp giúp threshold chuyển từ oneshot sang iterative adaptive thay vì sử dụng 1 constant.
2. **Comparative Study of Pruning Methods Related to the Lottery Ticket Hypothesis** (So sánh các Phương pháp Pruning liên quan đến Giả thuyết Vé Số): thử nghiệm và so sánh <= 8 thuật toán pruning. (nếu cải tiến không khả quan)

# Kế hoạch

## Tháng 1: Tháng 12/2025

1. **Tuần 1**: Đọc các tài liệu cốt lõi về pruning và lottery ticket hypothesis.
2. **Tuần 2**: Đọc tài liệu về các thuật toán pruning. Bắt đầu soạn thảo dàn ý Chương 2 (về Nền Tảng & Tổng Quan Tài Liệu). Xác định <= 8 baseline để triển khai (IMP, SNIP, GraSP, SynFlow, Early-Bird, Fisher Information pruning, Genetic algorithm, Hybrid).
3. **Tuần 3**: Viết bản nháp ban đầu của các phần Chương 2. Thiết lập môi trường phát triển (PyTorch, Git repo, wandb để theo dõi).
4. **Tuần 4**: Tái tạo 1 baseline (IMP) trên thiết lập nhỏ (CIFAR-10, ResNet-20).

## Tháng 2: Tháng 1/2026

5. **Tuần 5**: Hoàn tất bản nháp Chương 2. Tái tạo 2 (SNIP, GraSP) baseline. Xác thực so với kết quả tài liệu gốc.
6. **Tuần 6**: Tái tạo baseline còn lại. Bắt đầu pseudocode cho phương pháp cải tiến đề xuất.
7. **Tuần 7**: Gỡ lỗi baseline và chạy thử nghiệm ban đầu trên CIFAR-10/ResNet-20.
8. **Tuần 8**: Hoàn thiện Chương 2.

## Tháng 3: Tháng 2/2026

9. **Tuần 9**: Thiết kế và triển khai phương pháp cải tiến.
10. **Tuần 10**: Gỡ lỗi và xác thực trên quy mô nhỏ. So sánh kết quả ban đầu với baseline.
11. **Tuần 11**: Tinh chỉnh phương pháp dựa trên kết quả sớm. Bắt đầu soạn thảo Chương 3 (Phương Pháp: Thiết Kế Nghiên Cứu, Baseline, Phương Pháp Đề Xuất).
12. **Tuần 12**: Mở rộng đến các thiết lập khác (CIFAR-100, Resnet50/VGG-16). Chạy thử nghiệm sơ bộ.

## Tháng 4: Tháng 3/2026

13. **Tuần 13**: Mở rộng thử nghiệm đến thiết lập đầy đủ (2 bộ dữ liệu: CIFAR-10/100; 3 kiến trúc: ResNet-20/50, VGG-16). Chạy baseline với 3-5 seeds.
14. **Tuần 14**: Chạy phương pháp cải tiến trên tất cả cấu hình.
15. **Tuần 15**: Bắt đầu phân tích bổ sung.
16. **Tuần 16**: Thu thập kết quả ban đầu. Soạn thảo Chương 4 (Thiết Lập Thử Nghiệm: Bộ Dữ Liệu, Kiến Trúc, Chỉ Số).

## Tháng 5: Tháng 4/2026

17. **Tuần 17**: Hoàn thành bất kỳ thử nghiệm còn lại.
18. **Tuần 18**: Phân tích kết quả (so sánh baseline, ablation, hình ảnh hóa định tính). Soạn thảo Chương 5 (Kết Quả & Phân Tích: Bảng, biểu đồ, diễn giải).

   *Trước 15/4: Đăng ký tên đề tài*

19. **Tuần 19**: Hoàn thiện Chương 5. Bắt đầu bản nháp Chương 6 (Thảo Luận: Diễn Giải, So Sánh, Ý Nghĩa Thực Tế).
20. **Tuần 20**: Hoàn thành Chương 6. Bắt đầu bản nháp Chương 7 (Kết Luận & Công Việc Tương Lai).

## Tháng 6: Tháng 5/2026

21. **Tuần 21**: Viết Chương 1 (Giới Thiệu), Tóm Tắt, Lời Cảm Ơn. Hoàn thiện Chương 7.
22. **Tuần 22**: Tổng hợp toàn bộ khóa luận (tích hợp tất cả chương, tài liệu tham khảo, phụ lục). Thêm hình ảnh, đảm bảo 80-100 trang. Mở nguồn code trên GitHub.
23. **Tuần 23**: Kiểm tra cuối cùng (ngữ pháp, trích dẫn—định dạng IEEE/APA).
24. **Tuần 24**: Sửa chữa cuối cùng.

*Nộp trước 20/5/2026*