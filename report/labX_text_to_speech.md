# Lab X: Text-to-Speech (TTS)

Giang Nguyen Thi - 22001254

2025-12-01

---

## Mục tiêu chung

Báo cáo này nhằm cung cấp một cái nhìn tổng quan, rõ ràng và có hệ thống về bài toán Text-to-Speech (TTS), trình bày ba hướng tiếp cận chính từ truyền thống đến deep learning và zero-shot, phân tích ưu – nhược điểm của từng level, làm rõ nhu cầu dữ liệu – tài nguyên – khả năng cá nhân hóa, đồng thời mô tả cách các pipeline hiện đại tối ưu chất lượng giọng nói và tốc độ suy luận. Bên cạnh đó, báo cáo cũng nêu các thách thức còn tồn tại và đề cập góc nhìn đạo đức, đặc biệt là vấn đề watermark nhằm nhận diện giọng nói sinh bởi AI và hạn chế rủi ro deepfake.

---

## **1. Giới thiệu chung**

### **1.1. Bài toán Text-to-Speech là gì?**

Text-to-Speech (TTS) là bài toán chuyển đổi văn bản đầu vào thành tín hiệu giọng nói với mục tiêu tạo ra âm thanh tự nhiên, rõ ràng và dễ hiểu. Một hệ thống TTS hiện đại thường gồm ba thành phần chính:

- Text processing: chuẩn hóa văn bản, tách câu, chuyển văn bản thành chuỗi âm vị (phoneme) hoặc các đặc trưng ngôn ngữ cần thiết.
- Acoustic model: dự đoán đặc trưng âm học (mel-spectrogram) từ chuỗi văn bản hoặc phoneme; mô hình học ngữ điệu, trường độ, nhấn giọng.
- Vocoder: chuyển đặc trưng âm học thành dạng sóng (waveform); các vocoder hiện đại như WaveNet, HiFi-GAN giúp âm thanh trở nên tự nhiên hơn.

TTS được ứng dụng trong nhiều kịch bản thực tế: trợ lý ảo, đọc sách nói, tổng đài chăm sóc khách hàng, hệ thống phát thanh, sản xuất nội dung đa phương tiện, thiết bị IoT, phương tiện tự hành và hỗ trợ người khiếm thị trong truy cập thông tin. Đây là một trong những bài toán quan trọng của lĩnh vực xử lý ngôn ngữ và âm thanh hiện đại.

### **1.2. Mục tiêu cụ thể của báo cáo**

Báo cáo hướng tới việc trình bày một bức tranh toàn cảnh, rõ ràng và có hệ thống về ba hướng tiếp cận chính trong Text-to-Speech (TTS):

- Level 1: hệ thống dựa trên luật, chạy nhanh nhưng độ tự nhiên thấp.
- Level 2: mô hình deep learning, cho chất lượng giọng tự nhiên hơn nhưng yêu cầu dữ liệu và tài nguyên lớn.
- Level 3: mô hình few-shot/zero-shot, chỉ cần vài giây âm thanh để học giọng nhưng phức tạp và tốn nhiều tài nguyên.

Các mục tiêu cụ thể gồm:
- Tổng hợp tổng quan bài toán TTS và tình hình nghiên cứu theo ba level.
- Phân tích ưu điểm, nhược điểm và các use case phù hợp cho từng hướng tiếp cận.
- Trình bày cách các pipeline TTS hiện nay được thiết kế để giảm nhược điểm (tốc độ, tài nguyên, dữ liệu, khả năng đa ngôn ngữ, biểu cảm).
- Bổ sung góc nhìn đạo đức trong nghiên cứu TTS, bao gồm nhu cầu nhúng watermark vào đầu ra của mô hình nhằm nhận diện giọng nói sinh ra bởi AI, hỗ trợ phát hiện và giảm thiểu rủi ro deepfake và thông tin sai lệch.


## **2. Bức tranh tổng quan về TTS và các hướng phát triển**

### **2.1. Lịch sử và các giai đoạn lớn**

Trong quá trình phát triển, Text-to-Speech (TTS) đã trải qua ba giai đoạn quan trọng:

- Giai đoạn truyền thống: các hệ thống dựa trên luật (rule-based) và kỹ thuật nối ghép âm thanh (concatenative synthesis). Chất lượng ở mức chấp nhận được nhưng giọng còn cứng, ít tự nhiên và khó mở rộng sang nhiều ngữ cảnh.
- Giai đoạn deep learning: các mô hình như Tacotron, Tacotron 2, Transformer TTS hay FastSpeech giúp TTS học trực tiếp từ dữ liệu giọng nói, cải thiện mạnh mẽ độ tự nhiên nhờ khả năng học prosody, nhấn giọng và ngữ điệu.
- Giai đoạn few-shot/zero-shot: các mô hình lớn (large-scale TTS) có thể bắt chước giọng nói mới chỉ từ vài giây âm thanh. Đây là bước tiến lớn, cho phép nhân bản giọng nói nhanh nhưng đi kèm với độ phức tạp và chi phí tính toán rất cao.

### **2.2. Ba level tiếp cận**

Chia TTS thành ba level khác nhau, tương ứng với ba hướng phát triển:

- Level 1: dựa trên luật và âm vị. Hệ thống chạy nhanh, tiết kiệm tài nguyên và hỗ trợ tốt đa ngôn ngữ, nhưng giọng thiếu tự nhiên.
- Level 2: sử dụng deep learning. Chất lượng giọng cải thiện rõ rệt, có thể cá nhân hóa bằng cách thu vài phút giọng để fine-tune, nhưng yêu cầu dữ liệu lớn và tài nguyên huấn luyện cao.
- Level 3: few-shot hoặc zero-shot. Chỉ cần vài giây audio để học giọng mới, tạo ra giọng nói gần giống người thật nhưng mô hình rất phức tạp và tốn tài nguyên lớn khi suy luận.

Điểm khác biệt chính giữa ba level nằm ở:
- Mức độ tự nhiên của giọng: Level 1 thấp; Level 2 cao; Level 3 rất cao.
- Lượng dữ liệu cần thiết: Level 1 không cần dữ liệu huấn luyện; Level 2 cần vài phút đến vài giờ ghi âm; Level 3 chỉ cần vài giây.
- Mức độ phức tạp và tài nguyên tính toán: Level 1 thấp; Level 2 trung bình đến cao; Level 3 cao nhất.

## **3. Level 1 – TTS dựa trên luật và các phương pháp truyền thống**

### **3.1. Nguyên lý hoạt động**

Các hệ thống TTS truyền thống (rule-based và concatenative) hoạt động dựa trên việc mô phỏng quy tắc phát âm và ghép nối các đơn vị âm thanh nhỏ đã được thu âm sẵn. Quy trình tổng quát của một hệ thống Level 1 thường bao gồm:

- Sử dụng các luật ngữ âm và các quy tắc phát âm trong từ điển để xác định cách đọc mỗi từ trong văn bản.
- Phân tách và ánh xạ văn bản sang chuỗi âm vị hoặc các đơn vị ngôn ngữ như phoneme, diphone hoặc syllable.
- Lựa chọn các đoạn âm thanh tương ứng đã được lưu trữ từ trước trong cơ sở dữ liệu âm thanh.
- Ghép nối các đoạn âm thanh này theo thứ tự, đồng thời chỉnh sửa mức độ nhỏ như cường độ hoặc độ dài để tạo sự liền mạch.

Pipeline điển hình có thể mô tả như sau:

Văn bản => chuẩn hóa văn bản => chuyển sang chuỗi âm vị => áp dụng luật phát âm => ghép nối âm thanh đã thu.

Do toàn bộ quá trình đều dựa trên quy tắc và dữ liệu âm thanh cố định, hệ thống tạo ra đầu ra mang tính ổn định và dễ dự đoán.

### **3.2. Ưu điểm**

Các hệ thống Level 1 có một số ưu điểm nổi bật:

- Tốc độ xử lý rất nhanh và độ trễ thấp, phù hợp với các yêu cầu thời gian thực hoặc thiết bị cấu hình yếu.
- Tiêu tốn rất ít tài nguyên tính toán do không cần mô hình học sâu; có thể chạy tốt trên CPU và các hệ nhúng.
- Mô hình hoạt động theo cách xác định, dễ kiểm soát và dễ gỡ lỗi, bởi mọi bước xử lý đều dựa trên quy tắc rõ ràng.
- Khi đã xây dựng đủ bộ quy tắc ngữ âm và từ điển, việc mở rộng sang nhiều ngôn ngữ khác tương đối thuận lợi vì không cần thu thập dữ liệu âm thanh lớn.

### **3.3. Nhược điểm**

Hạn chế của Level 1 khá rõ ràng và là lý do khiến các mô hình này dần bị thay thế trong nhiều ứng dụng:

- Độ tự nhiên của giọng rất thấp, âm thanh thường mang cảm giác khô cứng, thiếu nhịp điệu và nghe giống giọng robot.
- Việc thể hiện cảm xúc gần như không khả thi do âm thanh được ghép từ các đơn vị rời rạc, không có khả năng mô phỏng prosody phức tạp.
- Rất khó mở rộng sang miền nội dung mới hoặc ngôn ngữ mới, bởi cần phải xây dựng thủ công bộ luật phát âm và xử lý tất cả trường hợp đặc biệt.
- Khó xử lý các tình huống ngữ cảnh phức tạp như từ đa nghĩa, cách đọc biến đổi theo ngữ cảnh hay câu dài với cấu trúc ngữ pháp đặc biệt.

### **3.4. Trường hợp sử dụng phù hợp**

Mặc dù có nhiều hạn chế, Level 1 vẫn có vai trò trong các ứng dụng yêu cầu tốc độ cao và tài nguyên thấp:

- Các thiết bị nhúng hoặc IoT như thiết bị thông minh đơn giản, hệ thống báo hiệu, đồ chơi điện tử hoặc máy công nghiệp.
- Các ứng dụng không yêu cầu giọng nói giống người thật, ví dụ loa thông báo tại nhà ga, xe buýt, máy bán hàng tự động hoặc hệ thống cảnh báo.
- Các tình huống ưu tiên tính ổn định và độ trễ thấp hơn chất lượng giọng, như hệ thống vận hành liên tục hoặc môi trường không cho phép chạy mô hình lớn.

## **4. Level 2 – Deep Learning TTS (mô hình học sâu)**

### **4.1. Mô tả chung và kiến trúc phổ biến**

Level 2 đánh dấu sự chuyển đổi quan trọng từ các hệ thống TTS dựa trên luật sang các mô hình học sâu có khả năng học trực tiếp mối quan hệ giữa văn bản và âm thanh từ dữ liệu thực tế. Ý tưởng cốt lõi của hướng tiếp cận này là sử dụng mạng nơ-ron để học ánh xạ từ văn bản (hoặc chuỗi phoneme) sang đặc trưng âm học, sau đó dùng vocoder để tái tạo dạng sóng từ các đặc trưng đó.

Một hệ thống TTS deep learning thường bao gồm ba thành phần chính:

- Text encoder: mã hóa chuỗi văn bản hoặc phoneme thành biểu diễn liên tục. Encoder có thể bao gồm các lớp CNN, LSTM, Transformer, tùy thuộc kiến trúc. Bộ mã hóa giúp mô hình hiểu cấu trúc ngôn ngữ và mối quan hệ giữa các đơn vị âm vị.
- Acoustic model: dự đoán đặc trưng âm học, phổ biến nhất là mel-spectrogram. Các mô hình như Tacotron, Tacotron 2, Transformer TTS, FastSpeech hoặc VITS học ngữ điệu, độ dài âm tiết, độ nhấn và các yếu tố prosody khiến giọng nói trở nên tự nhiên hơn.
- Neural vocoder: chuyển mel-spectrogram thành dạng sóng. Các vocoder thuộc họ WaveNet, WaveRNN, MelGAN, HiFi-GAN thường được sử dụng để tái tạo âm thanh sắc nét, giảm nhiễu và tăng độ tự nhiên.

Sự phối hợp giữa ba thành phần trên giúp hệ thống tạo ra giọng nói có sự liền mạch, biểu cảm và dễ nghe hơn so với phương pháp truyền thống. Đây là lý do Level 2 trở thành nền tảng của hầu hết hệ thống TTS hiện đại.

### **4.2. Fine-tuning cho từng người dùng**

Một ưu điểm quan trọng của Level 2 là khả năng cá nhân hóa giọng nói thông qua quá trình fine-tuning. Liên hệ với yêu cầu đề bài, mỗi người dùng có thể ghi âm một tập dữ liệu nhỏ và tinh chỉnh mô hình theo giọng của riêng mình.

Quy trình chung như sau:

- Thu một lượng dữ liệu giọng nói đủ lớn để mô hình học đặc trưng về tần số, ngữ điệu và độ dài âm tiết của người nói. Lượng dữ liệu thường từ vài phút đến vài chục phút.
- Trích xuất embedding đại diện cho người nói để mô hình nắm được đặc trưng giọng riêng biệt.
- Tiến hành fine-tune mô hình TTS với các mẫu âm thanh của người dùng nhằm điều chỉnh tham số mô hình theo phong cách nói riêng.
- Sau fine-tuning, mô hình có thể tổng hợp giọng nói mới mang phong cách của người đó, đạt chất lượng cao hơn nhiều so với Level 1 nhưng vẫn tiết kiệm tài nguyên hơn Level 3.

Cách tiếp cận này rất hữu ích khi người dùng muốn sở hữu giọng nói của riêng mình mà không cần huấn luyện lại toàn bộ mô hình.

### **4.3. Ưu điểm**

So với Level 1, Level 2 mang lại bước nhảy vọt về chất lượng nhờ khả năng học trực tiếp từ dữ liệu âm thanh. Các ưu điểm nổi bật gồm:

- Giọng nói tự nhiên hơn nhiều vì acoustic model học được ngữ điệu, độ dài âm tiết, nhấn giọng và các yếu tố prosody phức tạp mà hệ thống dựa trên luật không thể mô phỏng.
- Mô hình có thể tạo ra âm thanh với nhịp điệu linh hoạt, tránh sự khô cứng như giọng robot.
- Có thể điều khiển cảm xúc của giọng nói thông qua embedding cảm xúc hoặc tham số đầu vào.
- Hỗ trợ nhiều người nói và nhiều ngôn ngữ nếu được huấn luyện đúng cách.
- Khả năng cá nhân hóa cao: việc fine-tune cho từng người dùng trở nên dễ dàng và mang lại chất lượng tốt.

### **4.4. Nhược điểm**

Dù mang lại chất lượng âm thanh tốt, Level 2 vẫn tồn tại một số hạn chế:

- Cần một lượng dữ liệu ghi âm đáng kể để huấn luyện hoặc fine-tune. Đối với các ngôn ngữ ít tài nguyên, việc thu thập dữ liệu trở thành rào cản lớn.
- Tài nguyên tính toán ở mức trung bình đến cao, đặc biệt ở giai đoạn training. Việc huấn luyện mô hình từ đầu đòi hỏi GPU mạnh và thời gian dài.
- Việc triển khai trên thiết bị biên khó khăn hơn do trọng lượng mô hình lớn. Thường phải áp dụng các kỹ thuật giảm kích thước như quantization hoặc distillation.
- Vẫn chưa thể đạt tính linh hoạt cực cao như Level 3 (chỉ cần vài giây audio).

### **4.5. Trường hợp sử dụng phù hợp**

Level 2 thường được sử dụng trong các ứng dụng yêu cầu giọng nói tự nhiên, mượt mà nhưng vẫn ở mức tài nguyên cho phép:

- Ứng dụng thương mại như đọc sách nói, podcast, thuyết minh video, nơi chất lượng giọng được ưu tiên.
- Chatbot hoặc tổng đài chăm sóc khách hàng cần giao tiếp tự nhiên với người dùng.
- Bài toán cá nhân hóa giọng nói cho từng người dùng như streamer, người làm nội dung hoặc dịch vụ tạo giọng riêng.
- Các hệ thống có đủ tài nguyên để chạy mô hình deep learning nhưng không yêu cầu mức linh hoạt cực độ như zero-shot.

## **5. Level 3 – Few-shot/Zero-shot TTS (mô hình lớn)**

### **5.1. Ý tưởng và cơ chế chung**

Level 3 đại diện cho thế hệ TTS tiên tiến nhất hiện nay, nơi mô hình có khả năng học đặc trưng giọng nói mới từ một lượng dữ liệu cực kỳ ít, thường chỉ vài giây âm thanh. Điều này khả thi nhờ các mô hình TTS quy mô lớn được huấn luyện trên tập dữ liệu rất đa dạng, bao gồm hàng nghìn người nói, nhiều ngôn ngữ và nhiều phong cách giọng khác nhau.

Cơ chế của hệ thống thường gồm các bước sau:

- Mô hình sử dụng một mạng trích xuất embedding giọng nói để phân tích đoạn âm thanh ngắn của người dùng. Embedding này chứa các đặc trưng như tần số cơ bản, chất giọng, nhịp điệu và các sắc thái riêng của người nói.
- Khi người dùng cung cấp văn bản và đoạn audio mẫu, mô hình kết hợp text embedding và voice embedding để tạo ra mel-spectrogram theo phong cách giọng của người đó.
- Vocoder sẽ chuyển đặc trưng này thành dạng sóng, tạo ra âm thanh mang giọng nói mới mà không cần fine-tune mô hình.

Do không phải tinh chỉnh lại toàn bộ mô hình cho từng người, Level 3 cho phép cá nhân hóa giọng nói nhanh chóng và linh hoạt.

### **5.2. Ưu điểm**

Level 3 mang lại nhiều ưu thế vượt trội so với Level 1 và Level 2:

- Người dùng chỉ cần ghi âm một đoạn ngắn vài giây đến vài chục giây, thay vì vài phút hay hàng giờ như Level 2. Điều này giúp giảm đáng kể công sức và chi phí thu thập dữ liệu.
- Mức độ tự nhiên của giọng nói rất cao nhờ mô hình đã học đa dạng các phong cách nói trong quá trình huấn luyện. Trong nhiều trường hợp, đầu ra có thể khó phân biệt với giọng thật.
- Mô hình có khả năng nhân bản giọng nói gần như ngay lập tức, phù hợp cho các ứng dụng cần nhiều giọng nói khác nhau hoặc nhiều nhân vật.
- Do mô hình đã được huấn luyện trên nhiều ngôn ngữ, khả năng transfer sang ngôn ngữ mới (kể cả không có trong đoạn mẫu) được hỗ trợ tốt hơn các level trước.

### **5.3. Nhược điểm**

Mặc dù rất mạnh mẽ, Level 3 vẫn tồn tại các hạn chế đáng kể:

- Kích thước mô hình rất lớn, thường thuộc nhóm mô hình đa phương thức hoặc mô hình giọng nói quy mô hàng trăm triệu đến hàng tỷ tham số. Điều này khiến chi phí huấn luyện và suy luận tăng cao.
- Tài nguyên tính toán yêu cầu lớn, đặc biệt khi sinh âm thanh dài hoặc chạy trên thiết bị giới hạn tài nguyên. Việc triển khai thời gian thực là một thách thức.
- Rủi ro đạo đức là một trong những vấn đề quan trọng nhất: khả năng nhân bản giọng nói có thể bị lạm dụng để tạo deepfake, lừa đảo hoặc phát tán thông tin sai lệch nếu không có kiểm soát.
- Đối với các ngôn ngữ ít tài nguyên, chất lượng có thể chưa ổn định do mô hình không được huấn luyện đủ dữ liệu cho ngôn ngữ đó.

### **5.4. Trường hợp sử dụng phù hợp**

Do khả năng linh hoạt vượt trội, Level 3 phù hợp cho nhiều ứng dụng hiện đại:

- Sản xuất nội dung đa phương tiện, bao gồm phim, game, hoạt hình, nơi cần nhiều nhân vật với nhiều giọng khác nhau.
- Dịch vụ tạo voice-over nhanh cho video quảng cáo, thuyết minh hoặc bản demo, nơi thời gian triển khai ngắn là yếu tố quan trọng.
- Hệ thống hỗ trợ người sáng tạo nội dung muốn nhân bản giọng nói của mình mà không cần thu nhiều dữ liệu.
- Các nghiên cứu về mô hình đa phương thức hoặc đa ngôn ngữ, nơi cần mô hình linh hoạt và dễ mở rộng sang nhiều kiểu giọng hoặc nhiều ngôn ngữ.

## **6. So sánh tổng hợp ba hướng tiếp cận**

### **6.1. Bảng so sánh theo tiêu chí**

Bảng dưới đây tổng hợp sự khác biệt giữa ba level tiếp cận trong TTS dựa trên các tiêu chí quan trọng như độ tự nhiên, nhu cầu dữ liệu, mức độ phức tạp và khả năng cá nhân hóa.

| Tiêu chí                    | Level 1 – Rule-based | Level 2 – Deep Learning | Level 3 – Few-shot/Zero-shot |
|-----------------------------|-----------------------|--------------------------|-------------------------------|
| Độ tự nhiên của giọng      | Thấp                  | Cao                      | Rất cao                       |
| Nhu cầu dữ liệu            | Không cần dữ liệu huấn luyện | Cần vài phút đến vài giờ âm thanh | Chỉ cần vài giây đến vài chục giây |
| Tài nguyên tính toán       | Thấp                  | Trung bình đến cao       | Rất cao                        |
| Khả năng đa ngôn ngữ       | Phụ thuộc bộ luật     | Tốt nếu có dữ liệu       | Tốt, thường hỗ trợ nhiều ngôn ngữ |
| Mức độ cá nhân hóa         | Gần như không có      | Cao nhờ fine-tuning      | Rất cao, không cần fine-tune   |
| Khả năng thêm cảm xúc      | Rất hạn chế           | Tốt nếu có embedding cảm xúc | Tốt, mô hình học từ dữ liệu đa dạng |
| Độ phức tạp triển khai     | Thấp                  | Trung bình               | Cao                             |

Bảng cho thấy từng level có ưu điểm riêng, phù hợp với nhu cầu khác nhau về chất lượng và tài nguyên.

### **6.2. Phù hợp với từng loại bài toán cụ thể**

- Hệ thống ưu tiên tốc độ và tài nguyên nhỏ nên chọn Level 1. Đây là lựa chọn phù hợp cho các thiết bị nhúng, hệ thống thông báo công cộng hoặc ứng dụng đơn giản không đòi hỏi giọng tự nhiên.
- Hệ thống cần chất lượng giọng tự nhiên nhưng vẫn chấp nhận huấn luyện hoặc fine-tune nên chọn Level 2. Mô hình này phù hợp cho chatbot, đọc sách nói hoặc các ứng dụng thương mại cần tương tác bằng ngôn ngữ tự nhiên.
- Hệ thống cần mức linh hoạt cao, nhiều nhân vật, nhiều giọng nói khác nhau, hoặc yêu cầu cá nhân hóa nhanh với rất ít dữ liệu nên chọn Level 3. Điều này đặc biệt hữu ích trong sản xuất nội dung đa phương tiện, thuyết minh video nhanh hoặc dịch vụ tạo giọng cá nhân.

## **7. Các thách thức chung và mục tiêu nghiên cứu hiện tại**

Trong quá trình phát triển các hệ thống Text-to-Speech, có nhiều thách thức quan trọng mà cộng đồng nghiên cứu đang nỗ lực giải quyết. Những thách thức này liên quan đến tốc độ, tài nguyên, tính tự nhiên, khả năng đa ngôn ngữ, mức độ tiện dụng cho người dùng và các vấn đề đạo đức phát sinh từ việc nhân bản giọng nói bằng AI.

### **7.1. Hiệu suất nhanh và độ trễ thấp**

Một mục tiêu quan trọng của TTS hiện đại là rút ngắn thời gian suy luận để đạt tốc độ gần với Level 1, nhưng vẫn giữ được chất lượng giọng nói của Level 2 hoặc Level 3. Điều này đòi hỏi tối ưu hóa mô hình, giảm số bước suy luận và cải thiện pipeline xử lý để hệ thống có thể hoạt động trong thời gian thực.

### **7.2. Giảm tài nguyên tính toán**

Đối với các mô hình deep learning và zero-shot quy mô lớn, tài nguyên tính toán là rào cản lớn. Nghiên cứu hiện nay tập trung vào:
- Thiết kế mô hình nhỏ gọn nhưng hiệu quả.
- Áp dụng các kỹ thuật nén như distillation hoặc quantization để giảm kích thước mô hình.
- Tối ưu mô hình để chạy tốt trên GPU, CPU đa lõi hoặc các thiết bị di động, nhằm mở rộng khả năng ứng dụng.

### **7.3. Đảm bảo tính tự nhiên và cảm xúc**

Giọng nói tự nhiên không chỉ phụ thuộc âm sắc mà còn ở prosody. Các hướng nghiên cứu chính bao gồm:
- Học các đặc trưng prosody tốt hơn, như nhịp điệu, trường độ và nhấn âm.
- Mở rộng mô hình để điều khiển cảm xúc và biểu cảm, giúp giọng nói giàu sắc thái hơn.
- Sử dụng biểu diễn latent để điều chỉnh cảm xúc linh hoạt mà không cần dữ liệu gán nhãn phức tạp.

### **7.4. Đa ngôn ngữ và ngôn ngữ ít tài nguyên**

TTS cho ngôn ngữ ít dữ liệu là thách thức lớn. Một số hướng phát triển gồm:
- Áp dụng transfer learning để chuyển kiến thức từ ngôn ngữ giàu dữ liệu sang ngôn ngữ khác.
- Khai thác dữ liệu song ngữ hoặc đa ngôn ngữ để mô hình học được cấu trúc ngôn ngữ chung.
- Thiết kế mô hình đa ngôn ngữ, cho phép xử lý nhiều ngôn ngữ trong cùng một kiến trúc.

### **7.5. Giảm công sức người dùng**

Để người dùng dễ tiếp cận hơn, mục tiêu quan trọng là:
- Giảm lượng dữ liệu cần thiết cho quá trình cá nhân hóa giọng nói.
- Tối ưu hóa quy trình ghi âm để trở nên thuận tiện và nhanh chóng.
- Tạo giao diện đơn giản, tự động hóa nhiều bước xử lý để người dùng không cần chuyên môn kỹ thuật.

### **7.6. Khía cạnh đạo đức và watermark**

Sự phát triển của công nghệ voice cloning đặt ra nhiều rủi ro liên quan đến deepfake và sai lệch thông tin. Do đó, một hướng nghiên cứu quan trọng là:
- Nhúng watermark vào audio sinh bởi mô hình AI để nhận diện, theo đúng đề bài gợi ý.
- Phát triển kỹ thuật phát hiện giọng nói giả mạo nhằm hạn chế lạm dụng.
- Xây dựng tiêu chuẩn và quy tắc đạo đức cho việc sử dụng công nghệ TTS trong thực tế.

## **8. Cách các nghiên cứu xây dựng pipeline để tối ưu ưu/nhược điểm**

Các hướng tiếp cận TTS ở Level 1, Level 2 và Level 3 đều có ưu và nhược điểm riêng. Thay vì chỉ chọn một level duy nhất, nhiều nghiên cứu tập trung thiết kế pipeline để tận dụng điểm mạnh và giảm bớt hạn chế của từng hướng. Phần này trình bày một số cách tối ưu điển hình.

### **8.1. Tối ưu Level 1**

Đối với Level 1, mục tiêu chính là cải thiện chất lượng giọng nói và ngữ điệu mà không làm tăng quá nhiều chi phí tính toán. Một số hướng tối ưu phổ biến là:

- Kết hợp luật với các thành phần học máy đơn giản, ví dụ dùng mô hình thống kê hoặc mạng nơ-ron nhỏ để dự đoán ngữ điệu hoặc độ dài âm tiết, thay vì chỉ dựa hoàn toàn vào quy tắc cố định.
- Thêm các module thống kê để lựa chọn đoạn ghép mượt mà hơn, chẳng hạn như chọn diphone hoặc syllable dựa trên xác suất xuất hiện trong ngữ cảnh tương tự, giảm hiện tượng “giọng giật cục”.
- Cải thiện bước chuẩn hóa văn bản và xử lý ngôn ngữ để hệ thống ít gặp lỗi trong các trường hợp đặc biệt như số, chữ viết tắt, ký hiệu, từ mượn.

Nhờ các bước này, hệ thống Level 1 vẫn giữ được tốc độ nhanh và chi phí thấp nhưng chất lượng giọng nói được cải thiện so với các hệ thống rule-based thuần túy.

### **8.2. Tối ưu Level 2**

Với Level 2, bài toán không chỉ là chất lượng mà còn là tốc độ và tài nguyên. Một số kỹ thuật thường được áp dụng:

- Sử dụng kiến trúc non-autoregressive như FastSpeech để tăng tốc suy luận. Thay vì sinh từng khung thời gian một cách tuần tự, mô hình có thể dự đoán toàn bộ chuỗi đặc trưng âm học song song.
- Kết hợp với vocoder nhanh, chất lượng cao như HiFi-GAN để rút ngắn thời gian chuyển mel-spectrogram sang dạng sóng nhưng vẫn giữ chất lượng chấp nhận được cho ứng dụng thực tế.
- Thiết kế chiến lược fine-tune hiệu quả, chỉ điều chỉnh một phần tham số (ví dụ adapter layer hoặc speaker embedding) để cá nhân hóa giọng nói với ít dữ liệu và ít chi phí hơn so với việc tinh chỉnh toàn bộ mô hình.
- Áp dụng các kỹ thuật nén mô hình như distillation hoặc quantization để triển khai trên các thiết bị có tài nguyên hạn chế mà không giảm chất lượng quá nhiều.

Những tối ưu này giúp Level 2 trở nên thực dụng hơn trong môi trường sản xuất, cân bằng giữa chất lượng, tốc độ và chi phí.

### **8.3. Tối ưu Level 3**

Đối với Level 3, thách thức nằm ở tính ổn định, chi phí và rủi ro đạo đức. Các pipeline nghiên cứu thường tập trung vào:

- Thiết kế kiến trúc phân tách rõ giữa embedding giọng nói và embedding nội dung. Điều này giúp mô hình giữ được thông tin nội dung chính xác đồng thời thay đổi giọng nói một cách linh hoạt.
- Cải thiện cơ chế few-shot bằng cách sử dụng các mạng encoder chuyên biệt cho giọng nói, giúp mô hình trích xuất thông tin giọng ổn định chỉ từ một đoạn audio rất ngắn.
- Ổn định hóa quá trình suy luận để tránh tình trạng giọng bị méo, mất ngữ điệu hoặc không giống người nói gốc khi dữ liệu mẫu quá ít.
- Tích hợp cơ chế kiểm soát và watermark: nhúng tín hiệu nhận diện vào audio sinh ra để hệ thống phân biệt được giọng nói nhân tạo, giảm rủi ro lạm dụng deepfake.

Nhờ đó, Level 3 không chỉ mạnh về khả năng nhân bản giọng mà còn bắt đầu đáp ứng tốt hơn các yêu cầu an toàn và đạo đức.

### **8.4. Kết hợp nhiều hướng trong một pipeline**

Thay vì chọn riêng lẻ một level, nhiều hệ thống hiện đại kết hợp các ý tưởng của cả ba level trong cùng một pipeline. Một ví dụ điển hình:

- Sử dụng mô hình Level 2 làm nền tảng ổn định: một mô hình TTS deep learning đã được huấn luyện tốt, cho chất lượng giọng tự nhiên.
- Bổ sung module voice embedding giống Level 3 để cho phép mô hình học giọng mới từ ít dữ liệu, đưa hệ thống tiến gần tới khả năng few-shot hoặc zero-shot.
- Áp dụng các kỹ thuật tối ưu của Level 1 và Level 2 (non-autoregressive, vocoder nhanh, nén mô hình) để rút ngắn thời gian suy luận, hướng tới tốc độ gần với Level 1.

Cách kết hợp này cho phép hệ thống vừa đạt chất lượng giọng cao, vừa linh hoạt trong cá nhân hóa, lại vừa đủ nhanh và tiết kiệm tài nguyên để có thể triển khai trong thực tế.

## **9. Kết luận**

Text-to-Speech (TTS) là một trong những bài toán quan trọng nhất của lĩnh vực xử lý ngôn ngữ và âm thanh, với ứng dụng rộng rãi trong trợ lý ảo, đọc sách nói, tổng đài tự động, thiết bị IoT và sản xuất nội dung đa phương tiện. Ba hướng tiếp cận chính gồm Level 1, Level 2 và Level 3 phản ánh sự phát triển tuần tự của công nghệ, từ hệ thống dựa trên luật đơn giản đến các mô hình deep learning và cuối cùng là mô hình few-shot hoặc zero-shot có khả năng cá nhân hóa giọng nói chỉ từ vài giây âm thanh.

Mỗi level mang đặc điểm riêng và phù hợp với những nhu cầu khác nhau: Level 1 ưu tiên tốc độ và tài nguyên thấp, Level 2 cân bằng giữa chất lượng và chi phí huấn luyện, trong khi Level 3 mang lại sự linh hoạt và độ tự nhiên cao nhất nhưng đòi hỏi tài nguyên lớn. Xu hướng hiện nay là kết hợp điểm mạnh của cả ba level trong cùng một pipeline nhằm tối ưu chất lượng giọng nói, giảm thời gian suy luận và mở rộng khả năng hỗ trợ đa ngôn ngữ, cảm xúc cũng như cá nhân hóa.

Ngoài ra, các thách thức liên quan đến đạo đức, đặc biệt là nguy cơ lạm dụng voice cloning để tạo deepfake, khiến vấn đề kiểm soát và nhúng watermark trở thành một hướng nghiên cứu quan trọng. Trong tương lai, có thể tập trung vào các hướng như giảm dữ liệu cần thiết cho cá nhân hóa, cải thiện mô hình đa ngôn ngữ, tăng hiệu quả trên thiết bị biên và phát triển cơ chế bảo mật nhận diện giọng nhân tạo một cách đáng tin cậy.

## **10. Tài liệu tham khảo**

1. Xia, W., Zhang, Y., Zheng, W., Wang, C., & Xiao, Y. (2024). Mega-TTS 2: Boosting Zero-Shot Text-to-Speech with Cross-Lingual Semi-Supervised Learning.  
   https://arxiv.org/abs/2401.13891

2. Wang, Y. et al. (2017). Tacotron: Towards End-to-End Speech Synthesis. *Interspeech*.  
   https://arxiv.org/abs/1703.10135

3. Shen, J. et al. (2018). Natural TTS Synthesis by Combining WaveNet and Sequence-to-Sequence Models (Tacotron 2).  
   https://arxiv.org/abs/1712.05884

4. Ren, Y. et al. (2019). FastSpeech: Fast, Robust and Controllable Text to Speech. *NeurIPS*.  
   https://arxiv.org/abs/1905.09263

5. Ren, Y. et al. (2020). FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. *ICLR*.  
   https://arxiv.org/abs/2006.04558

6. Kim, J. et al. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End Speech Synthesis (VITS).  
   https://arxiv.org/abs/2106.06103

7. Jia, Y. et al. (2018). Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis. *NeurIPS*.  
   https://arxiv.org/abs/1806.04558

8. Cooper, E. et al. (2020). Zero-Shot Multi-Speaker Text-To-Speech with Speaker Encoder and Instance Normalization.  
   https://arxiv.org/abs/2010.04301

9. Wang, C. et al. (2023). NaturalSpeech 2: Latent Diffusion Models Are Natural and Zero-Shot Speech and Singing Synthesizers.  
   https://arxiv.org/abs/2304.09116

10. Wang, C. et al. (2024). Voicebox: Text-Guided Multilingual Universal Speech Generation.  
    https://arxiv.org/abs/2306.10486

11. Fernández, F. et al. (2023). AudioSeal: Audio Watermarking for Model Transparency.  
    https://arxiv.org/abs/2310.16789

12. Kharitonov, E. et al. (2023). SpeakEasy: Zero-Shot Speech Editing with Disentangled Representations.  
    https://arxiv.org/abs/2311.08457

13. Kotha, S. et al. (2024). Universal Speech Watermarking via Robust Auditory Masking.  
    https://arxiv.org/abs/2403.03135

