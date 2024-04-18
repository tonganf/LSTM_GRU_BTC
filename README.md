# Compare LSTM and GRU (BTC)

## LSTM 

<div align="center">
<img src="loss(1,50,3,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 3<br>

<img src="loss(1,50,5,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 5<br>
</div>
Sử dụng model kết hợp CrossEntropy và SGD(lr = 0.001) cho ra kết quả tốt. 

* Biểu Đồ Thứ Nhất (Hidden_dim = 3):

      // Mất mát huấn luyện và mất mát kiểm tra đều giảm đáng kể khi số lượng epochs tăng lên, cho thấy mô hình đang học từ dữ liệu.

      // Có một sự chênh lệch nhỏ giữa mất mát huấn luyện và mất mát kiểm tra, điều này cho thấy mô hình không bị quá khớp (overfitting) nhiều. Mô hình có thể tổng quát hóa tốt từ dữ liệu huấn luyện sang dữ liệu kiểm tra.

* Biểu Đồ Thứ Hai (Hidden_dim = 5):

      // Mất mát huấn luyện và mất mát kiểm tra có xu hướng giảm xuống và ổn định nhanh hơn so với mô hình với hidden_dim = 3.
      // Mất mát cho cả huấn luyện và kiểm tra ổn định ở mức thấp, cho thấy mô hình này có thể đang học tốt hơn và có khả năng tổng quát hóa tốt.
=> Không có sự chênh lệch lớn giữa mất mát huấn luyện và kiểm tra, điều này lại càng chỉ ra rằng mô hình không bị overfitting.
Nhận Xét Chung:

Tăng số lượng hidden_dim từ 3 lên 5 có vẻ đã cải thiện khả năng học của mô hình, dẫn đến việc mất mát giảm mạnh và ổn định hơn. Mất mát kiểm tra thấp và sự chênh lệch không lớn giữa mất mát huấn luyện và kiểm tra cho thấy mô hình có khả năng tổng quát hóa tốt 


## GRU
<div align="center">
  <img src="gru_loss(1,50,3,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 3<br>

  <img src="gru_loss(1,50,5,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 5<br>
</div>

Biểu Đồ Trên:

* Mất mát huấn luyện (training loss) và mất mát kiểm tra (testing loss) giảm mạnh ở những kỳ đầu và sau đó ổn định. Điều này chỉ ra rằng mô hình đã học được từ dữ liệu và khái quát hóa tốt.

* Mất mát kiểm tra cao hơn so với mất mát huấn luyện trong phần lớn quá trình huấn luyện nhưng sự chênh lệch không lớn, cho thấy mô hình có khả năng khái quát nhưng vẫn có thể cải thiện.

* Có một sự tăng vọt đột ngột ở mất mát kiểm tra ở cuối quá trình huấn luyện, có thể là do nhiễu trong dữ liệu kiểm tra hoặc mô hình có thể đang bắt đầu quá khớp với dữ liệu huấn luyện.

Biểu Đồ Dưới:

* Mất mát cho cả huấn luyện và kiểm tra cũng giảm đáng kể và ổn định, tương tự như biểu đồ trên.
Sự chênh lệch giữa mất mát huấn luyện và kiểm tra thấp hơn so với biểu đồ trên, cho thấy mô hình có khả năng tổng quát hóa tốt hơn.

* Không có sự tăng vọt đột ngột ở cuối như biểu đồ trên, điều này cho thấy kết quả kiểm tra ổn định hơn qua các kỳ.

Nhận Xét Chung:

* Cả hai mô hình đều học tốt từ dữ liệu và có khả năng khái quát, nhưng mô hình trong biểu đồ dưới có vẻ ít bị ảnh hưởng bởi nhiễu hoặc sự biến động trong dữ liệu kiểm tra hơn.

* Sự tăng vọt ở cuối biểu đồ trên cần được chú ý, nó có thể là dấu hiệu của sự không ổn định trong mô hình hoặc cần phải kiểm tra lại dữ liệu kiểm tra để đảm bảo rằng không có vấn đề về chất lượng dữ liệu.


# Kết quả

## LSTM
<div align="center">
  <img src="lstm(1,50,5,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 3<br>
</div>
<div align="center">
  <img src="lstm(1,50,5,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 5<br>
</div>

biểu đồ giá Bitcoin dự đoán có vẻ tạo ra dự đoán kỳ vọng một xu hướng giảm giá kéo dài trong khoảng thời gian từ tháng 4 đến tháng 7 năm 2024. Nếu biểu đồ này phản ánh một mô hình dự đoán thực sự thì:


* Sự Sụt Giảm Đáng Kể: Một sự kiện tiêu cực không xác định có thể dẫn đến đợt bán tháo lớn trong tháng 4, gây ra một sự sụt giảm giá nhanh chóng và mạnh mẽ.

* Áp Lực Bán Có Thể Tiếp Diễn: Áp lực bán có thể tiếp tục thống trị thị trường do tâm lý lo ngại về tương lai của Bitcoin, hoặc do những yếu tố nền tảng như thay đổi trong quy định hoặc sự cạnh tranh từ các loại tiền mã hóa khác.

* Tâm Lý Thận Trọng: Nhà đầu tư có thể thận trọng hơn và chần chừ trong việc đặt vốn vào Bitcoin, dẫn đến giá tiếp tục giảm dần mặc dù có những dấu hiệu nhất định của sự ổn định.


## GRU
<div align="center">
  <img src="lstm(1,50,5,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 3<br>
</div>
<div align="center">
  <img src="lstm(1,50,5,120).png" width="350" height="250"><br>Sử dụng hidden_dim = 5<br>
</div>

=> Tương tự như nhận định ở trên của lstm, giá BTC có thể sụp giảm và đi vào giai đoạn downtrend ngắn hạn quay trở về mốc hỗ trợ mới 35k

<div align="center">
<p>Biểu đồ dự đoán trong dài hạn đến tháng 7/2024</p>
</div>
<div align="center">
  <img src="all_gru(1,50,3,120).png" width="350" height="250"><br><br>
</div>
