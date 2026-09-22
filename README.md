# Crypto Multi-Agent Trading System

Hệ thống giao dịch tiền mã hóa đa tác tử (Multi-Agent Trading) ứng dụng kiến trúc **LangGraph**, lấy cảm hứng từ các nghiên cứu và framework hàng đầu:
- **TauricResearch/TradingAgents** (mô hình quỹ định lượng đa tác tử)
- **Chivu171/Multi-Agent-Crypto** (đo lường bất đồng ý kiến bằng Variance & kích hoạt vòng tranh biện Bull vs Bear)
- **Non-LLM Hard Risk Guardrails** (Bộ quy tắc an toàn cứng độc lập ngoài LLM)
- **Real-Time Monitoring Dashboard** (Web Terminal Dark-mode sang trọng với WebSocket streaming)

---

## 🌟 Tính Năng Nổi Bật

1. **Đội Chuyên Gia Phân Tích Song Song (Parallel Analysts):**
   - **Technical Analyst:** Đo lường động lượng và xu hướng (RSI 14, MACD, Bollinger Bands, ATR, EMA 20/50/200).
   - **Sentiment & News Analyst:** Tích hợp dữ liệu Crypto Fear & Greed Index và dòng tin tức thị trường.
   - **On-chain & Market Flow Analyst:** Giám sát funding rate, open interest và độ lệch sổ lệnh (orderbook depth/imbalance).

2. **Cơ Chế Thẩm Định Bất Đồng (Chivu171 Consensus Validator):**
   - Tự động tính toán phương sai (variance) và khoảng cách niềm tin (belief distance) giữa các chuyên gia.
   - Nếu ý kiến phân hóa cao vượt ngưỡng $\ge 0.35$ $\rightarrow$ Kích hoạt đấu trường tranh luận có cấu trúc giữa **Bull Researcher** và **Bear Researcher** (tối đa 2 vòng để tối ưu chi phí token).
   - Nếu đồng thuận cao $\rightarrow$ Chuyển thẳng sang Chief Trader để tiết kiệm độ trễ và chi phí API.

3. **Chief Investment Officer (Chief Trader):**
   - Tổng hợp toàn diện các báo cáo và diễn biến tranh biện.
   - Ra quyết định rõ ràng: `BUY`, `SELL`, hoặc `HOLD` kèm điểm tự tin (Conviction Score 1-10), Stop Loss và Take Profit mục tiêu.

4. **Bộ Quản Trị Rủi Ro Cứng (Non-LLM Hard Risk Engine):**
   - **Tuyệt đối không để LLM tự quyết định kích thước lệnh hay bỏ qua rủi ro.**
   - Tự động tính toán vị thế (Position Sizing) theo khoảng cách Stop Loss và tỷ lệ rủi ro tài khoản (mặc định tối đa 2% vốn / lệnh, cap 15% quy mô danh mục).
   - Bắt buộc tỷ lệ Risk/Reward $\ge 1.5:1$ (tự động điều chỉnh Take Profit nếu không đạt).
   - **Circuit Breaker & Drawdown Protection:** Tự động ngắt mạch giao dịch khẩn cấp nếu drawdown trong ngày chạm ngưỡng 4% hoặc biến động thị trường vượt giới hạn.

5. **Bộ Nhớ Quyết Định Bền Vững (Memory & Audit DB):**
   - Lưu toàn bộ snapshot thị trường, niềm tin của từng agent, biên bản tranh luận và kết quả thực thi vào SQLite để audit và tối ưu.

6. **Web Dashboard Trực Quan Thời Gian Thực (FastAPI + WebSocket):**
   - Giao diện Dark-Mode Trading Terminal cao cấp.
   - Luồng streaming thời gian thực hiển thị từng bước suy nghĩ của agent, các lập luận đối đầu trực tiếp của Bull/Bear và biến động số dư tài khoản.

---

## 🚀 Cấu Hình Mô Hình Google Gemini Tối Ưu Quota

Dựa trên bảng hạn mức API của Google AI Studio, hệ thống thiết lập sẵn phân tầng mô hình thông minh:
- **Đội Analyst (RPM: 15, RPD: 500):** Sử dụng `Gemini 3.5 Flash Lite` hoặc `Gemini 2.5 Flash Lite` — tốc độ siêu nhanh, lượng request trong ngày cực lớn.
- **Chief Trader & Debate Arena (RPM: 5, RPD: 20):** Sử dụng `Gemini 2.5 Flash` hoặc `Gemini 3 Flash` — khả năng suy luận sắc bén, tổng hợp luận điểm sâu sắc.
- **Dry-Run / Mock Heuristic Mode:** Khi chưa thiết lập API key, hệ thống tự động kích hoạt MockLLM thông minh để kiểm thử toàn bộ luồng mà không tốn quota.

---

## 🛠 Hướng Dẫn Sử Dụng

### 1. Cài Đặt Môi Trường
```bash
pip install -r requirements.txt
```

### 2. Thiết Lập API Key (Tùy Chọn)
Tạo file `.env` từ `.env.example`:
```env
GEMINI_API_KEY="AIzaSy..."
MODEL_ANALYST="gemini-2.5-flash-lite"
MODEL_REASONING="gemini-2.5-flash"
```

### 3. Chạy 1 Chu Kỳ Phân Tích (Advisory Mode CLI)
```bash
python main.py --mode advisory --symbol BTC/USDT --timeframe 15m
```

### 4. Khởi Chạy Web Dashboard Trực Quan (Server Mode)
```bash
python main.py --mode server --port 8000
```
Mở trình duyệt tại: `http://localhost:8000`

### 5. Chạy Backtest Lịch Sử & Đo Chi Phí Token
```bash
python main.py --mode backtest --symbol BTC/USDT
```

### 6. Chạy Kiểm Thử Tự Động (Unit Tests)
```bash
python -m unittest discover tests
```
