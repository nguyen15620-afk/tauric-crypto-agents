import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import uvicorn

from config.settings import settings
from agents.graph import TradingAgentGraph
from execution.paper_engine import PaperExecutionEngine
from storage.memory_db import MemoryDB
from backtest.runner import BacktestRunner
from data.ccxt_feed import CCXTMarketFeed

console = Console()

def run_advisory_cycle(symbol: str = "BTC/USDT", timeframe: str = "15m"):
    console.print(Panel(f"[bold cyan]KÍCH HOẠT CHU TRÌNH PHÂN TÍCH ĐA AGENT (CRYPTO TRADING)[/bold cyan]\n"
                        f"Cặp: [yellow]{symbol}[/yellow] | Khung: [yellow]{timeframe}[/yellow] | Model: [green]{settings.MODEL_REASONING}[/green]",
                        title="Tauric Multi-Agent System", border_style="blue"))

    db = MemoryDB()
    engine = PaperExecutionEngine(db=db)
    graph = TradingAgentGraph()

    with console.status("[bold green]Đang thu thập dữ liệu & chạy hội đồng chuyên gia...[/bold green]"):
        state = graph.run_cycle(symbol=symbol, timeframe=timeframe, portfolio_state=engine.get_state_dict())
        cycle_id = db.save_decision_cycle(state)

    snapshot = state.get("snapshot")
    reports = state.get("analyst_reports", [])
    divergence = state.get("divergence_score", 0.0)
    turns = state.get("debate_turns", [])
    decision = state.get("raw_decision")
    risk_val = state.get("risk_validation")

    # 1. Market Snapshot
    if snapshot:
        m_table = Table(title="1. Dữ Liệu Thị Trường & Chỉ Số Kỹ Thuật", border_style="cyan")
        m_table.add_column("Chỉ Số", style="dim")
        m_table.add_column("Giá Trị", style="bold")
        m_table.add_row("Giá Hiện Tại", f"${snapshot.current_price:,.2f} ({snapshot.change_24h_pct:+.2f}%)")
        m_table.add_row("RSI (14)", f"{snapshot.rsi_14}")
        m_table.add_row("MACD", f"{snapshot.macd:+.2f} (Signal: {snapshot.macd_signal:+.2f})")
        m_table.add_row("Bollinger Mid", f"${snapshot.bb_middle:,.2f}")
        m_table.add_row("ATR (14)", f"${snapshot.atr_14:,.2f}")
        m_table.add_row("Fear & Greed Index", f"{snapshot.fear_and_greed_score}/100 ({snapshot.fear_and_greed_label})")
        m_table.add_row("Sổ Lệnh Imbalance", f"{snapshot.orderbook_imbalance:+.2%}")
        console.print(m_table)

    # 2. Parallel Analysts
    a_table = Table(title="2. Báo Cáo Đội Chuyên Gia (Parallel Analysts)", border_style="magenta")
    a_table.add_column("Agent", style="cyan")
    a_table.add_column("Thiên Hướng", style="bold")
    a_table.add_column("Điểm Tin Cậy", justify="center")
    a_table.add_column("Tóm Tắt Nhận Định")
    
    for r in reports:
        bias_color = "green" if r.belief.bias == "BULLISH" else ("red" if r.belief.bias == "BEARISH" else "yellow")
        a_table.add_row(
            r.agent_name,
            f"[{bias_color}]{r.belief.bias} ({r.belief.score:+.2f})[/{bias_color}]",
            f"{int(r.belief.confidence * 100)}%",
            r.summary
        )
    console.print(a_table)

    # 3. Consensus & Debate
    console.print(f"\n[bold yellow]3. Thẩm Định Bất Đồng (Chivu171 Consensus Validator):[/bold yellow]")
    console.print(f"Độ lệch ý kiến: [bold]{divergence:.2f}[/bold] (Ngưỡng kích hoạt tranh luận: 0.35)")
    
    if turns:
        console.print(Panel(
            "\n\n".join([f"[bold {'green' if 'Bull' in d.speaker else 'red'}]{d.speaker} (Vòng {d.round_num}):[/bold {'green' if 'Bull' in d.speaker else 'red'}]\n{d.argument}" for d in turns]),
            title="Đấu Trường Tranh Luận (Bull vs Bear Debate Arena)", border_style="yellow"
        ))
    else:
        console.print("[green]✓ Đạt độ đồng thuận cao giữa các Analyst. Bỏ qua vòng tranh luận.[/green]")

    # 4. Chief Trader Decision
    if decision:
        act_color = "green" if decision.action.value == "BUY" else ("red" if decision.action.value == "SELL" else "yellow")
        console.print(Panel(
            f"Khuyến Nghị: [bold {act_color}]{decision.action.value}[/bold {act_color}] | Độ Tự Tin: [bold]{decision.conviction}/10[/bold]\n"
            f"Stop Loss: [red]${decision.stop_loss:,.2f}[/red] | Take Profit: [green]${decision.take_profit:,.2f}[/green]\n"
            f"Khối Lượng Đề Xuất: [bold]{decision.suggested_position_size_pct}%[/bold]\n\n"
            f"[italic]Luận điểm: {decision.rationale}[/italic]",
            title="4. Quyết Định Của Chief Trader (CIO)", border_style="green"
        ))

    # 5. Hard Risk Guardrails
    if risk_val:
        r_status = "[bold green]PHÊ DUYỆT[/bold green]" if risk_val.approved else "[bold red]TỪ CHỐI / ĐIỀU CHỈNH[/bold red]"
        console.print(Panel(
            f"Trạng Thái Thẩm Định Cứng: {r_status}\n"
            f"Lệnh Cho Phép: [bold]{risk_val.final_action.value}[/bold] | Cấp Vốn: [bold]{risk_val.approved_position_size_pct}% (${risk_val.approved_position_usd:,.2f})[/bold]\n"
            f"Cắt Lỗ: ${risk_val.stop_loss:,.2f} | Chốt Lời: ${risk_val.take_profit:,.2f}\n"
            f"Cảnh Báo: {', '.join(risk_val.warnings) if risk_val.warnings else 'Không có'}",
            title="5. Quản Trị Rủi Ro Cứng (Non-LLM Guardrails)", border_style="red"
        ))

    # Execute on paper
    if risk_val and snapshot:
        res = engine.execute_validation(risk_val, snapshot, cycle_id=cycle_id)
        console.print(f"[bold cyan]Kết quả khớp lệnh ảo (Paper Trading):[/bold cyan] {res}")

def run_backtest_mode(symbol: str = "BTC/USDT"):
    console.print(Panel(f"[bold green]CHẠY ENGINE BACKTEST LỊCH SỬ CHO {symbol}[/bold green]", border_style="green"))
    feed = CCXTMarketFeed()
    df = feed.fetch_ohlcv_df(symbol=symbol, timeframe="15m", limit=80)
    
    runner = BacktestRunner(initial_capital=10000.0)
    results = runner.run(df=df, symbol=symbol, step_interval=10, warmup_period=30)

    res_table = Table(title=f"KẾT QUẢ BACKTEST: {symbol}", border_style="green")
    res_table.add_column("Chỉ Số Đánh Giá", style="bold cyan")
    res_table.add_column("Giá Trị", style="bold")
    
    res_table.add_row("Số nến đã kiểm thử", str(results["total_bars_tested"]))
    res_table.add_row("Vốn khởi điểm", f"${results['initial_capital']:,.2f}")
    res_table.add_row("Vốn kết thúc (Equity)", f"${results['final_equity']:,.2f}")
    res_table.add_row("Lợi nhuận ròng (Net Return)", f"{results['net_return_pct']:+.2f}%")
    res_table.add_row("Max Drawdown (All-time)", f"{results['max_drawdown_pct']:.2f}%")
    res_table.add_row("Sharpe Ratio (Annualised)", f"{results.get('sharpe_ratio', 0):.3f}")
    res_table.add_row("Tổng số lệnh đóng", str(results["total_trades"]))
    res_table.add_row("Tỷ lệ thắng (Win Rate)", f"{results['win_rate_pct']:.1f}%")
    res_table.add_row("Tổng số token LLM ước tính", f"{results['estimated_tokens_used']:,} tokens")
    res_table.add_row("Chi phí API ước tính (Gemini Flash Lite)", f"${results['estimated_api_cost_usd']:.4f} USD")
    console.print(res_table)

def run_screener_mode():
    from data.screener import MarketScreener
    console.print(Panel("[bold cyan]QUÉT TOÀN THỊ TRƯỜNG TÌM MÃ COIN CÓ CƠ HỘI ĐỘT BIẾN CAO NHẤT[/bold cyan]", border_style="cyan"))
    
    with console.status("[bold green]Đang quét dữ liệu giá, khối lượng và động lượng trên Binance...[/bold green]"):
        screener = MarketScreener()
        top_coins = screener.scan_top_opportunities(top_n=5)

    table = Table(title="TOP CƠ HỘI TIỀM NĂNG NHẤT (MARKET SCREENER)", border_style="green")
    table.add_column("Hạng", justify="center", style="bold")
    table.add_column("Cặp Giao Dịch", style="bold cyan")
    table.add_column("Giá Hiện Tại", justify="right")
    table.add_column("Biến Động 24h", justify="right")
    table.add_column("Thanh Khoản (24h Vol)", justify="right")
    table.add_column("Dạng Tín Hiệu", style="yellow")
    table.add_column("Điểm Tiềm Năng", justify="center", style="bold green")

    for idx, c in enumerate(top_coins, 1):
        chg = c["change_24h_pct"]
        chg_style = "green" if chg >= 0 else "red"
        table.add_row(
            str(idx),
            c["symbol"],
            f"${c['current_price']:,.4f}" if c["current_price"] < 1 else f"${c['current_price']:,.2f}",
            f"[{chg_style}]{chg:+.2f}%[/{chg_style}]",
            f"${c['quote_volume_mil']:,.1f}M",
            c["signal_type"],
            str(c["opportunity_score"])
        )
    console.print(table)

    if top_coins:
        best_symbol = top_coins[0]["symbol"]
        console.print(f"\n[bold yellow]Tự động chuyển mã tiềm năng số 1 ({best_symbol}) cho Hội Đồng Đa Agent phân tích sâu:[/bold yellow]\n")
        run_advisory_cycle(symbol=best_symbol)

def main():
    parser = argparse.ArgumentParser(description="Crypto Multi-Agent Trading System")
    parser.add_argument("--mode", choices=["advisory", "server", "backtest", "scan"], default="advisory",
                        help="Chế độ chạy: advisory (1 mã), server (web dashboard), backtest (lịch sử), scan (tự tìm coin tiềm năng nhất)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Cặp giao dịch (mặc định BTC/USDT)")
    parser.add_argument("--timeframe", default="15m", help="Khung thời gian nến (mặc định 15m)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng chạy server (mặc định 8000)")
    
    args = parser.parse_args()

    if args.mode == "advisory":
        run_advisory_cycle(symbol=args.symbol, timeframe=args.timeframe)
    elif args.mode == "scan":
        run_screener_mode()
    elif args.mode == "backtest":
        run_backtest_mode(symbol=args.symbol)
    elif args.mode == "server":
        console.print(f"[bold green]Khởi chạy Web Dashboard tại http://localhost:{args.port}[/bold green]")
        uvicorn.run("server.app:app", host="0.0.0.0", port=args.port, reload=False)

if __name__ == "__main__":
    main()
