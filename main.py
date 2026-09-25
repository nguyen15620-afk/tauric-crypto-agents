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
from storage.memory_db import MemoryDB
from data.ccxt_feed import CCXTMarketFeed

console = Console()

def run_advisory_cycle(symbol: str = "BTC/USDT", timeframe: str = "15m"):
    console.print(Panel(f"[bold cyan]KÍCH HOẠT CHU TRÌNH PHÂN TÍCH ĐA AGENT (CRYPTO ADVISORY)[/bold cyan]\n"
                        f"Cặp: [yellow]{symbol}[/yellow] | Khung: [yellow]{timeframe}[/yellow] | Model: [green]{settings.MODEL_REASONING}[/green]",
                        title="Tauric AI Multi-Agent Advisory", border_style="blue"))

    db = MemoryDB()
    graph = TradingAgentGraph()

    with console.status("[bold green]Đang thu thập dữ liệu & chạy hội đồng chuyên gia...[/bold green]"):
        state = graph.run_cycle(symbol=symbol, timeframe=timeframe)
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
    a_table.add_column("Agent Chuyên Môn", style="cyan")
    a_table.add_column("Thiên Hướng", style="bold")
    a_table.add_column("Độ Tin Cậy", justify="center")
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
    console.print(f"\n[bold yellow]3. Thẩm Định Bất Đồng (Consensus Validator):[/bold yellow]")
    console.print(f"Độ lệch ý kiến giữa các Analyst: [bold]{divergence:.2f}[/bold] (Ngưỡng tranh luận: 0.35)")
    
    if turns:
        console.print(Panel(
            "\n\n".join([f"[bold {'green' if 'Bull' in d.speaker else 'red'}]{d.speaker} (Vòng {d.round_num}):[/bold {'green' if 'Bull' in d.speaker else 'red'}]\n{d.argument}" for d in turns]),
            title="Đấu Trường Tranh Biện (Bull vs Bear Debate Arena)", border_style="yellow"
        ))
    else:
        console.print("[green]✓ Đạt độ đồng thuận cao giữa các Analyst. Bỏ qua vòng tranh luận.[/green]")

    # 4. Master Synthesized Recommendation (Chief Investment Officer)
    if decision:
        final_action = risk_val.final_action.value if risk_val else decision.action.value
        act_color = "green" if final_action == "BUY" else ("red" if final_action == "SELL" else "yellow")
        sl_val = risk_val.stop_loss if risk_val and risk_val.stop_loss > 0 else decision.stop_loss
        tp_val = risk_val.take_profit if risk_val and risk_val.take_profit > 0 else decision.take_profit
        alloc_pct = risk_val.approved_position_size_pct if risk_val else decision.suggested_position_size_pct

        details = [
            f"Khuyến Nghị Hành Động: [bold {act_color}]{final_action}[/bold {act_color}]  |  Độ Tự Tin: [bold]{decision.conviction}/10[/bold]",
            f"Giá Thị Trường: [bold]${snapshot.current_price:,.2f}[/bold]" if snapshot else "",
            f"Vùng Cắt Lỗ (Stop Loss): [red]${sl_val:,.2f}[/red]" if sl_val else "Vùng Cắt Lỗ: Không áp dụng",
            f"Vùng Chốt Lời (Take Profit): [green]${tp_val:,.2f}[/green]" if tp_val else "Vùng Chốt Lời: Không áp dụng",
            f"Tỷ Trọng Vốn Khuyến Nghị: [bold]{alloc_pct:.1f}%[/bold]" if alloc_pct > 0 else "",
            "",
            f"[bold]Luận Điểm Tổng Hợp:[/bold]\n{decision.rationale}",
        ]

        if decision.bull_case_summary:
            details.append(f"\n[green]▲ Góc nhìn Bull:[/green] {decision.bull_case_summary}")
        if decision.bear_case_summary:
            details.append(f"[red]▼ Rủi ro Bear:[/red] {decision.bear_case_summary}")
        if decision.risk_assessment:
            details.append(f"[yellow]⚠ Đánh giá rủi ro:[/yellow] {decision.risk_assessment}")

        if risk_val and risk_val.warnings:
            details.append(f"\n[bold yellow]Lưu Ý / Cảnh Báo An Toàn:[/bold yellow] {'; '.join(risk_val.warnings)}")

        console.print(Panel(
            "\n".join([d for d in details if d is not None]),
            title="4. KHUYẾN NGHỊ ĐẦU TƯ TỔNG HỢP (CHIEF INVESTMENT OFFICER)",
            border_style="green" if final_action == "BUY" else ("red" if final_action == "SELL" else "yellow")
        ))


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
    parser = argparse.ArgumentParser(description="Tauric AI Crypto Advisory System")
    parser.add_argument("--mode", choices=["advisory", "scan", "server"], default="advisory",
                        help="Chế độ chạy: advisory (tư vấn 1 mã coin), scan (quét coin tiềm năng nhất), server (web dashboard)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Cặp giao dịch (mặc định BTC/USDT)")
    parser.add_argument("--timeframe", default="15m", help="Khung thời gian nến (mặc định 15m)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng chạy server (mặc định 8000)")
    
    args = parser.parse_args()

    if args.mode == "advisory":
        run_advisory_cycle(symbol=args.symbol, timeframe=args.timeframe)
    elif args.mode == "scan":
        run_screener_mode()
    elif args.mode == "server":
        console.print(f"[bold green]Khởi chạy Web Dashboard tại http://localhost:{args.port}[/bold green]")
        uvicorn.run("server.app:app", host="0.0.0.0", port=args.port, reload=False)

if __name__ == "__main__":
    main()
