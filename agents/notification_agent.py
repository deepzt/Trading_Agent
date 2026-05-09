"""
Sends trade signal alerts via Telegram and daily P&L reports via email.
Telegram is primary; email is optional.
"""

from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from agents.base_agent import BaseAgent
from agents.signal_agent import Signal


class NotificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("NotificationAgent")
        self._bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        raw_ids = os.getenv("TELEGRAM_CHAT_ID", "")
        self._chat_ids = [cid.strip() for cid in raw_ids.split(",") if cid.strip()]
        self._email_sender = os.getenv("EMAIL_SENDER", "")
        self._email_password = os.getenv("EMAIL_PASSWORD", "")
        self._email_recipient = os.getenv("EMAIL_RECIPIENT", "")

    def run(self, signals: List[Signal], portfolio_stats: Optional[dict] = None):
        """Send notifications for approved signals and optional portfolio update."""
        approved = [s for s in signals if s.status == "APPROVED"]
        for signal in approved:
            self.send_signal_alert(signal)

        if portfolio_stats:
            self.send_portfolio_update(portfolio_stats)

    def send_signal_alert(self, signal: Signal, quantity: int = 0) -> bool:
        msg = self._format_signal_message(signal, quantity)
        sent = False

        if self._bot_token and self._chat_ids:
            sent = self._send_telegram(msg)

        if not sent:
            self.log_info(f"[SIGNAL] {signal.format_alert()}")

        return sent

    def send_eod_report(self, portfolio_stats: dict, closed_trades: list, date_str: str = "") -> bool:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        pnl = portfolio_stats.get("total_pnl", 0)
        equity = portfolio_stats.get("current_equity", 0)
        win_rate = portfolio_stats.get("win_rate", 0)

        closed_lines = []
        for t in closed_trades:
            emoji = "✅" if t.get("pnl", 0) >= 0 else "❌"
            closed_lines.append(f"{emoji} {t['symbol']}: {t['reason']} | P&L: ₹{t['pnl']:.2f}")

        msg = (
            f"📊 *EOD Report — {date_str}*\n\n"
            f"Portfolio Equity: ₹{equity:,.2f}\n"
            f"Total P&L: ₹{pnl:,.2f}\n"
            f"Win Rate: {win_rate}%\n"
            f"Sharpe: {portfolio_stats.get('sharpe', 0)}\n"
            f"Max Drawdown: {portfolio_stats.get('max_drawdown_pct', 0)}%\n\n"
        )
        if closed_lines:
            msg += "*Today's Trades:*\n" + "\n".join(closed_lines)

        return self._send_telegram(msg)

    def send_portfolio_update(self, stats: dict) -> bool:
        msg = (
            f"💼 *Portfolio Update*\n"
            f"Equity: ₹{stats.get('current_equity', 0):,.2f}\n"
            f"P&L: ₹{stats.get('total_pnl', 0):,.2f}\n"
            f"Trades: {stats.get('total_trades', 0)} | Win: {stats.get('win_rate', 0)}%"
        )
        return self._send_telegram(msg)

    def send_alert(self, message: str) -> bool:
        """Send a custom alert message."""
        return self._send_telegram(f"⚠️ *Trading Alert*\n{message}")

    # ── Private ─────────────────────────────────────────────────────────────

    def _format_signal_message(self, signal: Signal, quantity: int) -> str:
        emoji = "🟢" if signal.signal_type == "BUY" else "🔴"
        risk_info = ""
        if quantity > 0:
            risk_inr = quantity * abs(signal.entry_price - signal.stop_loss)
            profit_t2 = quantity * abs(signal.target_2 - signal.entry_price)
            risk_info = (
                f"\n📦 Qty: {quantity} shares | Risk: ₹{risk_inr:,.0f} | "
                f"Potential T2: ₹{profit_t2:,.0f}"
            )

        verdict_line = ""
        if signal.claude_verdict == "APPROVE" and signal.claude_reasoning:
            verdict_line = f"\n🤖 *AI*: {signal.claude_reasoning}"

        return (
            f"{emoji} *{signal.signal_type} Signal: {signal.symbol}*\n"
            f"Strategy: {signal.strategy.title()} | Confidence: {signal.confidence}/10\n\n"
            f"Entry: ₹{signal.entry_price:,.2f}\n"
            f"Stop Loss: ₹{signal.stop_loss:,.2f} ({signal.sl_pct}% risk)\n"
            f"Target 1: ₹{signal.target_1:,.2f}\n"
            f"Target 2: ₹{signal.target_2:,.2f}\n"
            f"Risk/Reward: {signal.risk_reward}x\n"
            f"{risk_info}"
            f"\n📋 *Reasons*: {' · '.join(signal.reasons[:3])}"
            f"{verdict_line}\n"
            f"🕐 {signal.timestamp[:19]}"
        )

    def _send_telegram(self, text: str) -> bool:
        if not self._bot_token or not self._chat_ids:
            self.log_warning("Telegram not configured — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            return False
        import requests
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        success = False
        for chat_id in self._chat_ids:
            try:
                resp = requests.post(url, json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                }, timeout=10)
                resp.raise_for_status()
                success = True
            except Exception as e:
                self.log_error(f"Telegram send failed for {chat_id}: {e}")
        return success

    def _send_email(self, subject: str, body: str) -> bool:
        if not self._email_sender or not self._email_password:
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self._email_sender
            msg["To"] = self._email_recipient
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self._email_sender, self._email_password)
                server.sendmail(self._email_sender, self._email_recipient, msg.as_string())
            return True
        except Exception as e:
            self.log_error(f"Email send failed: {e}")
            return False
